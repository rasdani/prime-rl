from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from zeroband.logger import get_logger
import socket
import time
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
)

from zeroband.models import ModelType
from zeroband.training.world_info import get_world_info


from typing import Iterable, Optional, Union

import math
import torch.nn.utils

import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh

from torch.nn.utils.clip_grad import _get_total_norm, _clip_grads_with_norm_


@torch.no_grad()
def clip_grad_norm_(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    pp_mesh: Optional[DeviceMesh] = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm. # Added this line
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    grads = [p.grad for p in parameters if p.grad is not None]
    grads = [g.full_tensor() if isinstance(g, DTensor) else g for g in grads] # Added this line, not present in torchtitan
    # for p in parameters:
    #     if p.grad is not None:
    #         continue
    #     g = p.grad
    #     if not g.device_mesh == grads[0].device_mesh:
    #         print(f"Got incompatible meshes: {g.device_mesh} and {grads[0].device_mesh}")
    #         print("Original tensor shape:")
    # exit()

    # TODO: Figure out how they're sharded and try to reduce communication
    total_norm = _get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.

        # Remove FT replicate dimension if it exists.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def apply_ac_ckpt(model: ModelType, num: int):
    """Apply activation checkpointing to the model.
    Apply to layers multiple of `num`.

    Example if `num=2` only half of the layers are checkpointed.
    """
    logger = get_logger()

    layers_ckpt = 0

    for layer_id, transformer_block in model.model.layers.named_children():
        if layers_ckpt % num == 0:
            transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
            model.model.layers.register_module(layer_id, transformer_block)
            layers_ckpt += 1

    logger.debug(f"Applied activation checkpointing to {layers_ckpt} layers")


### code above inspired and copied from https://github.com/pytorch/torchtitan/blob/4b3f2e41a084bf79a8540068ed525539d1244edd/torchtitan/utils.py#L119


# hardcoded BF16 type peak flops for NVIDIA A100 and H100 GPU
def get_peak_flops(device_name: str) -> int:
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    else:  # for other GPU types, assume A100
        return 312e12


def get_num_flop_per_token(num_params: int, model_config: LlamaConfig, seq_len: int) -> int:
    l, h, q, t = (  # noqa: E741
        model_config.num_hidden_layers,
        model_config.num_attention_heads,
        model_config.hidden_size // model_config.num_attention_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token


def get_num_params(model: ModelType, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.lm_head.weight.numel()
    return num_params


class PerfCounter:
    """A class to count tokens per second with a rolling window.
    we use a rollowing window because time perf counter is not precise enough in some case
    """

    def __init__(self, window_size: int, model: LlamaForCausalLM, seq_len: int):
        self.window_size = window_size
        self.tokens = []
        self.times = []
        self.model = model

        self.gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(torch.device("cuda")))
        self.num_params = get_num_params(model, exclude_embedding=True)
        self.num_flop_per_token = get_num_flop_per_token(self.num_params, model.config, seq_len=seq_len)

        self._world_info = get_world_info()

    def count_tokens(self, tokens: int):
        self.tokens.append(tokens)
        self.times.append(time.perf_counter())
        if len(self.tokens) > self.window_size:
            self.tokens.pop(0)
            self.times.pop(0)

    def get_tokens_per_second(self) -> float | None:
        if len(self.tokens) < 2:
            return None
        return sum(self.tokens[1:]) / (self.times[-1] - self.times[0])

    def get_mfu(self) -> float | None:
        tokens_per_second = self.get_tokens_per_second()
        if tokens_per_second is None:
            return None
        return 100 * self.num_flop_per_token * tokens_per_second / self.gpu_peak_flops / self._world_info.world_size


def get_random_available_port_list(num_port):
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    ports = []

    while len(ports) < num_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            new_port = s.getsockname()[1]

        if new_port not in ports:
            ports.append(new_port)

    return ports


def get_random_available_port():
    return get_random_available_port_list(1)[0]


class FakeTokenizer(object):
    def __init__(self):
        self.vocab_size = 1000
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

    def __len__(self):
        return self.vocab_size
