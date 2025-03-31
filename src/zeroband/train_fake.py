from collections import defaultdict
import os
from typing import TYPE_CHECKING, Literal

import torch
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy  # type: ignore
import wandb
import shardcast

from zeroband.models import AttnImpl, ModelName, ModelType, get_model_and_tokenizer
from zeroband.training.checkpoint import TrainingProgress, load_checkpoint_fsdp_state
from zeroband.training.data import DataConfig, get_dataloader
from zeroband.training.loss import grpo_loss, entropy_loss
from zeroband.training.lr_scheduler import get_scheduler
from zeroband.training.utils import apply_ac_ckpt

from zeroband.logger import get_logger

from pydantic_config import BaseConfig, parse_argv
from jaxtyping import Float

from zeroband.training.world_info import WorldInfo, get_world_info

from pydantic import model_validator

from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from torch._guards import log as torch_log
import logging


class AdamConfig(BaseConfig):
    type: Literal["adam"] = "adam"
    lr: float = 4e-4
    weight_decay: float = 0.01
    betas1: float = 0.9
    betas2: float = 0.99


class OptimConfig(BaseConfig):
    optim: AdamConfig = AdamConfig()
    sched_type: Literal["cosine", "linear", "wsd-sqrt"] = "cosine"
    warmup_steps: int = 1000
    stable_steps: int = 80_000
    total_steps: int = 88_000
    batch_size: int = 512

    step_per_rollout: int = 1


class TrainConfig(BaseConfig):
    micro_bs: int = 1
    ac_ckpt: bool | int = False
    reshard_after_forward: bool = True  # old shard grad op True mean full shard
    memory_profile: str | None = None
    torch_compile: bool = True
    liger_qwen: bool = False

    attn_impl: AttnImpl = "flex_attention"


class CkptConfig(BaseConfig):
    path: str | None = None
    interval: int | None = None
    resume: str | None = None

    rollout_path: str | None = None  # if rollout path is set we saved at each step


class Config(BaseConfig):
    name_model: ModelName = "150M"

    ckpt: CkptConfig = CkptConfig()

    project: str = "prime_simple"
    wandb: bool = True

    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    train: TrainConfig

    gpus_ids: list[int] | None = None

    temperature: float = 0.6  # todo remove this and add this to the data
    grpo_epsilon: float = 0.2
    entropy_loss_coeff: float = 0.001

    on_policy_log_prob: bool = False
    max_async_level: int = 2  # the amount of rollout checkpoints to keep

    @model_validator(mode="after")
    def check_liger(self):
        if self.train.liger_qwen:
            assert "Qwen" in self.name_model, "train.liger_qwen can only be applied to Qwen2 models."
        return self


def get_gradient_accumulation_steps(batch_size: int, micro_bs: int, data_workers: int, world_info: WorldInfo) -> int:
    assert batch_size % world_info.world_size == 0
    batch_size = batch_size // world_info.world_size

    print(f"batch_size: {batch_size}, micro_bs: {micro_bs}, data_workers: {data_workers}")
    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    assert batch_size % data_workers == 0, str(
        f"The batch size ({batch_size}) must be divisible by the number of data workers ({data_workers})."
    )

    return batch_size // micro_bs


def apply_fsdp(model: ModelType, reshard_after_forward: bool):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=None)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(transformer_block, mp_policy=mp_policy, reshard_after_forward=layer_reshard_after_forward)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def get_device_placement(gpus_ids: list[int] | None, world_info: WorldInfo) -> int:
    """handle using a subset of GPUs. Should work like the CUDA_VISIBLE_DEVICES env var.
    The reason we use this is because in the rl launcher, torch is initialized before the env var is set, so we cannot use the CUDA_VISIBLE_DEVICES env var.
    """
    if gpus_ids is None:
        return world_info.local_rank

    if world_info.local_rank >= len(gpus_ids):
        raise ValueError(f"Local rank {world_info.local_rank} is greater than the number of available GPUs ({len(gpus_ids)})")

    return gpus_ids[world_info.local_rank]


def train(config: Config):
    if "ZERO_BAND_DEV" not in os.environ:
        torch._logging.set_logs(dynamo=logging.CRITICAL)  # silent flex attn error
        torch_log.setLevel(logging.CRITICAL)  #

    logger = get_logger()
    world_info = get_world_info()

    logger.info(f"start training on {world_info.world_size}")

    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    torch.cuda.set_device(get_device_placement(config.gpus_ids, world_info))

    # batch_size is the total batch size for all GPUs

    # gradient_accumulation_steps = get_gradient_accumulation_steps(
    #     config.optim.batch_size, config.train.micro_bs, config.data.num_workers, world_info
    # )

    if config.ckpt.rollout_path is not None and world_info.rank == 0:
        origin_data_dir = os.environ.get("SHARDCAST_OUTPUT_DIR", "./origin_data")
        shardcast.initialize(origin_data_dir, max_distribution_folders=config.max_async_level)

    model, tokenizer = get_model_and_tokenizer(config.name_model, config.train.attn_impl)

    train_dataloader, prefetcher = get_dataloader(
        tokenizer=tokenizer,
        micro_batch_size=config.train.micro_bs,
        batch_size=config.optim.batch_size * config.optim.step_per_rollout,
        data_config=config.data,
    )

    if config.train.liger_qwen:
        apply_liger_kernel_to_qwen2(
            rope=True,
            rms_norm=True,
            swiglu=True,
            model=model,
        )

    if config.train.ac_ckpt:
        num = 1 if isinstance(config.train.ac_ckpt, bool) else config.train.ac_ckpt
        apply_ac_ckpt(model, num)

    apply_fsdp(model, config.train.reshard_after_forward)

    optimizer = torch.optim.AdamW(params=model.parameters(),lr=config.optim.optim.lr,weight_decay=config.optim.optim.weight_decay,betas=(config.optim.optim.betas1, config.optim.optim.betas2))  # fmt: skip

    scheduler = get_scheduler(sched_type=config.optim.sched_type,optimizer=optimizer,num_warmup_steps=config.optim.warmup_steps,num_stable_steps=config.optim.stable_steps,num_training_steps=config.optim.total_steps)  # fmt: skip

    training_progress = TrainingProgress(total_tokens=0, step=0)

    if world_info.rank == 0 and config.wandb:
        wandb.init(project=config.project, config=config.model_dump())

    if config.train.torch_compile:
        model = torch.compile(model) if not TYPE_CHECKING else model

    if config.ckpt.resume:
        load_checkpoint_fsdp_state(model, [optimizer], training_progress, train_dataloader, scheduler, config.ckpt.resume)

    losses = defaultdict(list)
    verl_losses = defaultdict(list)

    if config.train.memory_profile and world_info.rank == 0:
        torch.cuda.memory._record_memory_history()

    for _grad_acc_step in range(32):
        # Load args
        # batch = next(logprobs_aware_iterator)

        batch = torch.load(f"save_data/data_to_save_0_{_grad_acc_step}.pt")

        input_ids = batch["inputs_ids"].to("cuda")
        loss_mask = batch["attention_mask"]

        # Forward
        logits: Float[torch.Tensor, "batch seq vocab"] = model(input_ids=input_ids).logits.contiguous()

        # Gather args for grpo loss
        advantages = batch["advantages"].to("cuda")
        loss_mask = loss_mask.to("cuda")
        original_logprobs = batch["old_log_prob"].to("cuda")

        original_logprobs = original_logprobs[:, 1:]

        prompt_len = logits.shape[1] - advantages.shape[1]

        advantages = torch.cat([torch.zeros(advantages.shape[0], prompt_len).to("cuda"), advantages], dim=1)
        original_logprobs = torch.cat([torch.zeros(original_logprobs.shape[0], prompt_len).to("cuda"), original_logprobs], dim=1)

        # logger.info(f"HERE: logits: {logits.shape}, input_ids: {input_ids.shape}, advantages: {advantages.shape}, original_logprobs: {original_logprobs.shape}, loss_mask: {loss_mask.shape}")

        # Loss
        pg_loss, clip_ratio = grpo_loss(
            logits, input_ids, advantages, original_logprobs, loss_mask, config.temperature, config.grpo_epsilon
        )
        entropy = entropy_loss(logits, loss_mask, config.temperature)

        loss = pg_loss - config.entropy_loss_coeff * entropy

        losses["loss"].append(loss.item())
        losses["pg_loss"].append(pg_loss.item())
        losses["entropy"].append(entropy.item())
        losses["clip_ratio"].append(clip_ratio.item())

        verl_losses["loss"].append(batch["policy_loss"])
        verl_losses["pg_loss"].append(batch["pg_loss"])
        verl_losses["entropy"].append(batch["pg_clipfrac"])
        verl_losses["clip_ratio"].append(batch["pg_clipfrac"])

        del batch, logits, input_ids, advantages, loss_mask, original_logprobs

        del loss, clip_ratio, pg_loss, entropy

    for key in losses:
        losses[key] = torch.tensor(losses[key]).mean()
        verl_losses[key] = torch.tensor(verl_losses[key]).mean()
        diff = losses[key] - verl_losses[key]

        logger.info(f"{key}: {diff.max()=}, {diff.min()=}, {diff.mean()=}")

    # for key in losses:
    #     torch.testing.assert_close(losses[key], verl_losses[key])


if __name__ == "__main__":
    train(Config(**parse_argv()))
