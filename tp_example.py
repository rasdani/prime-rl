
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.distributed.fsdp import fully_shard

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    #SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from zeroband.training.parallel_dims import ParallelDims

from typing import Any
import pickle

from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import IterableDataset

from typing import Iterable, Optional, Union

import math
import torch
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
    assert all(isinstance(g, DTensor) for g in grads)
    print(all([g.device_mesh == grads[0].device_mesh for g in grads]))
    #grads = [g.full_tensor() if isinstance(g, DTensor) else g for g in grads] # Added this line
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

class ParallelAwareDataloader(StatefulDataLoader):
    """Dataloader that is aware of distributed data parallelism.

    This dataloader is used to load data in a distributed data parallel fashion. It also
    utilizes ``torchdata.stateful_dataloader.StatefulDataLoader`` to implement the necessary
    methods such as ``__iter__``.

    Args:
        dataset (IterableDataset): The dataset to iterate over.
        dp_rank: Data parallelism rank for this dataloader.
        dp_world_size: The world size of the data parallelism.
        batch_size: The batch size to use for each iteration.
    """

    dp_rank: int
    dp_world_size: int
    batch_size: int

    def __init__(
        self,
        dataset: IterableDataset,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
    ):
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        super().__init__(dataset, batch_size)
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions.
        return {
            # We don't have to use pickle as DCP will serialize the state_dict. However,
            # we have to keep this for backward compatibility.
            self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # State being empty is valid.
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            print(
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                "expected key {self._rank_id}"
            )
            return

        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        # We don't have to use pickle as DCP will serialize the state_dict. However, we have to
        # keep this for backward compatibility.
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def train():
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(512, 1024)
            self.fc2 = nn.Linear(1024, 512)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    class MNISTModel(nn.Module):
        def __init__(self):
            super(MNISTModel, self).__init__()
            self.embed = nn.Linear(784, 512)
            self.layers = nn.ModuleList([MLP() for _ in range(3)])
            self.head = nn.Linear(512, 10)

        def forward(self, x):
            x = x.view(-1, 784)
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            x = self.head(x)
            return x

    
    pd = ParallelDims(
        dp_replicate=1,
        dp_shard=2,
        cp=1,
        tp=4,
        pp=1,
        world_size=8,
        enable_loss_parallel=False,
    )
    world_mesh = pd.build_mesh("cuda")

    tp_plan = {
        "embed": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(0)),
        "layers.*.fc1": ColwiseParallel(input_layouts=Shard(0), output_layouts=Shard(1)),
        "layers.*.fc2": RowwiseParallel(input_layouts=Shard(1), output_layouts=Shard(0)),
        "head": RowwiseParallel(input_layouts=Shard(0), output_layouts=Replicate()),
    }


    model = MNISTModel()

    if pd.tp > 1:
        tp_mesh = world_mesh["tp"]
        tp_rank = tp_mesh.get_local_rank()
        print(f"tp_mesh: {tp_mesh}, tp_rank: {tp_rank}")
        parallelize_module(model, tp_mesh, tp_plan)
    else:
        tp_rank = 0
    
    
    if pd.dp_shard > 1:
        dp_mesh = world_mesh[("dp_shard_cp",)]
        dp_rank = dp_mesh.get_local_rank()
        print(f"dp_mesh: {dp_mesh}, dp_rank: {dp_rank}")
        for layer in model.layers:
            fully_shard(layer, mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)
    else:
        dp_rank = 0

    model = model.to('cuda')

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-6,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )

    from torch.distributed._functional_collectives import all_reduce_inplace

    num_epochs = 50
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    #mnist_dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)
    train_loader = ParallelAwareDataloader(mnist_dataset, dp_rank, dp_world_size=pd.dp_shard, batch_size=64)
    for epoch in range(num_epochs):
        epoch_loss = torch.tensor(0.0, device='cuda')
        for i, (inputs, labels) in enumerate(train_loader):
            if pd.dp_shard > 1:
                model.set_requires_gradient_sync(True)
                pass

            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            #print(f"Labels on rank {(dp_rank, tp_rank)}: {labels}")

            #print(f"Creating fake data on ranks {(dp_rank, tp_rank)}")
            #torch.manual_seed(dp_rank) # Ensure data is the same across TP ranks.
            #inputs = torch.randn(inputs.shape, dtype=inputs.dtype, device="cuda")
            #labels = torch.randn(labels.shape, dtype=inputs.dtype, device="cuda")

            #print(f"Running forward pass on ranks {(dp_rank, tp_rank)}")
            outputs = model(inputs)

            #print(f"Calculating loss on ranks {(dp_rank, tp_rank)}")
            loss = loss_fn(outputs, labels)


            #print(f"Backward pass on ranks {(dp_rank, tp_rank)}")
            loss.backward()

            grad_norm = clip_grad_norm_(model.parameters(), 1.0)

            #print(f"Stepping optimizer on ranks {(dp_rank, tp_rank)}")
            optimizer.step()
            optimizer.zero_grad()

            #print(f"Allreducing loss on ranks {(dp_rank, tp_rank)}")
            loss_detached = loss.detach().clone()
            all_reduce_inplace(loss_detached, op="avg", group=world_mesh[("dp_shard_cp",)])
            loss_item = loss_detached.item()
            epoch_loss += loss_item
            print(f'\r\33[2K\033[0;32mEpoch [{epoch+1}/{num_epochs}]\033[0m, \033[0;31mLoss: {loss_item:.4f}\033[0m, \033[0;36mStep [{i+1}/{len(train_loader)}]\033[0m', end="")

        epoch_loss = epoch_loss.detach().clone()
        all_reduce_inplace(epoch_loss, op="avg", group=world_mesh[("dp_shard_cp",)])
        print(f"\r\33[2K\033[0;32mEpoch [{epoch+1}/{num_epochs}]\033[0m, \033[0;31mLoss: {epoch_loss.item()/len(train_loader):.4f}\033[0m")

train()