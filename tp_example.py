
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
        parallelize_module(model, tp_mesh, tp_plan)
    
    
    if pd.dp_shard > 1:
        dp_mesh = world_mesh[("dp_shard_cp",)]
        for layer in model.layers:
            fully_shard(layer, mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)

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
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()), batch_size=64, shuffle=True)
    for epoch in range(num_epochs):
        epoch_loss = torch.tensor(0.0, device='cuda')
        for i, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_detached = loss.detach().clone()
            all_reduce_inplace(loss_detached, op="avg", group=world_mesh[("dp_shard_cp",)])
            loss_item = loss_detached.item()
            epoch_loss += loss_item
            print(f'\r\33[2K\033[0;32mEpoch [{epoch+1}/{num_epochs}]\033[0m, \033[0;31mLoss: {loss_item:.4f}\033[0m, \033[0;36mStep [{i+1}/{len(train_loader)}]\033[0m', end="")

        epoch_loss = epoch_loss.detach().clone()
        all_reduce_inplace(epoch_loss, op="avg", group=world_mesh[("dp_shard_cp",)])
        print(f"\r\33[2K\033[0;32mEpoch [{epoch+1}/{num_epochs}]\033[0m, \033[0;31mLoss: {epoch_loss.item()/len(train_loader):.4f}\033[0m")

train()