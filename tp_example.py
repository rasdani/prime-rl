
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

    
    world_mesh = ParallelDims(
        dp_replicate=1,
        dp_shard=2,
        cp=1,
        tp=4,
        pp=1,
        world_size=8,
        enable_loss_parallel=False,
    ).build_mesh("cuda")

    tp_plan = {
        "embed": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(0), use_local_output=True),
        "layers.*.fc1": ColwiseParallel(input_layouts=Shard(0), output_layouts=Shard(1), use_local_output=True),
        "layers.*.fc2": RowwiseParallel(input_layouts=Shard(1), output_layouts=Shard(0), use_local_output=True),
        "head": RowwiseParallel(input_layouts=Shard(0), output_layouts=Replicate(), use_local_output=True),
    }


    model = MNISTModel()

    tp_mesh = world_mesh["tp"]
    parallelize_module(model, tp_mesh, tp_plan)
    
    dp_mesh = world_mesh[("dp_shard_cp",)]
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)

    model = model.to('cuda')

    loss_fn = nn.CrossEntropyLoss()


    # Arguments are the same as torch.optim.Adam/AdamW.
    # Torch's AdamW implementation is substantially different from the original paper,
    # while Adam is the same. We have implemented all of them.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-6,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )

    num_epochs = 50
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()), batch_size=64, shuffle=True)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            # Forward as normal
            outputs = model(inputs)

            # Calculate loss as normal
            loss = loss_fn(outputs, labels)

            # With a pipeline hook, as grads become available during backward() the optimizer step runs asynchronously on CPU.
            # Overlapping the optimizer step with backward() in this way leads to large speed improvements.
            # If you didn't define a pipeline hook, no changes.
            loss.backward()

            # If you're using a pipeline hook, the step() function waits for the async optimizer step queued by backward() to finish.
            # If you aren't using a pipeline hook, optimizer.step() behaves as normal.
            optimizer.step()

            # Zero grad as normal.
            optimizer.zero_grad()

            # Everything else is unchanged.
            epoch_loss += loss.item()
            print(f'\r\33[2K\033[0;32mEpoch [{epoch+1}/{num_epochs}]\033[0m, \033[0;31mLoss: {loss.item():.4f}\033[0m, \033[0;36mStep [{i+1}/{len(train_loader)}]\033[0m', end="")
        print(f"\r\33[2K\033[0;32mEpoch [{epoch+1}/{num_epochs}]\033[0m, \033[0;31mLoss: {epoch_loss/len(train_loader):.4f}\033[0m")

train()