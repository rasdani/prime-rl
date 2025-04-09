import pytest
import torch
from zeroband.training.data import packed_batch, FakeTokenizedDataset, DatasetOutput, BatchOutput, IterableDataset

def test_packed_batch():
    vocab_size = 512
    seq_len = 2048
    micro_bs = 1
    padding = 0

    torch.manual_seed(0)
    dataset: IterableDataset = FakeTokenizedDataset(seq_len, vocab_size)
    dataset_iter = iter(dataset)

    batches: list[DatasetOutput] = [next(dataset_iter) for _ in range(10)]
    packed: tuple[list[BatchOutput], int] = packed_batch(batches, seq_len, padding, micro_bs)
    microbatches, n_microbatches = packed

    assert n_microbatches == 6