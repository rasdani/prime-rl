import pytest
from zeroband.training.data import (
    ParquetDataset,
    _should_skip_index,
    collate_packing,
    FakeTokenizedDataset,
    pack_datatset_outputs_efficiently,
    collate_fn_padding,
)
from torch.utils.data import DataLoader


def test_pq_dataset(fake_rollout_files_dir):
    path = fake_rollout_files_dir(steps=[0, 1, 2, 3], num_files=4, batch_size=8)

    dataset = ParquetDataset(path, 8 * 4, timeout=2, step_count_init=0, ignore_zero_advantages=False)

    dataloader = DataLoader(dataset, batch_size=10, num_workers=2)

    with pytest.raises(TimeoutError, match="Timeout waiting for step 4 to be created"):
        for _ in dataloader:
            ...


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("workers_id", [0, 1, 2, 3])
def test_should_skip_index(rank, workers_id):
    world_size = 4
    num_workers = 4

    full_index = list(range(100))

    expected_results = full_index[rank::world_size][workers_id::num_workers]

    results = []
    for index in full_index:
        # If we should not skip this index, add it to results
        if not _should_skip_index(index, world_size, rank, num_workers, workers_id):
            results.append(index)

    assert results == expected_results


def test_pack_datatset_outputs_efficiently():
    BS = 16

    batch = []

    dataset = FakeTokenizedDataset(64, 128)

    for i in range(BS):
        batch.append(next(iter(dataset)))

    packed_batch = pack_datatset_outputs_efficiently(batch, 64)

    assert len(packed_batch) >= 1


def test_pack_bin_packing():
    bin_size = 3
    SEQ_LEN = 64

    bin = []

    dataset = FakeTokenizedDataset(seq_len=SEQ_LEN, vocab_size=128)

    for i in range(bin_size):
        bin.append(next(iter(dataset)))

    micro_batch = collate_packing(bin, 2048, 128)

    assert micro_batch["input_ids"].shape == (1, 2048)


def test_collate_fn_padding():
    micro_bs = 16
    SEQ_LEN = 64

    dataset = FakeTokenizedDataset(seq_len=SEQ_LEN, vocab_size=128)

    batch = []

    for i in range(micro_bs):
        batch.append(next(iter(dataset)))

    batch = collate_fn_padding(batch, SEQ_LEN, 128)

    assert batch["input_ids"].shape == (micro_bs, SEQ_LEN)
    assert batch["position_ids"].shape == (micro_bs, SEQ_LEN)
