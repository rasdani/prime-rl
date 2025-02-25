import pytest
from zeroband.training.data import ParquetDataset, _should_skip_index
from torch.utils.data import DataLoader


def test_pq_dataset(fake_rollout_files_dir):
    path = fake_rollout_files_dir(steps=[0, 1, 2, 3], num_files=4, batch_size=8)

    dataset = ParquetDataset(path, 8 * 4, timeout=2)

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
        if not _should_skip_index(index, 0, world_size, rank, num_workers, workers_id):
            results.append(index)

    assert results == expected_results
