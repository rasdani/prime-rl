import pytest
from zeroband.training.data import ParquetDataset
from torch.utils.data import DataLoader


def test_pq_dataset(fake_rollout_files_dir):
    path = fake_rollout_files_dir(steps=[0, 1, 2, 3], num_files=4, batch_size=8)

    dataset = ParquetDataset(path, 8 * 4, timeout=2)

    dataloader = DataLoader(dataset, batch_size=10, num_workers=2)

    with pytest.raises(TimeoutError, match="Timeout waiting for step 4 to be created"):
        for _ in dataloader:
            ...
