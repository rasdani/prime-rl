from zeroband.training.data import ParquetDataset
from torch.utils.data import DataLoader
import os


class FakeLogger:
    """
    A fake logger that stores the logs in a file
    """

    def __init__(self, tmp_file, original_logger):
        self.tmp_file = tmp_file
        self.original_logger = original_logger

    def info(self, msg):
        self.original_logger.info(msg)
        with open(self.tmp_file, "a") as f:
            f.write(f"INFO: {msg}\n")

    def warning(self, msg):
        self.original_logger.warning(msg)
        with open(self.tmp_file, "a") as f:
            f.write(f"WARNING: {msg}\n")


def test_pq_dataset(fake_rollout_files_dir, caplog, tmp_path):
    path = fake_rollout_files_dir(step=0, num_files=10) / "step_0"

    dataset = ParquetDataset()
    tmp_file = tmp_path / "tmp_log.txt"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    dataset._logger = FakeLogger(tmp_file, dataset._logger)
    dataloader = DataLoader(dataset, batch_size=10, num_workers=2)

    files = [path / f for f in os.listdir(path)]
    assert len(files) == 10
    dataset.update_files(files)

    for _ in dataloader:
        break

    logs = open(tmp_file).read()
    assert "Worker 0 has 5 files. shared files: 10" in logs
    assert "Worker 1 has 5 files. shared files: 10" in logs
