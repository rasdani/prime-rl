from pathlib import Path
import time
from typing import Any, Generator, TypedDict

from pydantic_config import BaseConfig


import torch
from torch.utils.data import IterableDataset, DataLoader

from jaxtyping import Float, Int

from pyarrow import dataset as ds

from zeroband.logger import get_logger
from zeroband.training.world_info import get_world_info


STABLE_FILE = "stable"


class DataConfig(BaseConfig):
    path: str = "datasets/fineweb-edu"
    seq_length: int = 1024
    fake: bool = False
    num_workers: int = 2

    batch_size: int  # will be set by the top config


class FakeTokenizedDataset(IterableDataset):
    """This is a dummy dataset that generates random sequences of length seq_len and vocab_size"""

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        assert vocab_size > 3, "Vocab size must be greater than 3"
        self.step = 0

    def __iter__(self) -> Generator[dict[str, Any], Any, None]:
        while True:
            len_ = self.seq_len
            input_ids = torch.randint(3, self.vocab_size, (len_,))
            advantages = torch.randn(len_)
            self.step += 1
            yield {"input_ids": input_ids, "advantages": advantages}


def _get_all_files_for_step(step_count: int, path: Path, timeout: float) -> list[Path]:
    """Get all the files for a given step. Waits until the step is created which is indicated by the stable file."""
    logger = get_logger()
    step_path = path / f"step_{step_count}"
    stable_file = step_path / STABLE_FILE

    start_time = time.time()
    while not stable_file.exists():
        if time.time() - start_time > timeout:
            logger.info("raising timeout")
            raise TimeoutError(f"Timeout waiting for step {step_count} to be created")

        logger.info(f"Waiting for step {step_count} to be created")
        time.sleep(0.5)

    files = list(step_path.glob("*.parquet"))
    return files


class ParquetDataset(IterableDataset):
    """
    This call is a wrapper around parquet dataset.

    It can be updated by calling update_files with a list of files. This will thrown away all previous files.

    If the dataset is exhausted, it will wait for new files to be added.
    """

    def __init__(self, path: Path, batch_size: int, pq_read_bs: int = 64, timeout: float = 360):
        self._logger = get_logger()
        self._path = path
        self._batch_size = batch_size
        self._pq_read_bs = pq_read_bs

        self._world_info = get_world_info()

        self._step_count = -1
        self._timeout = timeout

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        assert self._batch_size % (self._world_info.world_size * worker_info.num_workers) == 0, (
            "Batch size must be divisible by the number of workers time the world size"
        )
        # this assert should never be triggered because we check for it in the top config level. Keep it here for sanity

        target_sample_count_per_batch = self._batch_size // (self._world_info.world_size * worker_info.num_workers)

        while True:
            self._step_count += 1

            sample_count = 0

            self._logger.info(f"data: Processing step {self._step_count}")

            files = _get_all_files_for_step(self._step_count, self._path, self._timeout)

            # we are NOT splitting the files across datalaoder workers and rank like we did for intellect 1
            # This is because we cannot assume that the files would have the same number of samples each.
            # What we rather do here is that all the workers go over all the files and only yield some of them
            # this is unoptimal because they all load more data that they should, but since the data is already tokenized it should not be a big deal

            dataset = ds.dataset(files, format="parquet")
            # Set up a scanner with just the required columns
            required_columns = ["output_tokens", "advantages"]

            scanner = dataset.scanner(columns=required_columns, batch_size=self._pq_read_bs)

            self._logger.info(f"step {self._step_count} scanner: {scanner},sample_count: {sample_count}")

            for batch in scanner.to_batches():
                # Check if both required columns exist in this batch
                if all(col in batch.column_names for col in required_columns):
                    output_tokens = batch["output_tokens"]
                    advantages = batch["advantages"]
                    for i, (token, advantage) in enumerate(zip(output_tokens, advantages)):
                        try:
                            ids = torch.tensor(token.as_py())
                            adv = torch.tensor(data=[advantage.as_py()] * len(ids))
                            data = {"input_ids": ids, "advantages": adv}
                        except Exception as e:
                            self._logger.warn(f"Error processing row {i} sample {sample_count}: {str(e)}")
                            data = None

                        if data is not None:
                            yield data
                            sample_count += 1

                        if sample_count >= target_sample_count_per_batch:
                            break

                if sample_count >= target_sample_count_per_batch:
                    # need to break out of a second time because of the nested for loop
                    break


class BatchOutput(TypedDict):
    input_ids: Int[torch.Tensor, "batch seq"]
    advantages: Float[torch.Tensor, "batch seq"]


def get_dataloader(tokenizer, batch_size: int, data_config: DataConfig) -> DataLoader[BatchOutput]:
    """Get a dataloader for the training dataset"""
    if data_config.fake:
        train_dataset = FakeTokenizedDataset(data_config.seq_length, len(tokenizer))
    else:
        train_dataset = ParquetDataset(Path(data_config.path), data_config.batch_size)

    return DataLoader(train_dataset, batch_size=batch_size, num_workers=data_config.num_workers)
