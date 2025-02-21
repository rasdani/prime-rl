from multiprocessing import Manager
from pathlib import Path
import time
from typing import Any, Generator, Protocol, TypedDict

from pydantic_config import BaseConfig


import torch
from torch.utils.data import IterableDataset, DataLoader

from jaxtyping import Float, Int

from pyarrow import parquet as pq

from zeroband.logger import get_logger


class DataConfig(BaseConfig):
    path: str = "datasets/fineweb-edu"
    seq_length: int = 1024
    fake: bool = False
    num_workers: int = 2


class UpdateableDataset(Protocol):
    def update_files(self, files: list[Path]): ...


class FakeTokenizedDataset(IterableDataset, UpdateableDataset):
    """This is a dummy dataset that generates random sequences of length seq_len and vocab_size"""

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        assert vocab_size > 3, "Vocab size must be greater than 3"
        self.step = 0

    def update_files(self, files: list[Path]): ...

    def __iter__(self) -> Generator[dict[str, Any], Any, None]:
        while True:
            len_ = self.seq_len
            input_ids = torch.randint(3, self.vocab_size, (len_,))
            advantages = torch.randn(len_)
            self.step += 1
            yield {"input_ids": input_ids, "advantages": advantages}


class ParquetDataset(IterableDataset):
    """
    This call is a wrapper around parquet dataset.

    It can be updated by calling update_files with a list of files. This will thrown away all previous files.

    If the dataset is exhausted, it will wait for new files to be added.
    """

    def __init__(self):
        self._logger = get_logger()

        self._manager = Manager()
        self._shared_files = self._manager.list()

    def update_files(self, files: list[Path]):
        # Update the shared files list

        self._shared_files[:] = files  # Use slice assignment to update manager list
        self._logger.info(f"Updating files with {len(files)} files")

    def __iter__(self):
        while True:
            worker_info = torch.utils.data.get_worker_info()
            files = self._shared_files[worker_info.id :: worker_info.num_workers]  # split by worker
            self._logger.info(f"Worker {worker_info.id} has {len(files)} files. shared files: {len(self._shared_files)}")
            for file in files:
                table = pq.ParquetFile(str(file)).read()
                required_columns = ["output_tokens", "advantages"]

                skip = False
                for column in required_columns:
                    if column not in table.column_names:
                        skip = True
                        self._logger.warning(f"File {file} missing required column '{column}'")

                if not skip:
                    output_tokens = table["output_tokens"]
                    advantages = table["advantages"]

                    for ids, adv in zip(output_tokens, advantages):
                        ids = torch.tensor(ids.as_py())
                        adv = torch.tensor(
                            [adv.as_py()] * len(ids)
                        )  # advantes at inference is per sample, but here we want it to be per tokens
                        yield {"input_ids": ids, "advantages": adv}  #

            self._logger.info("Waiting for new files")
            time.sleep(0.5)


class BatchOutput(TypedDict):
    input_ids: Int[torch.Tensor, "batch seq"]
    advantages: Float[torch.Tensor, "batch seq"]


def get_dataloader(tokenizer, batch_size: int, data_config: DataConfig) -> tuple[DataLoader[BatchOutput], UpdateableDataset]:
    """Get a dataloader for the training dataset"""
    if data_config.fake:
        train_dataset = FakeTokenizedDataset(data_config.seq_length, len(tokenizer))
    else:
        train_dataset = ParquetDataset()

    return DataLoader(train_dataset, batch_size=batch_size, num_workers=data_config.num_workers), train_dataset
