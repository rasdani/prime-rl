from typing import Any, Generator, TypedDict

from pydantic_config import BaseConfig


import torch
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from jaxtyping import Float, Int


class DataConfig(BaseConfig):
    dataset_name_or_paths: str = "datasets/fineweb-edu"
    seq_length: int = 1024
    fake: bool = False
    num_workers: int = 4


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
            ref_logprobs = torch.randn(len_, self.vocab_size)
            self.step += 1
            yield {"input_ids": input_ids, "advantages": advantages, "ref_logprobs": ref_logprobs}
            # yield {"input_ids": input_ids}

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        itera = iter(self)
        for _ in range(self.step):
            next(itera)


class BatchOutput(TypedDict):
    input_ids: Int[torch.Tensor, "batch seq"]
    advantages: Float[torch.Tensor, "batch"]
    ref_logprobs: Float[torch.Tensor, "batch seq vocab"]


def get_dataloader(tokenizer, batch_size: int, data_config: DataConfig) -> StatefulDataLoader[BatchOutput]:
    """Get a dataloader for the training dataset"""
    if data_config.fake:
        train_dataset = FakeTokenizedDataset(data_config.seq_length, len(tokenizer))
    else:
        train_dataset = FakeTokenizedDataset(data_config.seq_length, len(tokenizer))

    return StatefulDataLoader(train_dataset, batch_size=batch_size, num_workers=data_config.num_workers)
