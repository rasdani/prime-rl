import pytest
from zeroband.training.data import (
    ParquetDataset,
    _should_skip_index,
    collate_packing,
    FakeTokenizedDataset,
    pack_datatset_outputs_efficiently,
    packed_batch,
)
from torch.utils.data import DataLoader

import torch


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


def test_pack_dataset_2():
    BS = 16
    SEQ_LEN = 2048

    batch = []

    for i in range(BS):
        seq_len = SEQ_LEN - 1
        input_ids = torch.randint(3, 128, (seq_len,))
        advantages = torch.randn(seq_len)
        batch.append(
            {
                "input_ids": input_ids,
                "advantages": advantages,
                "rewards": 0.5,
                "loss_mask": torch.ones(seq_len).int(),
                "logprobs": torch.randn(seq_len),
            }
        )
    packed_batch = pack_datatset_outputs_efficiently(batch, max_seq_len=seq_len)

    assert len(packed_batch) == BS


def test_pack_bin_packing():
    bin_size = 3
    SEQ_LEN = 64

    bin = []

    dataset = FakeTokenizedDataset(seq_len=SEQ_LEN, vocab_size=128)

    for i in range(bin_size):
        bin.append(next(iter(dataset)))

    micro_batch = collate_packing(bin, 2048, 128)

    assert micro_batch["input_ids"].shape == (1, 2048)


def test_packing_vs_padding():
    """
    Here we test that we don't lose any rewards or data when doing the different packing modes
    """

    BS = 32
    MICRO_BS = 4
    SEQ_LEN = 64

    batch_rollout = []

    for seq_len in [SEQ_LEN, SEQ_LEN // 8]:
        for i in range(BS // 2):
            data = {
                "input_ids": torch.ones(seq_len).int(),
                "advantages": torch.ones(seq_len),
                "loss_mask": torch.ones(seq_len).int(),
                "logprobs": torch.ones(seq_len),
                "seq_lens": torch.ones(seq_len),
                "rewards": torch.ones(1),
                "task_rewards": torch.ones(1),
                "length_penalties": torch.ones(1),
                "target_lengths": torch.ones(1),
            }

            batch_rollout.append(data)

    batch_packed = packed_batch(batch_rollout, max_seq_len=SEQ_LEN, packing_mode="packing", micro_bs=MICRO_BS, pad_token_id=0)
    batch_padded = packed_batch(batch_rollout, max_seq_len=SEQ_LEN, packing_mode="padding", micro_bs=MICRO_BS, pad_token_id=0)

    total_rewards_packed = sum(batch["rewards"].sum().item() for batch in batch_packed)
    total_rewards_padded = sum(batch["rewards"].sum().item() for batch in batch_padded)

    assert total_rewards_packed == total_rewards_padded

    total_input_ids_packed = sum(batch["input_ids"].sum().item() for batch in batch_packed)
    total_input_ids_padded = sum(batch["input_ids"].sum().item() for batch in batch_padded)

    assert total_input_ids_packed == total_input_ids_padded

    total_padded_tokens_packed = (
        sum(batch["input_ids"].shape[0] * batch["input_ids"].shape[1] for batch in batch_packed) - total_input_ids_packed
    )
    total_padded_tokens_padded = (
        sum(batch["input_ids"].shape[0] * batch["input_ids"].shape[1] for batch in batch_padded) - total_input_ids_padded
    )

    assert total_padded_tokens_packed < total_padded_tokens_padded
