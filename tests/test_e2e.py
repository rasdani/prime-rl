from zeroband.models import get_model_and_tokenizer

from zeroband.training.data import packed_batch, FakeTokenizedDataset
from zeroband.training.loss import grpo_loss, selective_log_softmax
import torch

ERROR_ATOL = {
    torch.float: 3e-4,
    torch.half: 4e-3,
    torch.bfloat16: 2e-2,
}
ERROR_RTOL = {
    torch.float: 2e-5,
    torch.half: 4e-4,
    torch.bfloat16: 5e-3,
}


def compute_states(batch, model):
    with torch.no_grad():
        temperature = 0.6
        input_ids = batch["input_ids"].to("cuda")
        logits = model(input_ids=input_ids, position_ids=batch["position_ids"].to("cuda")).logits.contiguous()

        input_ids_2 = input_ids[:, 1:]
        logits_2 = logits[:, :-1, :] / temperature

        per_token_logps = selective_log_softmax(logits_2, input_ids_2)

        pg_loss, clip_ratio = grpo_loss(
            logits,
            input_ids,
            batch["advantages"].to("cuda"),
            per_token_logps,
            batch["loss_mask"].to("cuda"),
            temperature,
            0.2,
            0.2,
            None,
        )

        return {"pg_loss": pg_loss, "clip_ratio": clip_ratio, "batch": batch}


def test_e2e():
    model, tokenizer = get_model_and_tokenizer("PrimeIntellect/llama-2m-fresh", "sdpa")
    model = model.to("cuda")
    assert model is not None
    assert tokenizer is not None

    SEQ_LEN = 64
    # create a fake dataset
    dataset = FakeTokenizedDataset(seq_len=SEQ_LEN, vocab_size=128)

    pad_token_id = 0

    batch_size = 16
    batch = []
    for i in range(batch_size):
        batch.append(next(iter(dataset)))

    micro_bs = 4

    batch_padded = packed_batch(batch, SEQ_LEN, pad_token_id, micro_bs, collate_mode="padding")

    loss_padded = 0
    for micro_batch in batch_padded:
        loss = compute_states(micro_batch, model)["pg_loss"]
        loss_padded += loss / len(batch_padded)

    batch_balanced = packed_batch(batch, SEQ_LEN, pad_token_id, micro_bs, collate_mode="balancing")

    loss_balanced = 0
    for micro_batch in batch_balanced:
        loss = compute_states(micro_batch, model)["pg_loss"]
        loss_balanced += loss / len(batch_balanced)

    torch.testing.assert_close(loss_padded, loss_balanced)
