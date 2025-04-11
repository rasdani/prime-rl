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


def compute_loss(batch, model):
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            temperature = 0.6
            input_ids = batch["input_ids"].to("cuda")
            # print(batch["input_ids"].shape)
            # print(batch["position_ids"])
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
            # loss_mask = batch["loss_mask"][:, 1:].to("cuda")

            # return _apply_mask(per_token_logps, loss_mask, None)
            return pg_loss


def test_e2e():
    model, tokenizer = get_model_and_tokenizer("PrimeIntellect/llama-2m-fresh", "flash_attention_2")
    model = model.to("cuda")
    assert model is not None
    assert tokenizer is not None

    SEQ_LEN = 64
    # create a fake dataset
    dataset = FakeTokenizedDataset(seq_len=SEQ_LEN, vocab_size=128)

    batch_size = 8
    batch = []
    for i in range(batch_size):
        batch.append(next(iter(dataset)))

    batch_padded = packed_batch(batch, SEQ_LEN, tokenizer.pad_token_id, batch_size, sequence_packing=False)

    assert len(batch_padded) == 1

    loss_padded = compute_loss(batch_padded[0], model)

    print("=========")
    batch_packed = packed_batch(batch, SEQ_LEN, tokenizer.pad_token_id, batch_size, sequence_packing=True)
    assert len(batch_packed) == 1

    loss_packed = compute_loss(batch_packed[0], model)

    assert batch_packed[0]["rewards"].sum() == batch_padded[0]["rewards"].sum()
    assert batch_packed[0]["seq_lens"].sum() == batch_padded[0]["seq_lens"].sum()

    print(loss_padded, loss_packed)
    torch.testing.assert_close(loss_padded, loss_packed)
