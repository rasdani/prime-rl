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

            return {"pg_loss": pg_loss, "clip_ratio": clip_ratio, "batch": batch}


def test_e2e():
    model, tokenizer = get_model_and_tokenizer("PrimeIntellect/llama-2m-fresh", "flash_attention_2")
    model = model.to("cuda")
    assert model is not None
    assert tokenizer is not None

    SEQ_LEN = 64
    # create a fake dataset
    dataset = FakeTokenizedDataset(seq_len=SEQ_LEN, vocab_size=128)

    pad_token_id = 0

    batch_size = 2
    batch = []
    for i in range(batch_size):
        batch.append(next(iter(dataset)))

    batch_padded = packed_batch(batch, SEQ_LEN, pad_token_id, batch_size, sequence_packing=False)

    assert len(batch_padded) == 1

    states_padded = compute_states(batch_padded[0], model)

    print("=========")
    batch_packed = packed_batch(batch, SEQ_LEN, pad_token_id, batch_size, sequence_packing=True)
    assert len(batch_packed) == 1

    states_packed = compute_states(batch_packed[0], model)

    assert batch_packed[0]["rewards"].sum() == batch_padded[0]["rewards"].sum()
    assert batch_packed[0]["seq_lens"].sum() == batch_padded[0]["seq_lens"].sum()

    all_rewards_padded = sum([batch_padded[0]["rewards"], batch_packed[0]["rewards"]])
    all_rewards_packed = states_padded["batch"]["rewards"] + states_packed["batch"]["rewards"]

    torch.testing.assert_close(all_rewards_padded, all_rewards_packed)

    print(f"{states_padded['batch']['input_ids']=}")
    print(f"{states_packed['batch']['input_ids']=}")

    all_inputs_ids_padded = states_padded["batch"]["input_ids"][states_padded["batch"]["input_ids"] != pad_token_id].sum()
    all_inputs_ids_packed = states_packed["batch"]["input_ids"][states_packed["batch"]["input_ids"] != pad_token_id].sum()

    torch.testing.assert_close(all_inputs_ids_padded, all_inputs_ids_packed)

    # all_padded_tokens_padded = states_padded["batch"]["input_ids"][states_padded["batch"]["input_ids"] == pad_token_id].sum()
    # all_padded_tokens_packed = states_packed["batch"]["input_ids"][states_packed["batch"]["input_ids"] == pad_token_id].sum()

    # assert all_padded_tokens_padded > all_padded_tokens_packed

    # all_padded_token_ = torch.cat([batch_padded[0]["input_ids"][batch_padded[0]["input_ids"] != tokenizer.pad_token_id], batch_packed[0]["input_ids"][batch_packed[0]["input_ids"] != tokenizer.pad_token_id]])
    # all_packed_token_ids = torch.cat([states_padded["batch"]["input_ids"][states_padded["batch"]["input_ids"] != tokenizer.pad_token_id], states_packed["batch"]["input_ids"][states_packed["batch"]["input_ids"] != tokenizer.pad_token_id]])

    # assert

    torch.testing.assert_close(all_inputs_ids_padded, all_inputs_ids_packed)

    torch.testing.assert_close(states_padded["pg_loss"], states_packed["pg_loss"])
