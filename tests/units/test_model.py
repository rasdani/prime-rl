from zeroband.models import get_model_and_tokenizer

import torch


def test_model():
    model, tokenizer = get_model_and_tokenizer("meta-llama/Meta-Llama-3-8B", "sdpa")
    assert model is not None

    BS = 2
    SEQ_LEN = 16

    model = model.to("cuda")

    inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")

    outputs = model(input_ids=inputs_ids).logits

    assert outputs.shape == (BS, SEQ_LEN, len(tokenizer))
