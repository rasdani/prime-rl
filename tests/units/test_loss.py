from zeroband.training.loss import grpo_loss
import torch
import pytest


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grpo_loss(dtype):
    logits = torch.randn(10, 10, 10, dtype=dtype).cuda()
    original_logprobs = torch.randn(10, 10, dtype=dtype).cuda()
    advantages = torch.randn(10, 10).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()
    input_ids = torch.randint(0, 10, (10, 10)).cuda()

    loss, clip_ratio = grpo_loss(logits, input_ids, advantages, original_logprobs, loss_mask, temperature=0.6, epsilon=0.2)
    assert loss.shape == ()
    assert loss.item() is not None
    assert clip_ratio.shape == ()
    assert clip_ratio.item() is not None
