from zeroband.training.loss import grpo_loss
import torch


def test_grpo_loss():
    logits = torch.randn(10, 10, 10).cuda()
    original_logprobs = torch.randn(10, 10).cuda()
    advantages = torch.randn(10, 10).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()
    input_ids = torch.randint(0, 10, (10, 10)).cuda()

    loss = grpo_loss(logits, input_ids, advantages, original_logprobs, loss_mask)
    assert loss.shape == ()
    assert loss.item() is not None
