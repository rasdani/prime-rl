from zeroband.training.loss import grpo_loss
import torch


def test_grpo_loss():
    policy_logprobs = torch.randn(10, 10).cuda()
    ref_logprobs = torch.randn(10, 10).cuda()
    advantages = torch.randn(10).cuda()
    loss = grpo_loss(policy_logprobs, ref_logprobs, advantages)
    assert loss.shape == ()
    assert loss.item() is not None
