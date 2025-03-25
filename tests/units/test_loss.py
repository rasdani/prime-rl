from zeroband.training.loss import grpo_loss_verl, log_probs_grpo
import torch
import pytest


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grpo_loss(dtype):
    
    batch_size = 10
    vocab_size = 32
    prompt_length = 10
    response_length = 20
    
    logits = torch.randn(batch_size, prompt_length + response_length, vocab_size, dtype=dtype).cuda()
    original_logprobs = torch.randn(batch_size, response_length, dtype=dtype).cuda()
    
    
    advantages = torch.randn(batch_size, response_length).cuda()
    loss_mask = torch.ones(batch_size, response_length).int().cuda()
    input_ids = torch.randint(0, vocab_size, (batch_size, response_length)).cuda()


    
    loss, clip_ratio = grpo_loss_verl(logits, input_ids, advantages, original_logprobs, loss_mask, temperature=0.6, epsilon=0.2)
    assert loss.shape == ()
    assert loss.item() is not None
    assert clip_ratio.shape == ()
    assert clip_ratio.item() is not None

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_log_probs_grpo(dtype): 
    batch_size = 10
    vocab_size = 32
    prompt_length = 10
    response_length = 20
    
    logits = torch.randn(batch_size, prompt_length + response_length, vocab_size, dtype=dtype).cuda()
    
    input_ids = torch.randint(0, vocab_size, (batch_size, response_length)).cuda()
    
    log_probs = log_probs_grpo(logits, input_ids, temperature=0.6)
    assert log_probs.shape == (batch_size, response_length)
