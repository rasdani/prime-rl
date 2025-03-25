import torch
from torch import Tensor
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker
import torch.nn.functional as F

from zeroband.training.verl_utils import logprobs_from_logits, masked_mean


# beartype here just make sure we have the correct shape
@jaxtyped(typechecker=typechecker)
def grpo_loss_verl(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch response_seq"],
    advantages: Float[Tensor, "batch response_seq"],
    original_logprobs: Float[Tensor, "batch response_seq"],
    loss_mask: Int[Tensor, "batch response_seq"],
    temperature: float,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    """
    DeepSeek Math Loss: https://arxiv.org/abs/2402.03300

    Args:
        policy_logprobs: Log probabilities from the policy model
        ref_logprobs: Log probabilities from the reference model
        advantages: Advantages for each token
        beta: KL penalty coefficient
        epsilon: Clipping parameter for PPO
        ignore_index: Specifies a target value that is ignored and does not contribute to the loss
    """
    log_probs = log_probs_grpo(logits, input_ids, temperature)
    pg_loss, pg_clipfrac, _ = compute_policy_loss(
        old_log_prob=original_logprobs, log_prob=log_probs, advantages=advantages, eos_mask=loss_mask, cliprange=epsilon
    )
    return pg_loss, pg_clipfrac

# beartype here just make sure we have the correct shape
@jaxtyped(typechecker=typechecker)
def log_probs_grpo(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch response_seq"],
    temperature: float,
) -> Float[Tensor, "batch response_seq"]:
    
    logits.div_(temperature)
    response_length = input_ids.size(-1)
    logits = logits[:, -response_length - 1 : -1]  # (bsz, response_length)
    log_probs = logprobs_from_logits(logits, input_ids)
    return log_probs


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl
