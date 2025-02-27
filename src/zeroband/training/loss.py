import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker


# beartype here just make sure we have the correct shape
@jaxtyped(typechecker=typechecker)
def grpo_loss(
    policy_logprobs: Float[Tensor, "batch seq vocab"],
    ref_logprobs: Float[Tensor, "batch seq vocab"],
    advantages: Float[Tensor, "batch seq"],
    beta: float = 0.04,
    epsilon: float = 0.2,
    ignore_index: int = -100,
):
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
    return _compile_grpo_loss(policy_logprobs, ref_logprobs, advantages, beta, epsilon, ignore_index)


@torch.compile
def _compile_grpo_loss(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    beta: float,
    epsilon: float,
    ignore_index: int,
):
    # Create mask for tokens that should be ignored
    # Shape: [batch, seq]
    mask = (advantages != ignore_index).float()

    # Expand advantages to match sequence dimension
    advantages = advantages.unsqueeze(-1)  # [batch, seq, 1]

    # Policy ratio
    ratio = torch.exp(policy_logprobs - ref_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    # Policy loss
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    # Apply mask to ignore specified indices
    # Expand mask to match policy_loss dimensions
    mask_expanded = mask.unsqueeze(-1)  # [batch, seq, 1]
    policy_loss = policy_loss * mask_expanded

    # KL penalty (commented out in original code)
    # kl_div = ref_logprobs / policy_logprobs - torch.log(ref_logprobs / policy_logprobs) - 1
    # if implemented, would need: kl_div = kl_div * mask_expanded

    # Reduce across sequence length with proper normalization
    # Sum and divide by number of non-ignored tokens in each batch
    # Using a small constant epsilon (1e-8) to prevent division by zero
    EPSILON = 1e-8
    seq_lengths = mask.sum(dim=-1, keepdim=True)  # [batch, 1]
    policy_loss = policy_loss.sum(dim=-1) / (seq_lengths + EPSILON)

    # kl_penalty = (kl_div.sum(dim=-1) / (seq_lengths + 1e-8)) if implemented
    kl_penalty = 0

    # Final loss (mean across batch)
    loss = (policy_loss + beta * kl_penalty).mean()

    return loss
