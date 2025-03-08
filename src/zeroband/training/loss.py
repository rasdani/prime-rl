import torch
from torch import Tensor
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker
from torch.nn import functional as F


# beartype here just make sure we have the correct shape
@jaxtyped(typechecker=typechecker)
def grpo_loss(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq"],
    loss_mask: Int[Tensor, "batch seq"],
    epsilon: float = 0.2,
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
    return _compile_grpo_loss(logits, input_ids, advantages, original_logprobs, loss_mask, epsilon)


@torch.compile
def _compile_grpo_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    advantages: torch.Tensor,
    original_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    epsilon: float,
):
    """
    Computes the GRPO loss.
    """
    # Get log probs from current policy
    log_probs = F.log_softmax(logits, dim=-1)

    # Extract per-token log probs for the tokens that were generated
    per_token_logps = torch.gather(log_probs, dim=2, index=input_ids.unsqueeze(-1)).squeeze(-1)

    # We only have original log probs for tokens that were generated, not for the first prompt token
    # So we need to drop the first token from all tensors
    per_token_logps = per_token_logps[:, 1:]
    advantages = advantages[:, 1:]
    original_logprobs = original_logprobs[:, 1:]
    loss_mask = loss_mask[:, 1:]

    # Compute ratio for PPO clipping
    ratio = torch.exp(per_token_logps - original_logprobs)

    # Compute clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    # Compute PPO loss
    unclipped_loss = ratio * advantages
    clipped_loss = clipped_ratio * advantages
    per_token_loss = torch.min(unclipped_loss, clipped_loss)

    # Apply mask and average
    masked_loss = (per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1e-5)

    # Return mean loss
    return -masked_loss.mean()  # Negative because we want to maximize reward
