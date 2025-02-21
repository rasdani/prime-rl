import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker


# beatype here just make sure we have the correct shape
@jaxtyped(typechecker=typechecker)
def grpo_loss(
    policy_logprobs: Float[Tensor, "batch seq vocab"],
    ref_logprobs: Float[Tensor, "batch seq vocab"],
    advantages: Float[Tensor, "batch seq"],
    beta: float = 0.04,
    epsilon: float = 0.2,
):
    """
    DeepSeek Math Loss: https://arxiv.org/abs/2402.03300
    """
    return _compile_grpo_loss(policy_logprobs, ref_logprobs, advantages, beta, epsilon)


@torch.compile
def _compile_grpo_loss(policy_logprobs: torch.Tensor, ref_logprobs: torch.Tensor, advantages: torch.Tensor, beta: float, epsilon: float):
    # Expand advantages to match sequence dimension
    advantages = advantages.unsqueeze(-1)  # [batch_size, 1]

    # Policy ratio
    ratio = torch.exp(policy_logprobs - ref_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    # Policy loss
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    # KL penalty (unbiased estimator)
    # kl_div = ref_logprobs / policy_logprobs - torch.log(ref_logprobs / policy_logprobs) - 1

    # Reduce across sequence length
    policy_loss = policy_loss.mean(dim=-1)
    # kl_penalty = kl_div.mean(dim=-1)
    kl_penalty = 0

    # Final loss (mean across batch)
    loss = (policy_loss + beta * kl_penalty).mean()

    return loss
