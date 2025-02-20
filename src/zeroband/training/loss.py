import torch
from torch import Tensor
from jaxtyping import Float


@torch.compile
def grpo_loss(
    policy_logprobs: Float[Tensor, "batch seq"],
    ref_logprobs: Float[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch"],
    beta: float = 0.04,
    epsilon: float = 0.2,
):
    """
    DeepSeek Math Loss: https://arxiv.org/abs/2402.03300
    """

    # Expand advantages to match sequence dimension
    advantages = advantages.unsqueeze(-1)  # [batch_size, 1]

    # Policy ratio
    ratio = torch.exp(policy_logprobs - ref_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    # Policy loss
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    # KL penalty (unbiased estimator)
    kl_div = ref_logprobs / policy_logprobs - torch.log(ref_logprobs / policy_logprobs) - 1

    # Reduce across sequence length
    policy_loss = policy_loss.mean(dim=-1)
    kl_penalty = kl_div.mean(dim=-1)

    # Final loss (mean across batch)
    loss = (policy_loss + beta * kl_penalty).mean()

    return loss
