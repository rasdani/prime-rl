import torch
from torch import Tensor
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker
import torch.nn.functional as F


# beartype here just make sure we have the correct shape
@jaxtyped(typechecker=typechecker)
def grpo_loss(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq"],
    loss_mask: Bool[Tensor, "batch seq"],
    temperature: float,
    epsilon: float = 0.2,
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
    return _compile_grpo_loss(
        logits=logits,
        input_ids=input_ids,
        advantages=advantages,
        original_logprobs=original_logprobs,
        loss_mask=loss_mask,
        temperature=temperature,
        epsilon=epsilon,
    )


def selective_log_softmax(logits, index):
    """
    credits to https://github.com/huggingface/trl/blob/07cfe1677e552b7d5c92b7740e5b2f0b057661d8/trl/trainer/utils.py#L1659

    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


@torch.compile
def _compile_grpo_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    advantages: torch.Tensor,
    original_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    temperature: float,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    # we start by dropping the bos token because it does not have a corresponding logit
    input_ids = input_ids[:, 1:]
    advantages = advantages[:, 1:]
    original_logprobs = original_logprobs[:, 1:]
    loss_mask = loss_mask[:, 1:]

    # from the logits we drop the last logits because it corresponds to the next token that will be sample but is not here yet
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token prediction

    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    logits = logits / temperature
    per_token_logps = selective_log_softmax(logits, input_ids)

    old_per_token_logps = per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old_per_token_logps)
    coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    loss = (per_token_loss * loss_mask).sum() / loss_mask.sum()

    is_clipped = (per_token_loss1 < per_token_loss2).float()
    clip_ratio = (is_clipped * loss_mask).sum() / loss_mask.sum()
    return loss, clip_ratio
