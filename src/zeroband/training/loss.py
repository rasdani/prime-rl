import torch
from torch import Tensor
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker
import torch.nn.functional as F


def remove_bos_tokens(packed: Tensor, position_ids: Tensor | None) -> Tensor:
    """
    Drop the bos tokens from the packed tensor, given the position ids
    Args:
        packed: Tensor of shape (batch, seq, ...)
        position_ids: Tensor of shape (1, seq) or None, implying torch.arange(0, seq).unsqueeze(0) (drop only the first token)
    """
    if position_ids is None:
        return packed[:, 1:, ...]
    else:
        position_ids = position_ids.squeeze(0)  # (seq)
        non_bos_mask = position_ids != 0
        packed = packed[:, non_bos_mask, ...]
        return packed


def remove_last_logit(packed: Tensor, position_ids: Tensor | None) -> Tensor:
    """
    Drop the eos tokens from the packed tensor, given the position ids
    Args:
        packed: Tensor of shape (batch, seq, ...)
        position_ids: Tensor of shape (1, seq) or None, implying torch.arange(0, seq).unsqueeze(0) (drop only the last logit)
    """
    if position_ids is None:
        return packed[:, :-1, ...]
    else:
        position_ids = position_ids.squeeze(0)  # (seq)

        # torch.diff returns 1 smaller, so add another position id to mark the eos logit.
        padded_pos_ids = torch.cat([position_ids, torch.tensor([-1], device=position_ids.device)])

        # Mask off the eos logits for each sequence
        keep_mask = (torch.diff(padded_pos_ids) >= 0)
        packed = packed[:, keep_mask, ...]
        return packed


# beartype here just make sure we have the correct shape
@jaxtyped(typechecker=typechecker)
def grpo_loss(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    position_ids: Int[Tensor, "batch seq"] | None,
    temperature: float,
    epsilon_low: float,
    epsilon_high: float,
    masked_mean_axis: int | None,
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
        position_ids=position_ids,
        temperature=temperature,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        masked_mean_axis=masked_mean_axis,
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


# @torch.compile
def _compile_grpo_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    advantages: torch.Tensor,
    original_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    temperature: float,
    epsilon_low: float,
    epsilon_high: float,
    masked_mean_axis: int | None,
) -> tuple[Tensor, Tensor]:
    # we start by dropping the bos token because it does not have a corresponding logit
    input_ids = remove_bos_tokens(input_ids, position_ids)
    advantages = remove_bos_tokens(advantages, position_ids)
    # original_logprobs = remove_bos_tokens(original_logprobs, position_ids) # no need to do it now
    loss_mask = remove_bos_tokens(loss_mask, position_ids)

    # from the logits we drop the last logits because it corresponds to the next token that will be sample but is not here yet
    logits = remove_last_logit(logits, position_ids)  # (B, L-1, V), exclude the last logit: it corresponds to the next token prediction

    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    logits = logits / temperature
    per_token_logps = selective_log_softmax(logits, input_ids)

    coef_1 = torch.exp(per_token_logps - original_logprobs)
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = -coef_1 * advantages
    per_token_loss2 = -coef_2 * advantages
    per_token_loss = torch.max(per_token_loss1, per_token_loss2)

    loss = _apply_mask(per_token_loss, loss_mask, masked_mean_axis)

    is_clipped = (per_token_loss1 < per_token_loss2).float()
    clip_ratio = _apply_mask(is_clipped, loss_mask, masked_mean_axis)
    return loss, clip_ratio


@jaxtyped(typechecker=typechecker)
def entropy_loss(
    logits: Float[Tensor, "batch seq vocab"],
    loss_mask: Int[Tensor, "batch seq"],
    position_ids: Int[Tensor, "batch seq"] | None,
    temperature: float,
    masked_mean_axis: int | None
) -> Tensor:
    return _compile_entropy_loss(
        logits=logits,
        loss_mask=loss_mask,
        position_ids=position_ids,
        temperature=temperature,
        masked_mean_axis=masked_mean_axis
    )


# @torch.compile
def _compile_entropy_loss(logits: torch.Tensor, loss_mask: torch.Tensor, position_ids: torch.Tensor | None, temperature: float, masked_mean_axis: int | None):
    logits = remove_last_logit(logits, position_ids)
    logits = logits / temperature

    loss_mask = remove_bos_tokens(loss_mask, position_ids)

    pd = torch.nn.functional.softmax(logits, dim=-1)
    sumreduce = torch.sum(pd * logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - sumreduce

    masked = _apply_mask(entropy, loss_mask, masked_mean_axis)
    return masked


def _apply_mask(tensor: torch.Tensor, mask: torch.Tensor, masked_mean_axis: int | None) -> torch.Tensor:
    # First sum over sequence dimension (dim=1), then mean over batch (dim=0)
    if masked_mean_axis is None:
        return (tensor * mask).sum() / (mask.sum() + 1e-8)  # Add small epsilon to avoid division by zero
    else:
        return ((tensor * mask).sum(dim=masked_mean_axis) / (mask.sum(dim=masked_mean_axis) + 1e-8)).mean()
