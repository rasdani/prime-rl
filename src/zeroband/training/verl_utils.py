# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contain small torch utilities
"""

import torch
import torch.distributed
import torch.nn.functional as F

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


def gather_from_labels(data, label):
    """Gather the label from data. The value in label should be [0, vocab_size)

    Args:
        data: (..., vocab_size)
        label (torch.IntTensor) : (...,)

    Returns:

    """

    output = torch.gather(data, -1, label.unsqueeze(-1)).squeeze(-1)
    return output


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_naive(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits, labels):
    output = -cross_entropy_loss(logits, labels)[0]
    return output


def logprobs_from_logits_naive(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logpy = gather_from_labels(logp, labels)
    return logpy


def logprobs_of_labels_v2(logits: torch.FloatTensor, labels):
    """
    A memory efficient implementation of logprobs_from_logits
    """
    assert logits.dtype == torch.float32, "Using bf16 logits with logprobs_of_labels_v2 may lead to divergence"
    logprobs_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1))
    logprobs_labels = logprobs_labels - torch.logsumexp(logits, dim=-1, keepdim=True)
    return logprobs_labels.squeeze(-1)


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
