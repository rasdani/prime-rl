# This code is adapted to GRPO from the Liger kernel library, which is BSD-2 licensed.
# The original code can be found at https://github.com/linkedin/Liger-Kernel/blob/9c20bd41568b6d5663673c6a0ef7f71b56b50e1b/src/liger_kernel/transformers/monkey_patch.py
# Here is the liscence:

# BSD 2-CLAUSE LICENSE
# Copyright 2024 LinkedIn Corporation 
# All Rights Reserved.
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from functools import partial
from typing import Callable

import logging
from transformers import PreTrainedModel
from typing import List, Optional, Tuple, Union
import torch

from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss
from liger_kernel.chunked_loss.fused_linear_rlhf import LigerFusedLinearRLHFBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from zeroband.logger import get_logger


def apply_fused_linear_grpo(
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2 models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    from transformers.models.qwen2 import modeling_qwen2

    modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_linear_grpo_forward


def qwen2_linear_grpo_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    **loss_kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    
    
    ref_logprobs = loss_kwargs.get("ref_logprobs", None)
    if ref_logprobs is None:
        raise ValueError("ref_logprobs must be provided for GRPO loss")
    advantages = loss_kwargs.get("advantages", None)
    if advantages is None:
        raise ValueError("advantages must be provided for GRPO loss")
    ignore_index = loss_kwargs.get("ignore_index", -100)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=output_attentions if output_attentions is not None else self.config.output_attentions,
        output_hidden_states=output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states,
        return_dict=return_dict if return_dict is not None else self.config.use_return_dict,
        cache_position=cache_position,
    )

    # TODO: Expose in config
    beta: float = 0.04
    epsilon: float = 0.2

    policy_logprobs = outputs.logits

    hidden_states = outputs[0]

    logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

    # if in training mode, don't materialize logits
    if self.training:
        loss = LigerFusedLinearGRPOFunction.apply(
            policy_logprobs, # _input,
            self.lm_head.weight, # lin_weight,
            attention_mask,
            advantages, # rewards,
            bias,
            ref_input,
            ref_weight,
            ref_bias,
            beta,
            False, # self.compiled, (We will compile ourselves)
            False, # self.use_ref_model,
            self.num_generations,
        )

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class LigerFusedLinearGRPOFunction(LigerFusedLinearRLHFBase):
    @staticmethod
    def rlhf_loss_fn(
        log_probs,
        attention_mask,
        rewards,
        ref_log_probs=None,
        beta=0.1,
        **kwargs,
    ):
        """GRPO Loss Function matching GRPOTrainer implementation."""
        # Get chosen token probabilities
        chosen_tokens = log_probs.argmax(dim=-1)  # (batch_size, seq_len)
        chosen_token_logprobs = log_probs.gather(dim=-1, index=chosen_tokens.unsqueeze(-1)).squeeze(
            -1
        )  # (batch_size, seq_len)

        # Get reference model probabilities
        if ref_log_probs is not None:
            with torch.no_grad():
                ref_token_logprobs = ref_log_probs.gather(dim=-1, index=chosen_tokens.unsqueeze(-1)).squeeze(-1)
        else:
            ref_token_logprobs = chosen_token_logprobs.detach()

        # Compute advantages per batch entry in a grouped fashion
        mean_grouped_rewards = rewards.mean()  # [batch_size,]
        std_grouped_rewards = rewards.std()  # [batch_size,]

        # Calculate advantages using the same epsilon as in GRPOTrainer
        eps = 1e-4
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + eps)

        # Compute policy gradient loss with importance sampling ratio
        ratio = torch.exp(chosen_token_logprobs - chosen_token_logprobs.detach())
        policy_loss = -ratio * advantages.unsqueeze(1)

        # Compute KL penalty
        kl_div = (
            torch.exp(ref_token_logprobs - chosen_token_logprobs) - (ref_token_logprobs - chosen_token_logprobs) - 1.0
        )

        # Combine losses
        per_token_loss = policy_loss + beta * kl_div

        # Apply masking and normalize
        masked_loss = per_token_loss * attention_mask
        seq_lengths = attention_mask.sum()
        seq_lengths = torch.clamp(seq_lengths, min=1.0)
        loss = masked_loss.sum() / seq_lengths

        # Calculate metrics
        metrics = (
            chosen_token_logprobs.mean(),  # mean log prob
            chosen_token_logprobs.std(),  # std log prob
            log_probs.mean(),  # mean all log probs
            ((kl_div * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)).mean(),  # mean KL div
        )

        return loss, metrics

    @staticmethod
    def forward(
        ctx,
        _input,
        weight,
        attention_mask,
        rewards,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        beta=0.1,
        compiled=True,
        use_ref_model=True,
        num_generations=1,
    ):
        return LigerFusedLinearRLHFBase.forward(
            ctx=ctx,
            _input=_input,
            weight=weight,
            attention_mask=attention_mask,
            loss_fn=LigerFusedLinearGRPOFunction.rlhf_loss_fn,
            rewards=rewards,
            bias=bias,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            beta=beta,
            compiled=compiled,
            use_ref_model=use_ref_model,
            num_generations=num_generations,
        )

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for GRPO loss.

        Args:
            grad_output: Gradient of the loss (scalar)
            grad_metrics: Gradients of the metrics (not used in backward computation)
        """
        grads = LigerFusedLinearRLHFBase.backward(ctx, grad_output)
        return (
            *grads[:5],  # grad_input, grad_weight, grad_attention_mask, grad_rewards, grad_bias
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
            None,  # grad_beta
            None,  # grad_compiled
            None,  # grad_use_ref_model
            None,  # grad_num_generations
        )
