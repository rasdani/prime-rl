from typing import Literal, TypeAlias
import torch
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)

ModelName: TypeAlias = Literal["debugmodel", "150M", "1B"]

name_to_hf_model = {
    "debugmodel": "PrimeIntellect/llama-2m-fresh",
    "150M": "PrimeIntellect/llama-150m-fresh",
    "1B": "PrimeIntellect/llama-1b-fresh",
}

name_to_hf_tokenizer = {
    "debugmodel": "mistralai/Mistral-7B-v0.1",
    "150M": "mistralai/Mistral-7B-v0.1",
    "1B": "mistralai/Mistral-7B-v0.1",
}


def get_model_and_tokenizer(model_name: ModelName) -> tuple[torch.nn.Module, AutoTokenizer, int]:
    tokenizer = AutoTokenizer.from_pretrained(name_to_hf_tokenizer[model_name])
    config_model = LlamaConfig.from_pretrained(name_to_hf_model[model_name], attn_implementation="flex_attention")
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=name_to_hf_model[model_name], config=config_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer
