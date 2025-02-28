from typing import Literal, TypeAlias
import torch
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)
ModelName: TypeAlias = Literal["debugmodel", "150M", "1B", "Qwen32B"]

name_to_hf_model = {
    "debugmodel": "PrimeIntellect/llama-2m-fresh",
    "150M": "PrimeIntellect/llama-150m-fresh",
    "1B": "PrimeIntellect/llama-1b-fresh",
    "Qwen32B": "Qwen/Qwen2.5-32B",
}

name_to_hf_tokenizer = {
    "debugmodel": "mistralai/Mistral-7B-v0.1",
    "150M": "mistralai/Mistral-7B-v0.1",
    "1B": "mistralai/Mistral-7B-v0.1",
    "Qwen32B": "Qwen/Qwen2.5-32B",
}

name_to_class = {
    "debugmodel": (LlamaConfig, LlamaForCausalLM),
    "150M": (LlamaConfig, LlamaForCausalLM),
    "1B": (LlamaConfig, LlamaForCausalLM),
    "Qwen32B": (Qwen2Config, Qwen2ForCausalLM),
}

def get_model_and_tokenizer(model_name: ModelName) -> tuple[torch.nn.Module, AutoTokenizer]:
    config_class, model_class = name_to_class[model_name]
    tokenizer = AutoTokenizer.from_pretrained(name_to_hf_tokenizer[model_name])
    config_model = config_class.from_pretrained(name_to_hf_model[model_name], attn_implementation="flex_attention")
    model = model_class.from_pretrained(pretrained_model_name_or_path=name_to_hf_model[model_name], config=config_model)
    return model, tokenizer # type: ignore
