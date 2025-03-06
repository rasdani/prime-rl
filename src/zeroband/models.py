from typing import Literal, TypeAlias
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

ModelName: TypeAlias = Literal["debugmodel", "150M", "1B", "Qwen32B", "Qwen1.5B", "Qwen7B", "Llama8B", "QwQ32B"]
ModelType: TypeAlias = LlamaForCausalLM | Qwen2ForCausalLM

name_to_hf_model = {
    "debugmodel": "PrimeIntellect/llama-2m-fresh",
    "150M": "PrimeIntellect/llama-150m-fresh",
    "1B": "PrimeIntellect/llama-1b-fresh",
    "Qwen1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Llama8B": "meta-llama/Meta-Llama-3-8B",
    "QwQ32B": "Qwen/QwQ-32B",
}

name_to_hf_tokenizer = {
    "debugmodel": "mistralai/Mistral-7B-v0.1",
    "150M": "mistralai/Mistral-7B-v0.1",
    "1B": "mistralai/Mistral-7B-v0.1",
    "Qwen1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Llama8B": "meta-llama/Meta-Llama-3-8B",
    "QwQ32B": "Qwen/QwQ-32B",
}

name_to_class = {
    "debugmodel": (LlamaConfig, LlamaForCausalLM),
    "150M": (LlamaConfig, LlamaForCausalLM),
    "1B": (LlamaConfig, LlamaForCausalLM),
    "Qwen1.5B": (Qwen2Config, Qwen2ForCausalLM),
    "Qwen7B": (Qwen2Config, Qwen2ForCausalLM),
    "Qwen32B": (Qwen2Config, Qwen2ForCausalLM),
    "Llama8B": (LlamaConfig, LlamaForCausalLM),
    "QwQ32B": (Qwen2Config, Qwen2ForCausalLM),
}


def get_model_and_tokenizer(model_name: ModelName) -> tuple[ModelType, AutoTokenizer]:
    config_class, model_class = name_to_class[model_name]
    tokenizer = AutoTokenizer.from_pretrained(name_to_hf_tokenizer[model_name])
    config_model = config_class.from_pretrained(name_to_hf_model[model_name], attn_implementation="flex_attention")
    config_model.use_cache = False
    model = model_class.from_pretrained(pretrained_model_name_or_path=name_to_hf_model[model_name], config=config_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer  # type: ignore
