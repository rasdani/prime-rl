from typing import Literal, TypeAlias
from transformers import AutoTokenizer, LlamaForCausalLM, Qwen2ForCausalLM, AutoConfig, AutoModelForCausalLM


ModelName: TypeAlias = Literal[
    "PrimeIntellect/llama-2m-fresh",
    "PrimeIntellect/llama-150m-fresh",
    "PrimeIntellect/llama-1b-fresh",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/QwQ-32B",
]

ModelType: TypeAlias = LlamaForCausalLM | Qwen2ForCausalLM
AttnImpl: TypeAlias = Literal["flex_attention", "sdpa", "flash_attention_2"]


def get_model_and_tokenizer(model_name: ModelName, attn_impl: AttnImpl) -> tuple[ModelType, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config_model = AutoConfig.from_pretrained(model_name, attn_implementation=attn_impl)
    config_model.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, config=config_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer  # type: ignore
