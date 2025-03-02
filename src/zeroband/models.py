from typing import Literal, TypeAlias
from transformers import AutoTokenizer

from torchtune.training import FullModelHFCheckpointer
from torchtune.models.llama2 import llama2
from torchtune.modules import TransformerDecoder

ModelName: TypeAlias = Literal["debugmodel", "150M", "1B"]
ModelType: TypeAlias = TransformerDecoder

name_to_hf_model = {
    "debugmodel": "PrimeIntellect/llama-2m-fresh",
    "150M": "PrimeIntellect/llama-150m-fresh",
    "1B": "PrimeIntellect/llama-1b-fresh",
    "Qwen1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}

name_to_hf_tokenizer = {
    "debugmodel": "mistralai/Mistral-7B-v0.1",
    "150M": "mistralai/Mistral-7B-v0.1",
    "1B": "mistralai/Mistral-7B-v0.1",
    "Qwen1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}



def get_model_and_tokenizer(model_name: ModelName) -> tuple[ModelType, AutoTokenizer]:
    
    # 1. Define the checkpointer and load the checkpoint
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir="/workspace/hub/models--PrimeIntellect--llama-2m-fresh/snapshots/2b2d5f245f5a922578f0a0c17d3cc94484d0b10b/",
        checkpoint_files=["helo"],
        output_dir="/tmp/llama_2m_fresh", # not used
        model_type="LLAMA2",        
    )

    checkpoint = checkpointer.load_checkpoint()
    model_state_dict = checkpoint["model"]
    
    model = llama2()  # Configure with appropriate parameters as needed
    model.load_state_dict(model_state_dict)
    
    
    tokenizer = AutoTokenizer.from_pretrained(name_to_hf_tokenizer[model_name])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer  # type: ignore
