from typing import Iterable, List
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams
from pydantic_config import BaseConfig, parse_argv

from zeroband.models import ModelName, get_model_and_tokenizer, name_to_hf_model

from datasets import load_dataset


# These are the defaults set by vllm.
class SamplingParamConfig(BaseConfig):
    temperature: float = 1
    top_p: float = 1
    top_k: int = -1
    use_beam_search: bool = False
    stop: str | List[str] | None = None
    ignore_eos: bool = False
    max_tokens: int = 16
    presence_penalty: float = 0
    frequency_penalty: float = 0
    logprobs: int | None = None



class Config(BaseConfig):
    name_model: ModelName = "150M"
    dataset: str = "justus27/test-vcu"
    batch_size: int = 32
    max_samples: int | None = None
    sampling_params: SamplingParamConfig = SamplingParamConfig()


def fake_chat_template(messages):
    formatted_prompts = []

    for conversation in messages:
        prompt = ""
        for message in conversation:
            if message["role"] == "user":
                prompt += f"Human: {message['content']}\n\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n\n"
        formatted_prompts.append(prompt.strip())

    return formatted_prompts

def get_new_weights(locator: str) -> dict:
    return {}

def save_vllm_weights(llm: LLM, locator: str):
    model_executor = llm.llm_engine.model_executor
    model_executor.save_sharded_state(path=locator,
                                      pattern="model-{}.pt",
                                      max_size=1e9)
    llm.llm_engine.model_executor.driver_worker.model_runner.model.save_state_dict(locator)


def update_vllm_weights(llm: LLM, locator: str):
    """
    All processes should call this function to get and update the weights of the model.
    """
    new_weights = get_new_weights(locator)
    llm.load_state_dict(new_weights)

def rollout(llm: LLM, dataset, sampling_params: SamplingParams) -> Iterable[CompletionOutput]:

    max_samples = config.max_samples or len(dataset)

    # Process batches
    for i in range(0, min(len(dataset), max_samples), config.batch_size):
        batch = dataset.select(range(i, min(i + config.batch_size, len(dataset))))

        messages = [[{"role": "user", "content": item["prompt"]}, 
                     {"role": "assistant", "content": "<think>\n"}] for item in batch]

        prompts = fake_chat_template(messages)

        outputs: RequestOutput = llm.generate(prompts, sampling_params)

        print(f"Processed {i + len(batch)} samples of {min(len(dataset), max_samples)}")
        for output in outputs:
            yield output


def main(config: Config):

    dataset = load_dataset(config.dataset, split="train")
    sampling_params = SamplingParams(**config.sampling_params.model_dump())

    model, tokenizer = get_model_and_tokenizer(config.name_model)
    import torch
    torch.save(model.state_dict(), "weights.pt")

    print(config.sampling_params.model_dump())

    llm = LLM(model=name_to_hf_model[config.name_model])
    

    for batch in rollout(llm, dataset, sampling_params):
        pass


if __name__ == "__main__":
    config = Config(**parse_argv())  # type: ignore

    main(config)
