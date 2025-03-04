from collections import defaultdict
import os
from pathlib import Path
from typing import Literal
import uuid
import torch
from vllm import LLM, SamplingParams
from pydantic_config import BaseConfig, parse_argv
import vllm
import time

# from vllm.model_executor.model_loader
from vllm.model_executor.model_loader.loader import _process_weights_after_loading

from zeroband.logger import get_logger
from zeroband.models import ModelName, name_to_hf_model

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq


class SamplingParamConfig(BaseConfig):
    temperature: float = 0.6
    max_tokens: int = 4096
    ignore_eos: bool = False
    top_p: float = 0.95


dataset_key = defaultdict(lambda: "prompt", {"justus27/difficulty-calibrated-deepscaler": "problem"})


class Config(BaseConfig):
    name_model: ModelName = "150M"
    dataset: str = "justus27/test-vcu"
    batch_size: int = 32
    max_samples: int | None = None
    output_path: str = "outputs"
    tp: int | Literal["all"] = 1
    max_seq_len: int | None = None
    total_step: int | None = None
    step_batch_size: int | None = None  # will be use to create stable file
    rollout_path: str | None = None

    quant: Literal["fp8"] | None = None

    sampling: SamplingParamConfig = SamplingParamConfig()


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


pa_schema = pa.schema(
    [
        ("input_tokens", pa.list_(pa.int32())),
        ("output_tokens", pa.list_(pa.int32())),
        ("advantages", pa.float32()),
        ("proofs", pa.binary()),
        ("step", pa.int32()),
    ]
)


def get_parquet_table(generated_tokens: list[vllm.RequestOutput], step: int) -> pa.Table:
    # Initialize lists for each column
    input_tokens_list = []
    output_tokens_list = []
    advantages_list = []
    proofs_list = []
    steps_list = []

    # Process each RequestOutput
    for request in generated_tokens:
        # For each output in the request (handling top-n outputs)
        for output in request.outputs:
            # Input tokens are the prompt tokens
            input_tokens_list.append(request.prompt_token_ids)

            # Output tokens from the completion
            output_tokens_list.append(output.token_ids)

            # Initialize with 0 advantage as it's not part of RequestOutput
            # You might want to modify this based on your advantage calculation
            advantages_list.append(0)

            # TODO: Add toploc proof
            proofs_list.append("I am toploc proof, handcrafted by jack".encode())

            # Add step
            steps_list.append(step)

    # Create PyArrow arrays
    arrays = [
        pa.array(input_tokens_list, type=pa.list_(pa.int32())),
        pa.array(output_tokens_list, type=pa.list_(pa.int32())),
        pa.array(advantages_list, type=pa.float32()),
        pa.array(proofs_list, type=pa.binary()),
        pa.array(steps_list, type=pa.int32()),
    ]

    # Create and return table
    return pa.Table.from_arrays(arrays, schema=pa_schema)


def reload_model_weights(llm: LLM, ckpt_path: str):
    # Access the internal model from vLLM
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    # Load state dict
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Create a better weight iterator that filters out empty keys and handles prefixes
    def weights_iterator():
        for name, tensor in state_dict.items():
            # Skip empty keys
            if not name:
                continue
            yield name, tensor

    # Load weights
    model.load_weights(weights_iterator())

    # Process weights after loading (important for some models)
    model_config = llm.llm_engine.model_config
    device = next(model.parameters()).device
    _process_weights_after_loading(model, model_config, device)

    return llm


def main(config: Config):  # -> list[dict[str, Any]]:
    if config.tp == "all":
        config.tp = torch.cuda.device_count()

    llm = LLM(
        model=name_to_hf_model[config.name_model],
        tensor_parallel_size=config.tp,
        max_model_len=config.max_seq_len,
        quantization=config.quant,
    )
    logger = get_logger("INFERENCE")

    sampling_params = SamplingParams(**config.sampling.model_dump())

    dataset = load_dataset(config.dataset, split="train")

    max_samples = config.max_samples or len(dataset)

    step = 0
    total_samples = 0
    total_tokens = 0

    for i in range(0, min(len(dataset), max_samples), config.batch_size):
        if config.rollout_path is not None:
            last_step = list(Path(config.rollout_path).glob("step_*"))
            if len(last_step) > 0:
                last_step = max(last_step, key=lambda x: int(x.stem.split("_")[-1]))
                maybe_new_step = int(last_step.stem.split("_")[-1])

                if step < maybe_new_step:
                    stable_file = last_step / "stable"

                    if stable_file.exists():
                        logger.info(f"Reloading model weights from {config.rollout_path} step {maybe_new_step}")
                        llm = reload_model_weights(llm, Path(config.rollout_path) / f"step_{maybe_new_step}/model.pt")
                        step = maybe_new_step
                        logger.info(f"Reloaded model weights from {config.rollout_path} step {maybe_new_step}")
                    else:
                        logger.info(f"No stable file found at {config.rollout_path} step {maybe_new_step}")

        # Get batch
        batch = dataset.select(range(i, min(i + config.batch_size, len(dataset))))

        # Prepare messages
        messages = [
            [{"role": "user", "content": item[dataset_key[config.dataset]]}, {"role": "assistant", "content": "<think>\n"}]
            for item in batch
        ]

        # Get tokenized inputs
        prompts = fake_chat_template(messages)

        start_time = time.time()
        generated_tokens = llm.generate(prompts, sampling_params, use_tqdm=False)
        end_time = time.time()

        # Calculate tokens and throughput
        batch_input_tokens = sum(len(req.prompt_token_ids) for req in generated_tokens)
        batch_output_tokens = sum(sum(len(output.token_ids) for output in req.outputs) for req in generated_tokens)
        batch_total_tokens = batch_input_tokens + batch_output_tokens
        total_tokens += batch_total_tokens

        avg_seq_length = batch_total_tokens / len(generated_tokens) if generated_tokens else 0

        elapsed_time = end_time - start_time
        tokens_per_second = batch_total_tokens / elapsed_time if elapsed_time > 0 else 0

        logger.info(
            f"Batch throughput: {tokens_per_second:.2f} tok/sec ({batch_total_tokens} tokens in {elapsed_time:.2f}s, avg seq len: {avg_seq_length:.1f})"
        )

        if config.step_batch_size is not None and total_samples % config.step_batch_size == 0:
            logger.info(f"Generated {total_samples} total samples")

        table = get_parquet_table(generated_tokens, step)

        step_path = Path(config.output_path) / f"step_{step}"
        os.makedirs(step_path, exist_ok=True)

        pq.write_table(table, f"{step_path}/{uuid.uuid4()}.parquet")
        total_samples += len(prompts)
        logger.info(f"Generated {total_samples} total samples")

        if config.step_batch_size is not None and total_samples % config.step_batch_size == 0:
            # logger.info(f"Reached step batch size {config.step_batch_size}. Writing stable file")
            stable_file = step_path / "stable"
            stable_file.touch()

        if config.total_step is not None:
            if step >= config.total_step:
                logger.info(f"Reached total step {config.total_step}, stopping inference")
                break


if __name__ == "__main__":
    config = Config(**parse_argv())  # type: ignore

    main(config)
