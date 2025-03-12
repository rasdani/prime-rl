import atexit
import os
import shutil
import signal
import sys
import time

from pydantic import model_validator
from pydantic_config import parse_argv, BaseConfig
import torch
from zeroband.train import Config as TrainConfig
from zeroband.inference import Config as InferenceConfig
from zeroband.inference import inference_run
from zeroband.train import train
from zeroband.logger import get_logger

import torch.multiprocessing as mp

from zeroband.training.mp import EnvWrapper, cuda_available_devices

processes = []


class Config(BaseConfig):
    train: TrainConfig
    inference: InferenceConfig

    n_gpus: int | None = None
    ratio: float = 0.5  # for now we have half train and half inference

    torchrun_rdzv_address: str = "localhost"
    torchrun_rdzv_port: int = 29500

    total_steps: int | None = None

    rollout_path: str  # rollout_path is define at the top and is inherited by train and inference via the model_validator above

    rollout_data: str  # going to be use by inference to save file and training to load

    batch_size: int | None = None  # going to be use by inference to save file and training to load

    @model_validator(mode="after")
    def validate_ckpt_path(self):
        assert self.train.ckpt.rollout_path is None, "train.ckpt.rollout_path must be None when ckpt_path is set"
        assert self.inference.rollout_path is None, "inference.rollout_path must be None when ckpt_path is set"

        self.train.ckpt.rollout_path = self.rollout_path
        self.inference.rollout_path = self.rollout_path

        return self

    @model_validator(mode="after")
    def validate_rollout_data(self):
        self.train.data.path = self.rollout_data
        self.inference.output_path = self.rollout_data
        return self

    @model_validator(mode="after")
    def total_steps_check(self):
        if self.total_steps is not None:
            self.train.optim.total_steps = self.total_steps * self.train.optim.step_per_rollout
            self.inference.total_step = self.total_steps

        return self

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size is not None:
            assert self.batch_size % self.train.optim.step_per_rollout == 0, "batch_size must be divisible by step_per_rollout"

            self.inference.step_batch_size = self.batch_size // self.inference.sampling.n
            self.train.optim.batch_size = self.batch_size // self.train.optim.step_per_rollout

        return self

    @model_validator(mode="after")
    def validate_model_name(self):
        self.inference.name_model = self.train.name_model
        return self


def train_torchrun(config: TrainConfig, rdzv_address: str, rdzv_port: int, gpus_ids: list[int]) -> list[mp.Process]:
    """
    This funciton simulated torchrun but manage to wrap a function call instead of starting from a files.

    Under the hood it just created n_proc processes and set the environment variables for each of them.

    Torchrun is doing this as well under the hood but wrap logs and more advance rdzv features that we don't need yet.

    """
    # Set start method to 'spawn' to avoid CUDA initialization issues
    config.gpus_ids = gpus_ids
    nproc_per_node = len(gpus_ids)

    processes = []
    for rank in range(nproc_per_node):
        # Prepare environment variables
        envs = {}
        envs["MASTER_ADDR"] = rdzv_address
        envs["MASTER_PORT"] = str(rdzv_port)
        envs["CUDA_VISIBLE_DEVICES"] = cuda_available_devices(gpus_ids)
        envs["RANK"] = str(rank)
        envs["LOCAL_RANK"] = str(rank)
        envs["LOCAL_WORLD_SIZE"] = str(nproc_per_node)
        envs["WORLD_SIZE"] = str(nproc_per_node)
        fn_env = EnvWrapper(train, envs)
        p = mp.Process(target=fn_env, args=(config,))
        p.start()
        processes.append(p)

    return processes


def cleanup_subprocesses():
    """Kill all registered multiprocessing processes"""
    for process in processes:
        try:
            if process.is_alive():  # Check if mp.Process is still running
                logger.info(f"Terminating process with PID {process.pid}")
                process.terminate()  # Try to terminate gracefully

                # Wait for a bit to see if it terminates
                process.join(timeout=3)

                # If it's still alive, force kill it
                if process.is_alive():
                    logger.info(f"Process {process.pid} didn't terminate, killing...")
                    # On Unix, we can use os.kill for a hard kill
                    try:
                        os.kill(process.pid, signal.SIGKILL)
                    except Exception as e:
                        logger.info(f"Failed to kill process {process.pid}: {e}")
        except Exception as e:
            logger.info(f"Error cleaning up process: {e}")


def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, cleaning up...")
    cleanup_subprocesses()
    sys.exit(0)


def main(config: Config):
    if config.n_gpus is None:
        config.n_gpus = torch.cuda.device_count()

    gpus_ids = list(range(config.n_gpus))
    cutoff = int(config.n_gpus * config.ratio)

    train_gpus_ids = gpus_ids[:cutoff]
    inference_gpus_ids = gpus_ids[cutoff:]

    logger.info(f"start rl training with {len(train_gpus_ids)} GPUs, {len(inference_gpus_ids)}. Total: {len(gpus_ids)}")
    logger.info(f"train_gpus_ids: {train_gpus_ids}")
    logger.info(f"inference_gpus_ids: {inference_gpus_ids}")

    if config.rollout_path is not None:
        logger.info(f"Removing rollout path {config.rollout_path}")
        shutil.rmtree(config.rollout_path, ignore_errors=True)
    if config.rollout_data is not None:
        logger.info(f"Removing rollout data {config.rollout_data}")
        shutil.rmtree(config.rollout_data, ignore_errors=True)

    mp.set_start_method("spawn", force=True)

    train_processes = train_torchrun(
        config.train, rdzv_address=config.torchrun_rdzv_address, rdzv_port=config.torchrun_rdzv_port, gpus_ids=train_gpus_ids
    )

    config.inference.gpus_ids = inference_gpus_ids
    inference_process = inference_run(config.inference)

    processes.extend(train_processes)
    processes.extend(inference_process)

    try:
        while any(p.is_alive() for p in processes):
            for p in processes:
                if not p.is_alive() and p.exitcode != 0:
                    logger.info(f"Process {p.pid} died with exit code {p.exitcode}, terminating all processes")
                    cleanup_subprocesses()
                    sys.exit(1)
            # Sleep to avoid high CPU usage
            time.sleep(1)
    except Exception as e:
        logger.info(f"Error in main process: {e}")
        cleanup_subprocesses()
        sys.exit(1)


if __name__ == "__main__":
    # Register cleanup function to be called when the program exits
    atexit.register(cleanup_subprocesses)

    # Register signal handlers to handle SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    #### the code above alow to kill all subprocess like torchrun do

    logger = get_logger("RL_LAUNCHER")
    main(Config(**parse_argv()))
