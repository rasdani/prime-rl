import atexit
import os
import signal
import sys

from pydantic_config import parse_argv, BaseConfig
from zeroband.train import Config as TrainConfig
from zeroband.inference import Config as InferenceConfig
from zeroband.inference import main as inference
from zeroband.train import train
from zeroband.logger import get_logger

import torch.multiprocessing as mp

processes = []


class Config(BaseConfig):
    train: TrainConfig
    inference: InferenceConfig

    n_gpus: int
    ratio: float = 0.5  # for now we have half train and half inference

    torchrun_rdzv_address: str = "localhost"
    torchrun_rdzv_port: int = 29500


class EnvWrapper:
    """
    This class wrapp a function call and overide the environment variables
    FYI: cannot use a simple function because of pickle issues
    """

    def __init__(self, fn, envs):
        self.fn = fn
        self.envs = envs

    def __call__(self, *args, **kwargs):
        os.environ.update(self.envs)
        return self.fn(*args, **kwargs)


def _cuda_available_devices(gpus_ids: list[int]) -> str:
    return ",".join(map(str, gpus_ids))


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
        envs["CUDA_VISIBLE_DEVICES"] = _cuda_available_devices(gpus_ids)
        envs["RANK"] = str(rank)
        envs["LOCAL_RANK"] = str(rank)
        envs["LOCAL_WORLD_SIZE"] = str(nproc_per_node)
        envs["WORLD_SIZE"] = str(nproc_per_node)
        fn_env = EnvWrapper(train, envs)
        p = mp.Process(target=fn_env, args=(config,))
        p.start()
        processes.append(p)

    return processes


def inference_run(config: InferenceConfig, gpus_ids: list[int]) -> list[mp.Process]:
    """
    This function is used to run inference by creating a sub process.
    """
    envs = {"CUDA_VISIBLE_DEVICES": _cuda_available_devices(gpus_ids)}

    config.tp = len(gpus_ids)

    fn_env = EnvWrapper(inference, envs)
    process = mp.Process(target=fn_env, args=(config,))
    process.start()

    return [process]


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
    gpus_ids = list(range(config.n_gpus))
    cutoff = int(config.n_gpus * config.ratio)

    train_gpus_ids = gpus_ids[cutoff:]
    inference_gpus_ids = gpus_ids[:cutoff]

    logger.info(f"start rl training with {len(train_gpus_ids)} GPUs, {len(inference_gpus_ids)}. Total: {len(gpus_ids)}")
    logger.info(f"train_gpus_ids: {train_gpus_ids}")
    logger.info(f"inference_gpus_ids: {inference_gpus_ids}")

    mp.set_start_method("spawn", force=True)

    train_processes = train_torchrun(
        config.train, rdzv_address=config.torchrun_rdzv_address, rdzv_port=config.torchrun_rdzv_port, gpus_ids=train_gpus_ids
    )
    inference_process = inference_run(config.inference, gpus_ids=inference_gpus_ids)

    processes.extend(train_processes)
    processes.extend(inference_process)

    try:
        for p in processes:
            p.join()
    except Exception as e:
        logger.info(f"Error in main process: {e}")
        # The cleanup will happen via atexit


if __name__ == "__main__":
    # Register cleanup function to be called when the program exits
    atexit.register(cleanup_subprocesses)

    # Register signal handlers to handle SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    #### the code above alow to kill all subprocess like torchrun do

    logger = get_logger("RL_LAUNCHER")
    main(Config(**parse_argv()))
