import os
import shutil
import sys
import time
from copy import deepcopy
from subprocess import Popen
from threading import Event, Thread
from typing import Annotated, Literal

import tomli_w
import torch
from pydantic import Field, model_validator

from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.trainer.config import CheckpointConfig, FakeDataLoaderConfig, TrainerConfig
from prime_rl.utils.config import WandbMonitorConfig
from prime_rl.utils.pydantic_config import BaseSettings, get_temp_toml_file, parse_argv

# Import the original functions from rl.py
from prime_rl.rl import (
    LogConfig,
    setup_logger,
    cleanup_threads,
    cleanup_processes,
    monitor_process,
)


class MultiNodeConfig(BaseSettings):
    """Multi-node specific configuration."""
    
    mode: Annotated[
        Literal["inference", "training", "both"],
        Field(description="Node role: 'inference' (runs inference+orchestrator), 'training' (runs training only), 'both' (single node mode)")
    ] = "both"
    
    master_addr: Annotated[
        str,
        Field(description="Master node IP address for coordination")
    ] = "localhost"
    
    master_port: Annotated[
        int,
        Field(description="Master node port for torchrun coordination")
    ] = 29500
    
    inference_addr: Annotated[
        str,
        Field(description="Inference node IP address (for training nodes to connect to)")
    ] = "localhost"
    
    inference_port: Annotated[
        int,
        Field(description="Inference server port")
    ] = 8000
    
    nnodes: Annotated[
        int,
        Field(description="Total number of nodes")
    ] = 1
    
    node_rank: Annotated[
        int,
        Field(description="Rank of this node (0-based)")
    ] = 0


class MultiNodeRLConfig(BaseSettings):
    """Multi-node RL training configuration."""

    trainer: TrainerConfig
    orchestrator: OrchestratorConfig
    inference: Annotated[
        InferenceConfig | None,
        Field(description="The inference config. If None, will not start an inference process.")
    ] = None

    log: LogConfig = LogConfig()
    multinode: MultiNodeConfig = MultiNodeConfig()

    exp_id: Annotated[str | None, Field(description="The experiment ID.")] = "rl-multinode"

    trainer_gpus: Annotated[int, Field(description="The number of GPUs to use for trainer.")] = 1
    inference_gpus: Annotated[int, Field(description="The number of GPUs to use for inference.")] = 1

    bench: Annotated[bool, Field(description="Whether to run in benchmark mode.")] = False
    clean: Annotated[bool, Field(description="Whether to clean directories at start.")] = True

    @model_validator(mode="after")
    def validate_multinode_config(self):
        # For split-node setups, adjust GPU allocations
        if self.multinode.mode == "inference":
            # Inference node uses all available GPUs for inference
            available_gpus = torch.cuda.device_count()
            self.inference_gpus = available_gpus
            self.trainer_gpus = 0
            
        elif self.multinode.mode == "training":
            # Training node uses all available GPUs for training
            available_gpus = torch.cuda.device_count()
            self.trainer_gpus = available_gpus
            self.inference_gpus = 0
            self.inference = None  # No inference process on training node
            
        elif self.multinode.mode == "both":
            # Single node mode - validate normal constraints
            available_gpus = torch.cuda.device_count()
            if self.trainer_gpus + self.inference_gpus > available_gpus:
                raise ValueError(
                    f"Total GPUs ({self.trainer_gpus + self.inference_gpus}) exceeds available ({available_gpus})"
                )
                
        return self

    @model_validator(mode="after")
    def setup_multinode_networking(self):
        # Configure network settings for multi-node
        if self.multinode.mode == "inference" and self.inference:
            # Inference node: listen on all interfaces
            self.inference.server.host = "0.0.0.0"
            self.inference.server.port = self.multinode.inference_port
            
        elif self.multinode.mode == "training":
            # Training node: connect to inference node
            self.orchestrator.client.host = self.multinode.inference_addr
            self.orchestrator.client.port = self.multinode.inference_port
            
        return self

    @model_validator(mode="after")
    def auto_setup_inference_parallelism(self):
        # Configure inference parallelism for multi-GPU setups
        if self.inference and self.inference_gpus > 1:
            # Use data parallelism across all inference GPUs
            self.inference.parallel.dp = self.inference_gpus
            self.inference.parallel.tp = 1
            
        return self

    # Import all other validators from the original RLConfig
    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            self.trainer.bench = True
            self.orchestrator.bench = True
            self.trainer.data.fake = FakeDataLoaderConfig(
                micro_batch_size=self.orchestrator.micro_batch_size,
                batch_size=self.orchestrator.batch_size,
                seq_len=self.orchestrator.seq_len,
            )
        return self

    @model_validator(mode="after")
    def auto_setup_logs(self):
        self.trainer.log.level = self.log.level
        self.orchestrator.log.level = self.log.level
        return self

    @model_validator(mode="after")
    def auto_setup_wandb(self):
        if self.orchestrator and self.trainer.monitor.wandb:
            if not self.orchestrator.monitor.wandb:
                self.orchestrator.monitor.wandb = WandbMonitorConfig()
            self.orchestrator.monitor.wandb.project = self.trainer.monitor.wandb.project
            if self.trainer.monitor.wandb.name:
                run_name = deepcopy(self.trainer.monitor.wandb.name)
                self.trainer.monitor.wandb.name = f"{run_name}-trainer"
                self.orchestrator.monitor.wandb.name = f"{run_name}-orchestrator"
        return self

    @model_validator(mode="after")
    def auto_setup_model(self):
        self.orchestrator.model.name = self.trainer.model.name
        if self.inference:
            self.inference.model.name = self.trainer.model.name
        return self

    @model_validator(mode="after")
    def auto_setup_max_step(self):
        if self.trainer.max_steps is not None:
            self.orchestrator.max_steps = self.trainer.max_steps
        return self

    @model_validator(mode="after")
    def auto_setup_async_level(self):
        self.orchestrator.async_level = self.trainer.async_level
        return self

    @model_validator(mode="after")
    def auto_setup_exp(self):
        if self.exp_id:
            self.log.path = self.log.path / self.exp_id
            self.trainer.data.path = self.trainer.data.path / self.exp_id
            self.trainer.weights.path = self.trainer.weights.path / self.exp_id
        return self

    @model_validator(mode="after")
    def auto_setup_paths(self):
        self.orchestrator.rollout_path = self.trainer.data.path
        self.orchestrator.weights_path = self.trainer.weights.path
        return self

    @model_validator(mode="after")
    def auto_setup_ckpt(self):
        if self.trainer.ckpt:
            self.orchestrator.ckpt = CheckpointConfig()
            self.orchestrator.ckpt.path = self.trainer.ckpt.path
            self.orchestrator.ckpt.interval = self.trainer.ckpt.interval
            if self.trainer.ckpt.resume_step:
                self.orchestrator.ckpt.resume_step = self.trainer.ckpt.resume_step
        return self

    @model_validator(mode="after")
    def auto_setup_num_train_workers(self):
        if self.trainer_gpus > 1:
            self.orchestrator.num_train_workers = self.trainer_gpus
        return self


def wait_for_inference_server(host: str, port: int, timeout: int = 60) -> bool:
    """Wait for inference server to be ready."""
    import requests
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def rl_multinode(config: MultiNodeRLConfig):
    """Multi-node RL training with support for split inference/training nodes."""
    
    # Setup logger
    logger = setup_logger(config.log)
    logger.info(f"Starting multi-node RL run (mode: {config.multinode.mode})")
    
    # Clean directories if requested
    if config.clean and config.multinode.mode != "training":
        logger.info("Cleaning directories")
        for path in [config.log.path, config.trainer.data.path, config.trainer.weights.path]:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                
        if config.trainer.ckpt and not config.trainer.ckpt.resume_step:
            if config.trainer.ckpt.path.exists():
                shutil.rmtree(config.trainer.ckpt.path, ignore_errors=True)

    # Create necessary directories
    config.log.path.mkdir(parents=True, exist_ok=True)
    config.trainer.data.path.mkdir(parents=True, exist_ok=True)
    config.trainer.weights.path.mkdir(parents=True, exist_ok=True)

    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    try:
        # INFERENCE NODE: Start inference server and orchestrator
        if config.multinode.mode in ["inference", "both"]:
            if config.inference:
                logger.info(f"Starting inference server on {config.inference_gpus} GPUs")
                inference_file = get_temp_toml_file()
                with open(inference_file, "wb") as f:
                    tomli_w.dump(config.inference.model_dump(exclude_none=True, mode="json"), f)

                inference_cmd = ["uv", "run", "inference", "@", inference_file.as_posix()]
                
                with open(config.log.path / "inference.log", "w") as log_file:
                    inference_process = Popen(
                        inference_cmd,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(config.inference_gpus)))},
                        stdout=log_file,
                        stderr=log_file,
                    )
                processes.append(inference_process)

                # Monitor inference process
                stop_event = Event()
                stop_events["inference"] = stop_event
                monitor_thread = Thread(
                    target=monitor_process,
                    args=(inference_process, stop_event, error_queue, "inference"),
                    daemon=True,
                )
                monitor_thread.start()
                monitor_threads.append(monitor_thread)
                
                # Wait for inference server to be ready
                logger.info("Waiting for inference server to start...")
                if not wait_for_inference_server(config.multinode.inference_addr, config.multinode.inference_port):
                    raise RuntimeError("Inference server failed to start")
                logger.success("Inference server ready!")

            # Start orchestrator
            logger.info("Starting orchestrator")
            orchestrator_file = get_temp_toml_file()
            with open(orchestrator_file, "wb") as f:
                tomli_w.dump(config.orchestrator.model_dump(exclude_none=True, mode="json"), f)

            orchestrator_cmd = ["uv", "run", "orchestrator", "@", orchestrator_file.as_posix()]
            
            with open(config.log.path / "orchestrator.log", "w") as log_file:
                orchestrator_process = Popen(
                    orchestrator_cmd,
                    stdout=log_file,
                    stderr=log_file,
                    env={**os.environ, "LOGURU_FORCE_COLORS": "1"},
                )
            processes.append(orchestrator_process)

            stop_event = Event()
            stop_events["orchestrator"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(orchestrator_process, stop_event, error_queue, "orchestrator"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)

        # TRAINING NODE: Wait for inference, then start training
        if config.multinode.mode in ["training", "both"]:
            # For training nodes, wait for inference server
            if config.multinode.mode == "training":
                logger.info(f"Waiting for inference server at {config.multinode.inference_addr}:{config.multinode.inference_port}")
                if not wait_for_inference_server(config.multinode.inference_addr, config.multinode.inference_port):
                    raise RuntimeError("Cannot connect to inference server")
                logger.success("Connected to inference server!")

            # Start training
            logger.info(f"Starting training on {config.trainer_gpus} GPUs")
            trainer_file = get_temp_toml_file()
            with open(trainer_file, "wb") as f:
                tomli_w.dump(config.trainer.model_dump(exclude_none=True, mode="json"), f)

            trainer_cmd = [
                "uv", "run", "torchrun",
                f"--rdzv-backend=c10d",
                f"--rdzv-endpoint={config.multinode.master_addr}:{config.multinode.master_port}",
                f"--rdzv-id={config.exp_id}",
                f"--nnodes={config.multinode.nnodes}",
                f"--node-rank={config.multinode.node_rank}",
                f"--nproc-per-node={config.trainer_gpus}",
                "src/prime_rl/trainer/train.py",
                "@", trainer_file.as_posix(),
            ]

            gpu_ids = list(range(config.trainer_gpus))
            logger.info(f"Training command: {' '.join(trainer_cmd)}")
            
            with open(config.log.path / "trainer.log", "w") as log_file:
                trainer_process = Popen(
                    trainer_cmd,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
                        "LOGURU_FORCE_COLORS": "1",
                    },
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(trainer_process)

            stop_event = Event()
            stop_events["trainer"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(trainer_process, stop_event, error_queue, "trainer"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)

        # Monitor processes
        if config.multinode.mode == "inference":
            logger.success("Inference node ready! Waiting for training to complete...")
            main_process = "orchestrator"
        elif config.multinode.mode == "training":
            logger.success("Training node started!")
            main_process = "trainer" 
        else:
            logger.success("Single-node training started!")
            main_process = "trainer"

        # Show logs from main process
        if main_process in stop_events:
            tail_process = Popen(["tail", "-F", config.log.path / f"{main_process}.log"])
            processes.append(tail_process)

            # Wait for completion
            while not stop_events[main_process].is_set():
                if error_queue:
                    error = error_queue[0]
                    logger.error(f"Error: {error}")
                    raise error
                time.sleep(1)

        logger.success("Multi-node RL training completed!")

    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating all processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        raise
    finally:
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)


def main():
    rl_multinode(parse_argv(MultiNodeRLConfig))


if __name__ == "__main__":
    main()