import os
import subprocess
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from zeroband.inference import pa_schema
from pathlib import Path


def _test_torchrun(num_gpus, config, extra_args=[]):
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "src/zeroband/train.py",
        f"@configs/training/{config}",
        *extra_args,
    ]

    process = subprocess.Popen(cmd)
    result = process.wait()
    if result != 0:
        pytest.fail(f"Process {result} failed {result}")


@pytest.mark.parametrize("num_gpus", [1, 2])
def test_train(num_gpus):
    _test_torchrun(num_gpus=num_gpus, config="debug.toml")


def _create_one_pa_file(file_name: str):
    # Initialize lists for each column
    input_tokens_list = [[1] * 10]  # Wrap in list
    output_tokens_list = [[1] * 100]  # Wrap in list
    advantages_list = [1] * len(output_tokens_list)
    proofs_list = [b"I am toploc proof, handcrafted by jack"] * len(output_tokens_list)
    steps_list = [0] * len(output_tokens_list)

    # Create PyArrow arrays
    arrays = [
        pa.array(input_tokens_list, type=pa.list_(pa.int32())),
        pa.array(output_tokens_list, type=pa.list_(pa.int32())),
        pa.array(advantages_list, type=pa.float32()),
        pa.array(proofs_list, type=pa.binary()),
        pa.array(steps_list, type=pa.int32()),
    ]

    # Create and return table
    table = pa.Table.from_arrays(arrays, schema=pa_schema)

    pq.write_table(table, file_name)


def _create_fake_rollout_parquet_file(path: Path, step: int, num_files: int):
    path = path / f"step_{step}"
    for i in range(num_files):
        os.makedirs(path, exist_ok=True)
        _create_one_pa_file(path / f"{i}.parquet")


def test_train_with_rollout_file(tmp_path: Path):
    """
    this test will create a fake rollout file and then train with it
    """
    path = tmp_path / "test_train_with_rollout_file"
    path.mkdir(parents=True, exist_ok=True)
    _create_fake_rollout_parquet_file(path, 0, 10)
    _test_torchrun(num_gpus=1, config="debug.toml", extra_args=["--data.path", str(path), "--no-data.fake"])
