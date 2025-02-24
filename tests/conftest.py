import pyarrow as pa
import pyarrow.parquet as pq
from zeroband.inference import pa_schema
from pathlib import Path
import os
import pytest

from zeroband.training.data import STABLE_FILE


def _create_one_pa_file(file_name: str, batch_size: int = 1):
    # Initialize lists for each column
    input_tokens_list = [[1] * 10 for _ in range(batch_size)]  # Wrap in list
    output_tokens_list = [[1] * 10 for _ in range(batch_size)]  # Wrap in list
    advantages_list = [1] * batch_size
    proofs_list = [b"I am toploc proof, handcrafted by jack"] * batch_size
    steps_list = [0] * batch_size

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


def _create_fake_rollout_parquet_file(path: Path, steps: list[int], num_files: int, batch_size: int):
    for s in steps:
        step_path = path / f"step_{s}"

        for i in range(num_files):
            os.makedirs(step_path, exist_ok=True)
            _create_one_pa_file(step_path / f"{i}.parquet", batch_size)

        stable_file = step_path / STABLE_FILE
        with open(stable_file, "w"):
            pass


@pytest.fixture
def fake_rollout_files_dir(tmp_path):
    def _create(steps: list[int] = [0], num_files: int = 1, batch_size: int = 1):
        os.makedirs(tmp_path, exist_ok=True)
        _create_fake_rollout_parquet_file(tmp_path, steps, num_files, batch_size)
        return tmp_path

    return _create
