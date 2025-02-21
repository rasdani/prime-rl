import pyarrow as pa
import pyarrow.parquet as pq
from zeroband.inference import pa_schema
from pathlib import Path
import os
import pytest


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


@pytest.fixture
def fake_rollout_files_dir(tmp_path):
    def _create(step: int = 0, num_files: int = 1):
        os.makedirs(tmp_path, exist_ok=True)
        _create_fake_rollout_parquet_file(tmp_path, step, num_files)
        return tmp_path

    return _create
