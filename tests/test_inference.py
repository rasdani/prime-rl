import subprocess
import pytest
import pyarrow.parquet as pq


def _test_torchrun(config, extra_args=[]):
    cmd = [
        "python",
        "src/zeroband/inference.py",
        f"@configs/{config}",
        *extra_args,
    ]

    process = subprocess.Popen(cmd)
    result = process.wait()
    if result != 0:
        pytest.fail(f"Process {result} failed {result}")


@pytest.mark.parametrize("tp", [1, 2])
def test_inference(tmp_path, tp):
    _test_torchrun(config="inference/debug.toml", extra_args=["--output_path", str(tmp_path), "--tp", str(tp)])

    assert tmp_path.joinpath("step_0").exists()

    for file in tmp_path.joinpath("step_0").iterdir():
        if file.suffix == ".parquet":
            table = pq.read_table(file)
            assert table.num_rows > 0
