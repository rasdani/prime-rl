import subprocess
import pytest


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


def test_inference():
    _test_torchrun(config="inference/debug.toml")
