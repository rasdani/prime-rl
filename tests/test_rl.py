import subprocess
import pytest
from pathlib import Path


def test_rl_launcher(tmp_path: Path):
    """
    this test will create a real rollout from inference and update the model
    """
    rollout_data = tmp_path / "rollout"
    outputs = tmp_path / "outputs"
    cmd = f"uv run src/zeroband/rl_launcher.py @ configs/rl_debug.toml --rollout_path {outputs} --rollout_data {rollout_data}".split()
    process = subprocess.Popen(cmd)
    result = process.wait()
    if result != 0:
        pytest.fail(f"Process  failed {result}")

    assert outputs.exists()
    assert rollout_data.exists()
