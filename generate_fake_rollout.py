# this is a script to generate fake rollout data for testing
# should be removed once we have inference working
# uv run python generate_fake_rollout.py  @ configs/training/150M/A40.toml --data.path data/fake_rollout --optim.total_steps 1000

import os
from pathlib import Path
from pydantic_config import parse_argv
from zeroband.train import Config

from tests.conftest import _create_fake_rollout_parquet_file


def main(config: Config):
    path = Path(config.data.path)
    os.makedirs(path, exist_ok=True)

    num_files = 4
    batch_size = config.optim.batch_size // num_files

    _create_fake_rollout_parquet_file(
        path, list(range(config.optim.total_steps)), num_files=num_files, batch_size=batch_size, seq_len=config.data.seq_length
    )

    print(f"Created test data at: {path}")


if __name__ == "__main__":
    config = Config(**parse_argv())  # type: ignore
    main(config)
