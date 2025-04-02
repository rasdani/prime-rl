# this is a script to generate fake rollout data for testing
# should be removed once we have inference working
# uv run python generate_fake_rollout.py  @ configs/training/150M/A40.toml --data.path data/fake_rollout --optim.total_steps 1000

import os
from pathlib import Path
from pydantic_config import parse_argv
from zeroband.inference import Config

from tests.conftest import _create_fake_rollout_parquet_file


def main(config: Config):
    path = Path(config.output_path)
    os.makedirs(path, exist_ok=True)

    total_step = config.max_samples // config.batch_size

    _create_fake_rollout_parquet_file(
        path, list(range(total_step)), num_files=num_files, batch_size=config.batch_size, seq_len=config.sampling.max_tokens
    )

    print(f"Created test data at: {path}")


if __name__ == "__main__":
    config = Config(**parse_argv())  # type: ignore
    main(config)
