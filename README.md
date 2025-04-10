# prime-rl - decentralized RL training at scale

fork of prime to add RL training at scale. Private for now


quick install
```
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/install.sh | bash
```


## Dev


1. Clone: 

```bash
git clone git@github.com:PrimeIntellect-ai/prime-rl.git
cd prime-rl
```

2. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Set up the environment:
```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync
uv pip install flash-attn --no-build-isolation
```

4. Precommit install

```bash
uv run pre-commit install
```

5. Test

```bash
uv run pytest
```

6. debug run 

training

```bash
uv run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/debug.toml
```

inference
```bash
uv run python src/zeroband/inference.py @ configs/inference/debug.toml
```


## Debug math run

on two different terminal do:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
uv run python src/zeroband/inference.py @ configs/inference/Qwen1.5B/debug_math.toml
```

then start the trainer

```bash
ulimit -n 4096
export CUDA_VISIBLE_DEVICES=6,7
uv  run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/Qwen1.5B/debug_math.toml
```


## 2k seq length run

on two different terminal do:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
uv run python src/zeroband/inference.py @ configs/inference/Qwen1.5B/Qwen1.5B.toml
```

then start the trainer

```bash
ulimit -n 4096
export CUDA_VISIBLE_DEVICES=6,7
uv  run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/Qwen1.5B/Qwen1.5b.toml
```

if running on h100 node instead of H200 you should add ` --train.micro_bs 4`



## Checkpoints management

To save a checkpoint to gcp you need to:

authentificate
```bash
gcloud auth login yourname@primeintellect.ai
```

then to push a file or a folder

```bash
gsutil -m cp -r yourfile gs://workspaces_research/yourname/yourfolder/.
```

to download a file or a folder

```bash
gcloud storage cp -r  gs://workspaces_research/yourname/yourfile .
```


