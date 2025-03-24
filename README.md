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

## Larger run

For now you need to generate fake rollout data for testing. 

```bash
uv run python generate_fake_rollout.py @ configs/inference/Qwen1.5B/Qwen1.5B.toml --max-samples 10000 --sampling.max_tokens 8192
```

and then do the training on it

```bash
uv run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/150M/A40.toml --data.path data/fake_rollout 
```

when using `on_policy_log_prob` you might need to do `ulimit -n 4096` to avoid crash.

## RL launcher

rl launcher is a script that allow to start training and inference at the same time and assign GPUs to each process.

Under the hood its just start script a bit like torchrun do.

```bash
uv run src/zeroband/rl_launcher.py @ configs/rl_debug.toml --rollout_path outputs --rollout_data data_rollout
```

You can pass any config that you would pass for training via `--train.<config_name>` and for inference via `--inference.<config_name>`.

In the future this launcher will make sure that both training and inference configs are compatible with each other. For now there is no specific config validation logic.

## manual 4k run

on two different terminal do:

```bash
export CUDA_VISIBLE_DEVICES=6,7
uv  run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/Qwen1.5B/Qwen1.5b.toml --data.path data_rollout --ckpt.rollout_path outputs --train.micro_bs 4 --data.seq_length 4096 --optim.batch_size 64 --optim.step_per_rollout 16 --train.attn_impl flash_attention_2
```

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
uv run python src/zeroband/inference.py @ configs/inference/Qwen1.5B/Qwen1.5B.toml --batch-size 22 --dp 6 --rollout_path outputs --output_path data_rollout --step_batch_size 132  --max_model_len 4096 --seed 42
```



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


