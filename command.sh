uv run torchrun --nproc_per_node=8 src/zeroband/train.py @ configs/training/Qwen1.5B/Qwen1.5b.toml --data.fake --train.micro_bs 1 --optim.total_steps=10
