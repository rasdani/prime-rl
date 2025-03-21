from pydantic_config import parse_argv
import torch.distributed as dist
from transformers import AutoTokenizer
from zeroband.training.data import get_dataloader
from zeroband.train import Config, get_gradient_accumulation_steps
from zeroband.training.world_info import get_world_info


def main(config: Config):
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

    train_dataloader, _ = get_dataloader(
        tokenizer=tokenizer,
        micro_batch_size=config.train.micro_bs,
        batch_size=config.optim.batch_size * config.optim.step_per_rollout,
        data_config=config.data,
    )

    world_info = get_world_info()
    gradient_accumulation_steps = get_gradient_accumulation_steps(
        config.optim.batch_size, config.train.micro_bs, data_workers=config.data.num_workers, world_info=world_info
    )

    train_dataloader_iterator = iter(train_dataloader)

    step = 0
    while True:
        avg_rewards = 0
        non_masked_avg_rewards = 0
        for i in range(config.optim.step_per_rollout):
            batch_rewards = 0
            non_masked_rewards = 0
            for j in range(gradient_accumulation_steps):
                batch = next(train_dataloader_iterator)
                rewards = batch["rewards"]
                mask = batch["loss_mask"].bool()
                batch_rewards += rewards[mask].mean() / gradient_accumulation_steps
                non_masked_rewards += rewards.mean() / gradient_accumulation_steps

            batch_rewards = batch_rewards / world_info.world_size
            non_masked_rewards = non_masked_rewards / world_info.world_size

            dist.all_reduce(batch_rewards, op=dist.ReduceOp.SUM)
            dist.all_reduce(non_masked_rewards, op=dist.ReduceOp.SUM)

            avg_rewards += batch_rewards / config.optim.step_per_rollout
            non_masked_avg_rewards += non_masked_rewards / config.optim.step_per_rollout
            step += 1

        print(f"[rank {world_info.rank}] step {step} rewards: {avg_rewards:.4f}, non-masked rewards: {non_masked_avg_rewards:.4f}")

        if step >= config.optim.total_steps:
            break


if __name__ == "__main__":
    dist.init_process_group()
    main(Config(**parse_argv()))
