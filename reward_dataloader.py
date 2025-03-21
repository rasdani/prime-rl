import os
from pydantic_config import parse_argv
import torch
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

    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    step = 0
    while True:
        if step % config.optim.step_per_rollout == 0:
            avg_rewards = 0
            non_masked_avg_rewards = 0
            mask_avg_rollout = 0
            mask_bool_rollout = 0
            rewards_bool = 0
            proper_rewards_rollout = 0

        batch_rewards = 0
        non_masked_rewards = 0
        mask_avg = 0
        mask_bool_avg = 0
        batch_rewards_bool = 0

        proper_rewards_sum = torch.tensor(0.0)
        proper_rewards_token_count = torch.tensor(0.0)

        for j in range(gradient_accumulation_steps):
            batch = next(train_dataloader_iterator)
            rewards = batch["rewards"]
            mask = batch["loss_mask"].float()
            mask_bool = batch["loss_mask"].bool()
            mask_avg += mask.mean() / gradient_accumulation_steps
            mask_bool_avg += mask_bool.float().mean() / gradient_accumulation_steps
            batch_rewards += (rewards * mask).mean() / gradient_accumulation_steps
            batch_rewards_bool += (rewards[mask_bool]).mean() / gradient_accumulation_steps
            non_masked_rewards += rewards.mean() / gradient_accumulation_steps

            r = rewards[mask_bool]

            proper_rewards_sum += r.sum()
            proper_rewards_token_count += r.numel()

        batch_rewards = batch_rewards / world_info.world_size
        non_masked_rewards = non_masked_rewards / world_info.world_size
        mask_avg = mask_avg / world_info.world_size
        mask_bool_avg = mask_bool_avg / world_info.world_size
        batch_rewards_bool = batch_rewards_bool / world_info.world_size

        dist.all_reduce(proper_rewards_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(proper_rewards_token_count, op=dist.ReduceOp.SUM)
        proper_rewards = proper_rewards_sum / proper_rewards_token_count

        dist.all_reduce(batch_rewards, op=dist.ReduceOp.SUM)
        dist.all_reduce(non_masked_rewards, op=dist.ReduceOp.SUM)
        dist.all_reduce(mask_avg, op=dist.ReduceOp.SUM)
        dist.all_reduce(mask_bool_avg, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_rewards_bool, op=dist.ReduceOp.SUM)
        avg_rewards += batch_rewards / config.optim.step_per_rollout
        non_masked_avg_rewards += non_masked_rewards / config.optim.step_per_rollout
        mask_avg_rollout += mask_avg / config.optim.step_per_rollout
        mask_bool_rollout += mask_bool_avg / config.optim.step_per_rollout
        rewards_bool += batch_rewards_bool / config.optim.step_per_rollout
        proper_rewards_rollout += proper_rewards / config.optim.step_per_rollout
        step += 1

        if os.environ.get("RANK") == "0":
            print(f"step {step} , proper rewards: {proper_rewards:.4f}, avg_rewards: {avg_rewards:.4f}")
        if step % config.optim.step_per_rollout == 0:
            print(
                f"[rank {world_info.rank}] step {step}\n"
                f"rewards: {avg_rewards:.4f}\n"
                f"non-masked rewards: {non_masked_avg_rewards:.4f}\n"
                f"rewards bool: {rewards_bool:.4f}\n"
                f"mask avg: {mask_avg_rollout:.4f}\n"
                f"mask bool avg: {mask_bool_rollout:.4f}\n"
                f"proper rewards: {proper_rewards_rollout:.4f}\n"
            )

        if step >= config.optim.total_steps:
            break


if __name__ == "__main__":
    dist.init_process_group()
    main(Config(**parse_argv()))
