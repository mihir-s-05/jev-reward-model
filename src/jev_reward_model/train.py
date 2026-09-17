from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor

from .config import ExperimentConfig
from .data import load
from .jev import JevClient
from .judge import LocalQwenJudge
from .ppo import PPORollout, ValueHead, action_logprob, add_gae, ppo_loss, state_value
from .rewards import GroundedReward, JevPotentialShapingReward, JevTerminalReward, QwenJudgeReward
from .rollout import run_episode


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_policy(cfg: ExperimentConfig):
    processor = AutoProcessor.from_pretrained(cfg.model)
    base = AutoModelForMultimodalLM.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    hidden_size = model.config.text_config.hidden_size
    value_head = ValueHead(hidden_size).to(model.device)
    return model, value_head, processor


def make_reward(cfg: ExperimentConfig, frozen_judge=None):
    if cfg.reward == "grounded":
        return GroundedReward()
    if cfg.reward == "jev_terminal":
        return JevTerminalReward(JevClient())
    if cfg.reward == "jev_shaping":
        return JevPotentialShapingReward(JevClient(), cfg.gamma, cfg.shaping_alpha)
    if cfg.reward == "qwen_judge":
        return QwenJudgeReward(frozen_judge)
    raise ValueError(cfg.reward)


def collect_episode(model, value_head, processor, task, reward_source, cfg):
    raw_records, final = run_episode(
        model,
        processor,
        task,
        reward_source,
        cfg.max_episode_steps,
        cfg.max_new_tokens,
    )
    rollout: list[PPORollout] = []
    for record in raw_records:
        prompt_ids = record["inputs"]["input_ids"].detach()
        generated = record["generated"].detach()
        with torch.no_grad():
            old_lp = action_logprob(model, prompt_ids, generated).detach()
            old_v = state_value(model, value_head, prompt_ids).detach()
        rollout.append(
            PPORollout(
                prompt_ids=prompt_ids,
                generated_ids=generated,
                old_logprob=old_lp,
                old_value=old_v,
                reward=float(record["reward"]),
                done=bool(record["done"]),
            )
        )
    add_gae(rollout, cfg.gamma, cfg.gae_lambda)
    return rollout, final


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = ExperimentConfig.load(args.config)
    seed_all(cfg.seed)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load(cfg.train_data)
    model, value_head, processor = load_policy(cfg)

    frozen_judge = None
    if cfg.reward == "qwen_judge":
        judge_model = AutoModelForMultimodalLM.from_pretrained(
            cfg.model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        frozen_judge = LocalQwenJudge(judge_model, processor)
    reward_source = make_reward(cfg, frozen_judge)

    trainable = [
        p for p in list(model.parameters()) + list(value_head.parameters()) if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
    log_path = out / "train.jsonl"

    for step in range(cfg.steps):
        batch: list[PPORollout] = []
        episodic = []
        model.eval()
        for _ in range(cfg.rollouts_per_step):
            task = random.choice(tasks)
            rollout, final = collect_episode(
                model, value_head, processor, task, reward_source, cfg
            )
            batch.extend(rollout)
            episodic.append(
                {
                    "task": task.task_id,
                    "sim_success": final.success,
                    "model_reward": sum(x.reward for x in rollout),
                }
            )

        advantages = torch.cat([x.advantage.reshape(-1) for x in batch])
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False).clamp_min(1e-6)
        for item in batch:
            item.advantage = (item.advantage - adv_mean) / adv_std

        model.train()
        epoch_losses = []
        for _ in range(cfg.ppo_epochs):
            random.shuffle(batch)
            for item in batch:
                optimizer.zero_grad()
                loss = ppo_loss(
                    model,
                    value_head,
                    item,
                    cfg.clip_range,
                    cfg.value_coef,
                    cfg.entropy_coef,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().item()))

        row = {
            "step": step,
            "loss": float(np.mean(epoch_losses)),
            "mean_model_reward": float(np.mean([x["model_reward"] for x in episodic])),
            "sim_success": float(np.mean([x["sim_success"] for x in episodic])),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if (step + 1) % 25 == 0:
            ckpt = out / f"checkpoint-{step + 1}"
            model.save_pretrained(ckpt)
            torch.save(value_head.state_dict(), ckpt / "value_head.pt")


if __name__ == "__main__":
    main()
