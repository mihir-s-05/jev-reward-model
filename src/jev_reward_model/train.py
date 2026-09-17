from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor

from .config import ExperimentConfig
from .data import load
from .jev import JevClient
from .judge import LocalQwenJudge
from .rewards import GroundedReward, JevPotentialShapingReward, JevTerminalReward, QwenJudgeReward
from .rollout import run_episode


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_policy(cfg: ExperimentConfig):
    processor = AutoProcessor.from_pretrained(cfg.model)
    base = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, device_map="auto")
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    return model, processor


def sequence_logprob(model, inputs, generated):
    prompt = inputs["input_ids"]
    full = torch.cat([prompt, generated], dim=1)
    attention = torch.ones_like(full)
    logits = model(input_ids=full, attention_mask=attention).logits[:, :-1]
    labels = full[:, 1:]
    token_lp = F.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = prompt.shape[1] - 1
    return token_lp[:, start:].sum(dim=1)


def make_reward(cfg, frozen_judge=None):
    if cfg.reward == "grounded": return GroundedReward()
    if cfg.reward == "jev_terminal": return JevTerminalReward(JevClient())
    if cfg.reward == "jev_shaping": return JevPotentialShapingReward(JevClient(), cfg.gamma, cfg.shaping_alpha)
    if cfg.reward == "qwen_judge": return QwenJudgeReward(frozen_judge)
    raise ValueError(cfg.reward)


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out, ret = [], 0.0
    for r in reversed(rewards):
        ret = r + gamma * ret
        out.append(ret)
    return list(reversed(out))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = ExperimentConfig.load(args.config)
    seed_all(cfg.seed)
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    tasks = load(cfg.train_data)
    model, processor = load_policy(cfg)

    frozen_judge = None
    if cfg.reward == "qwen_judge":
        judge_model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, device_map="auto")
        frozen_judge = LocalQwenJudge(judge_model, processor)
    reward_source = make_reward(cfg, frozen_judge)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # Compact REINFORCE-style clipped policy update. This intentionally avoids a large
    # trainer abstraction so reward semantics remain obvious. A learned critic is the
    # natural next extension; this first comparison isolates reward-source quality.
    baseline = 0.0
    beta = 0.95
    log_path = out / "train.jsonl"
    for step in range(cfg.steps):
        optimizer.zero_grad()
        losses, episodic = [], []
        for _ in range(cfg.rollouts_per_step):
            task = random.choice(tasks)
            records, final = run_episode(model, processor, task, reward_source, cfg.max_episode_steps, cfg.max_new_tokens)
            returns = discounted_returns([r["reward"] for r in records], cfg.gamma)
            episodic.append({"task": task.task_id, "sim_success": final.success, "reward": sum(r["reward"] for r in records)})
            for record, ret in zip(records, returns):
                lp = sequence_logprob(model, record["inputs"], record["generated"])
                advantage = ret - baseline
                losses.append(-(lp * advantage).mean())
                baseline = beta * baseline + (1 - beta) * ret
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {"step": step, "loss": float(loss.item()), "mean_reward": float(np.mean([x["reward"] for x in episodic])), "sim_success": float(np.mean([x["sim_success"] for x in episodic]))}
        with log_path.open("a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
        if (step + 1) % 25 == 0:
            model.save_pretrained(out / f"checkpoint-{step + 1}")


if __name__ == "__main__":
    main()
