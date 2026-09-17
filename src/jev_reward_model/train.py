"""Train one arm and seed; retain grounded outcomes and all evaluator inputs for audit."""
from __future__ import annotations

import argparse
import random
import sqlite3
import time
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import assert_disjoint, load
from .ppo import add_gae, update
from .rewards import assign_rewards, make_judge
from .utils import append_jsonl, provenance, write_json


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def summarize(episodes, gamma: float) -> dict:
    n = len(episodes)
    return {"success": sum(e.oracle["success"] for e in episodes) / n,
            "violation": sum(e.oracle["violation"] for e in episodes) / n,
            "progress": sum(e.oracle["progress"] for e in episodes) / n,
            "steps": sum(len(e.turns) for e in episodes),
            "action_tokens": sum(t.action_ids.numel() for e in episodes for t in e.turns),
            "prompt_tokens": sum(t.prompt_ids.numel() for e in episodes for t in e.turns),
            "discounted_reward": sum(sum(gamma ** t * r for t, r in enumerate(e.rewards))
                                      for e in episodes) / n}


def main() -> None:
    from .policy import Actor
    from .rollout import collect

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--output-dir")
    p.add_argument("--model-revision", help="Pin the same Hub commit across every arm")
    p.add_argument("--qwen-judge-revision", help="Recorded frozen-server revision; must match your deployment")
    p.add_argument("--resume", type=Path, help="Trusted local checkpoint; includes optimizer/RNG state")
    args = p.parse_args()
    cfg = ExperimentConfig.load(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.model_revision:
        cfg.model_revision = args.model_revision
    if args.qwen_judge_revision:
        cfg.qwen_judge_revision = args.qwen_judge_revision
    cfg.validate()
    out = Path(cfg.output_dir)
    if out.exists() and any(out.iterdir()):
        p.error("Output directory is nonempty. Resume into a NEW --output-dir to preserve old logs.")
    out.mkdir(parents=True, exist_ok=True)
    train_tasks = load(cfg.train_data, "train")
    validation = load(cfg.eval_data, "validation")
    assert_disjoint(train_tasks, validation)
    prov = provenance([cfg.train_data, cfg.eval_data])
    seed_all(cfg.seed)
    task_rng, shuffle_rng = random.Random(cfg.seed + 1000), random.Random(cfg.seed + 2000)
    started = time.perf_counter()
    actor = Actor(cfg, args.resume)
    # Resolve mutable Hub aliases before writing configs or resuming another process.
    if actor.resolved_revision:
        cfg.model_revision = actor.resolved_revision
    optimizer = actor.optimizer()
    counters = dict(episodes=0, env_steps=0, action_tokens=0, prompt_tokens=0,
                    rollout_seconds=0.0, reward_seconds=0.0, ppo_seconds=0.0, validation_seconds=0.0)
    first, elapsed_before = 0, 0.0
    prior_judge = {}
    if args.resume:
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=False)
        saved_cfg, current_cfg = dict(state["config"]), cfg.to_dict()
        for key in ("output_dir", "updates"):
            saved_cfg.pop(key)
            current_cfg.pop(key)
        if saved_cfg != current_cfg or state["data_sha256"] != prov["data_sha256"]:
            raise ValueError("Resume configuration/data differ; use the saved resolved_config.yaml")
        if cfg.updates <= state["update"]:
            raise ValueError("Resume needs a larger total updates target than the saved update")
        optimizer.load_state_dict(state["optimizer"])
        task_rng.setstate(state["task_rng"])
        shuffle_rng.setstate(state["shuffle_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        counters, first = state["counters"], state["update"]
        elapsed_before, prior_judge = state["elapsed_seconds"], state["judge_totals"]
    import yaml
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    write_json(out / "manifest.json", {"config": cfg.to_dict(), **prov,
            "resolved_actor_revision": actor.resolved_revision,
            "lora_trainable_parameters": sum(p.numel() for p in actor.model.parameters() if p.requires_grad),
            "gpu": torch.cuda.get_device_name(actor.device) if actor.device.type == "cuda" else None,
            "resumed_from": str(args.resume) if args.resume else None,
            "status": "initialized; no results implied"})
    if args.resume and (args.resume.parent / "judge_cache.sqlite3").exists():
        with sqlite3.connect(args.resume.parent / "judge_cache.sqlite3") as source:
            with sqlite3.connect(out / "judge_cache.sqlite3") as destination:
                source.backup(destination)
    judge = make_judge(cfg, out)
    if judge:
        judge.models.update(prior_judge.get("models", []))

    def synchronize():
        if actor.device.type == "cuda":
            torch.cuda.synchronize(actor.device)

    def elapsed():
        return elapsed_before + time.perf_counter() - started

    def judge_totals():
        current = judge.metrics() if judge else {"api_usd": 0.0}
        result = dict(current)
        for key, previous in prior_judge.items():
            if key == "retry_billing_unknown":
                result[key] = bool(previous or current.get(key))
            elif isinstance(previous, (int, float)) and not isinstance(previous, bool):
                result[key] = previous + (current.get(key) or 0)
        return result

    def save(number):
        checkpoint_dir = out / f"checkpoint-{number:06d}"
        actor.save(checkpoint_dir)
        torch.save({"config": cfg.to_dict(), "data_sha256": prov["data_sha256"],
                    "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "task_rng": task_rng.getstate(), "shuffle_rng": shuffle_rng.getstate(),
                    "counters": counters, "update": number, "elapsed_seconds": elapsed(),
                    "judge_totals": judge_totals()}, checkpoint_dir / "training_state.pt")
        write_json(out / "latest.json", {"checkpoint": str(checkpoint_dir), "update": number})

    try:
        for number in range(first, cfg.updates + 1):
            if number == first or number % cfg.eval_every == 0 or number == cfg.updates:
                t0 = time.perf_counter()
                evaluated = collect(actor, validation[:cfg.eval_tasks], cfg, training=False)
                synchronize()
                counters["validation_seconds"] += time.perf_counter() - t0
                append_jsonl(out / "validation.jsonl", {"update": number, "seed": cfg.seed,
                    "reward": cfg.reward, **summarize(evaluated, cfg.gamma), **counters,
                    "elapsed_seconds": elapsed(), "judge": judge_totals()})
                for episode in evaluated:
                    append_jsonl(out / "validation_traces.jsonl", {"update": number, **episode.record()})
            if number == cfg.updates:
                break
            tasks = [task_rng.choice(train_tasks) for _ in range(cfg.rollouts_per_update)]
            t0 = time.perf_counter()
            episodes = collect(actor, tasks, cfg)
            synchronize()
            counters["rollout_seconds"] += time.perf_counter() - t0
            # Preserve rollouts BEFORE API calls. A failed evaluator cannot select away failures.
            for episode in episodes:
                append_jsonl(out / "rollouts.jsonl", {"update": number + 1, **episode.record(include_tokens=True)})
            t0 = time.perf_counter()
            assign_rewards(episodes, cfg, judge)
            counters["reward_seconds"] += time.perf_counter() - t0
            turns = []
            for episode in episodes:
                for turn, reward in zip(episode.turns, episode.rewards):
                    turn.reward = reward - cfg.reference_kl_coef * (turn.old_logprob - turn.ref_logprob)
                add_gae(episode.turns, cfg.gamma, cfg.gae_lambda)
                turns.extend(episode.turns)
                append_jsonl(out / "train_traces.jsonl", {"update": number + 1, **episode.record()})
            t0 = time.perf_counter()
            losses = update(actor, turns, optimizer, cfg, shuffle_rng)
            synchronize()
            counters["ppo_seconds"] += time.perf_counter() - t0
            summary = summarize(episodes, cfg.gamma)
            summary["discounted_objective_return"] = sum(
                sum(cfg.gamma ** t * turn.reward for t, turn in enumerate(e.turns))
                for e in episodes) / len(episodes)
            counters["episodes"] += len(episodes)
            counters["env_steps"] += summary["steps"]
            counters["action_tokens"] += summary["action_tokens"]
            counters["prompt_tokens"] += summary["prompt_tokens"]
            append_jsonl(out / "train.jsonl", {"update": number + 1, "reward": cfg.reward,
                "seed": cfg.seed, **summary, **losses, **counters,
                "elapsed_seconds": elapsed(), "judge": judge_totals()})
            if (number + 1) % cfg.checkpoint_every == 0 or number + 1 == cfg.updates:
                save(number + 1)
    finally:
        # API failures still leave traces, accounting, and the last completed checkpoint.
        if judge:
            judge.close()
        write_json(out / "accounting.json", {**counters, "elapsed_seconds": elapsed(),
                                             "judge": judge_totals()})


if __name__ == "__main__":
    main()
