"""Rollouts, independent evaluation, checkpointing, and the complete comparison loop."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch

from .config import ARMS, Config, canonical, digest
from .env import FulfillmentEnv, Task, make_task
from .judges import Judge, QUESTIONS, RUBRIC_VERSION
from .policy import ContextBudgetError, Policy, Turn
from .ppo import prepare_episode, update_policy
from .rewards import score_episode

EVAL_SEED = 20260917  # Fixed common held-out cases, independent of actor training seed.


@dataclass
class Episode:
    task: Task
    turns: list[Turn]
    snapshots: list[dict]
    oracle: dict


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(canonical(record) + "\n")


def write_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")


def runtime_metadata() -> dict:
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "httpx", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        revision, dirty = None, None
    return {"python": platform.python_version(), "packages": versions, "code_revision": revision,
            "code_dirty": dirty, "cuda_runtime": torch.version.cuda}


def collect_episode(policy: Policy, task: Task, greedy: bool = False) -> Episode:
    env = FulfillmentEnv(task)
    snapshots, turns = [env.snapshot()], []
    while not env.closed:
        try:
            turn = policy.sample(snapshots[-1], greedy=greedy)
        except ContextBudgetError:
            if not turns:
                raise  # A bad configuration must not silently remove hard tasks.
            env.close("context_budget")
            snapshots[-1] = env.snapshot()
            break
        # Token-budget truncation is not EOS: retain all tokens and let JSON validation
        # reject incomplete calls. A mechanically complete JSON call may still execute.
        env.step(turn.text)
        turns.append(turn)
        snapshots.append(env.snapshot())
    return Episode(task, turns, snapshots, env.verify())


def trace_record(episode: Episode, update: int, phase: str, reward: dict | None = None) -> dict:
    return {"episode_id": episode.task.id, "split": episode.task.split, "family": episode.task.family,
            "update": update, "phase": phase, "trajectory": episode.snapshots[-1],
            "oracle": episode.oracle, "reward": reward,
            "sampled_actions": [{"tokens": t.response_ids, "ended_with_eos": t.ended_with_eos,
                                 "old_logprobs": t.old_logprobs} for t in episode.turns]}


async def evaluate(policy: Policy, cfg: Config, split: str, count: int, directory: Path,
                   update: int, judge: Judge | None) -> dict:
    started = time.perf_counter()
    episodes = [collect_episode(policy, make_task(split, i, EVAL_SEED, cfg.task), greedy=True)
                for i in range(count)]
    results = (await asyncio.gather(*(judge.evaluate(e.snapshots[-1]) for e in episodes))
               if judge is not None else [None] * count)
    for episode, result in zip(episodes, results):
        append_jsonl(directory / "eval-trajectories.jsonl",
                     trace_record(episode, update, "evaluation", asdict(result) if result else None))
    success = [e.oracle["success"] for e in episodes]
    predictions = [r.success for r in results if r is not None]
    return {"split": split, "episodes": count, "success": float(np.mean(success)),
            "mean_steps": float(np.mean([e.oracle["steps"] for e in episodes])),
            "invalid_action_rate": sum(e.oracle["invalid_actions"] for e in episodes)
                                   / max(1, sum(e.oracle["steps"] for e in episodes)),
            "mean_judge_success": float(np.mean(predictions)) if predictions else None,
            "judge_brier": float(np.mean((np.array(predictions) - success) ** 2)) if predictions else None,
            "seconds": time.perf_counter() - started}


def checkpoint_run(policy: Policy, optimizer: torch.optim.Optimizer, directory: Path,
                   cfg: Config, arm: str, seed: int, update: int, rng: random.Random,
                   counters: dict, judge: Judge | None) -> Path:
    parent = directory / "checkpoints"
    parent.mkdir(exist_ok=True)
    final = parent / f"update-{update:05d}"
    temporary = parent / f".update-{update:05d}.tmp"
    temporary.mkdir(exist_ok=False)
    policy.save_adapter(temporary)
    # Local research checkpoints only. All state objects are tensors or primitive containers.
    torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "sample_rng": policy.generator.get_state(), "shuffle_rng": rng.getstate()},
               temporary / "trainer.pt")
    write_json(temporary / "metadata.json", {"config": cfg.to_dict(), "config_hash": digest(cfg.to_dict()),
               "arm": arm, "seed": seed, "update": update, "counters": counters,
               "judge_model": judge.resolved_model if judge else None, "model": policy.metadata()})
    temporary.rename(final)
    return final


async def train(cfg: Config, arm: str, seed: int, directory: Path,
                resume: Path | None = None, evaluation_judge: str | None = "same") -> None:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm}")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Output directory must be empty; resume writes a NEW continuation directory.")
    directory.mkdir(parents=True, exist_ok=True)
    provider = "jev" if arm.startswith("jev") else "llm" if arm.startswith("llm") else None
    if evaluation_judge == "same":
        evaluation_judge = provider
    resume_meta = json.loads((resume / "metadata.json").read_text()) if resume else None
    if resume_meta:
        if (resume_meta["arm"], resume_meta["seed"]) != (arm, seed):
            raise ValueError("Resume arm/seed mismatch.")
        cfg.model.revision = resume_meta["model"]["revision"]
        if digest(cfg.to_dict()) != resume_meta["config_hash"]:
            raise ValueError("Resume requires the identical configuration (resolved model revision excepted).")
    # Check credentials before allocating a model or making any paid requests.
    training_judge = Judge(provider, cfg.judge, directory / "training-judge") if provider else None
    eval_judge = Judge(evaluation_judge, cfg.judge, directory / "evaluation-judge") if evaluation_judge else None
    try:
        empty = {"goal": {}, "events": [], "closed": False}
        for current in (training_judge, eval_judge):
            if current:
                current.request(empty)  # Local request construction only, no network call.
        policy = Policy(cfg.model, seed)
        cfg.model.revision = policy.revision
        optimizer = policy.optimizer(cfg.ppo.learning_rate, cfg.ppo.critic_learning_rate)
        rng = random.Random(seed + 31337)
        start_update = 0
        counters = {"training_episodes": 0, "environment_steps": 0, "generated_tokens": 0,
                    "training_seconds": 0.0, "evaluation_seconds": 0.0,
                    "prior_api_known_cost_usd": 0.0, "prior_api_unknown_cost_requests": 0}
        if resume_meta:
            policy.load_adapter(resume)
            state = torch.load(resume / "trainer.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"])
            if state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            policy.generator.set_state(state["sample_rng"])
            rng.setstate(state["shuffle_rng"])
            start_update, counters = resume_meta["update"], resume_meta["counters"]
            if training_judge:
                training_judge.resolved_model = resume_meta["judge_model"]
        write_json(directory / "config.json", cfg.to_dict())
        write_json(directory / "manifest.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                   "arm": arm, "seed": seed, "resume_from": str(resume) if resume else None,
                   "evaluation_judge": evaluation_judge, "model": policy.metadata(),
                   "runtime": runtime_metadata(), "rubric_version": RUBRIC_VERSION,
                   "rubric_hash": digest(QUESTIONS), "protocol": "fulfillment-token-ppo-v1"})
        initial = await evaluate(policy, cfg, "validation", cfg.ppo.eval_episodes, directory, start_update, eval_judge)
        counters["evaluation_seconds"] += initial["seconds"]
        append_jsonl(directory / "metrics.jsonl", {"phase": "validation", "update": start_update,
                     "arm": arm, "seed": seed, **counters, **initial})
        for update in range(start_update + 1, cfg.ppo.updates + 1):
            started = time.perf_counter()
            episodes = [collect_episode(policy, make_task("train", (update - 1) * cfg.ppo.episodes_per_update + i,
                                                        seed, cfg.task))
                        for i in range(cfg.ppo.episodes_per_update)]
            rollout_seconds = time.perf_counter() - started
            # Fresh policy is unchanged throughout this complete on-policy rollout batch.
            consistency = policy.check_rollout_consistency(episodes[0].turns[0])
            reference_started = time.perf_counter()
            for episode in episodes:
                for turn in episode.turns:
                    turn.ref_logprobs = (policy.reference_statistics(turn) if cfg.ppo.kl_coefficient
                                         else list(turn.old_logprobs))
            reference_seconds = time.perf_counter() - reference_started
            judge_started = time.perf_counter()
            scored = await asyncio.gather(*(score_episode(arm, e.snapshots, e.oracle["success"], training_judge,
                                                         cfg.ppo.gamma, cfg.ppo.shaping_alpha) for e in episodes))
            judge_seconds = time.perf_counter() - judge_started
            prepared = []
            for episode, (rewards, details) in zip(episodes, scored):
                prepared.append(prepare_episode(episode.turns, rewards, cfg.ppo))
                append_jsonl(directory / "train-trajectories.jsonl", trace_record(episode, update, "training", details))
            optimization_started = time.perf_counter()
            optimization = update_policy(policy, optimizer, [t for e in episodes for t in e.turns], cfg.ppo, rng)
            optimization_seconds = time.perf_counter() - optimization_started
            counters["training_episodes"] += len(episodes)
            counters["environment_steps"] += sum(e.oracle["steps"] for e in episodes)
            counters["generated_tokens"] += sum(p["generated_tokens"] for p in prepared)
            counters["training_seconds"] += time.perf_counter() - started
            judge_stats = training_judge.summary() if training_judge else None
            record = {"phase": "training", "arm": arm, "seed": seed, "update": update, **counters,
                      "success": float(np.mean([e.oracle["success"] for e in episodes])),
                      "mean_terminal_reward": float(np.mean([s[1]["terminal"] for s in scored])),
                      "reference_kl_sample_mean": float(np.mean([p["reference_kl_sample_mean"] for p in prepared])),
                      "rollout_logprob_max_error": consistency, "rollout_seconds": rollout_seconds,
                      "reference_seconds": reference_seconds, "judge_seconds": judge_seconds,
                      "optimization_seconds": optimization_seconds, "ppo": optimization, "judge": judge_stats}
            append_jsonl(directory / "metrics.jsonl", record)
            print(canonical({k: record[k] for k in ("arm", "seed", "update", "success", "training_seconds")}), flush=True)
            if update % cfg.ppo.eval_every == 0 or update == cfg.ppo.updates:
                measured = await evaluate(policy, cfg, "validation", cfg.ppo.eval_episodes, directory, update, eval_judge)
                counters["evaluation_seconds"] += measured["seconds"]
                append_jsonl(directory / "metrics.jsonl", {"phase": "validation", "arm": arm, "seed": seed,
                             "update": update, **counters, **measured})
            if update % cfg.ppo.checkpoint_every == 0 or update == cfg.ppo.updates:
                saved_counters = dict(counters)
                if training_judge:
                    saved_counters["prior_api_known_cost_usd"] += training_judge.stats["known_cost_usd"]
                    saved_counters["prior_api_unknown_cost_requests"] += training_judge.stats["unknown_cost_requests"]
                checkpoint_run(policy, optimizer, directory, cfg, arm, seed, update, rng,
                               saved_counters, training_judge)
        final = {}
        for split in ("test", "composition", "long"):
            final[split] = await evaluate(policy, cfg, split, cfg.ppo.eval_episodes, directory,
                                         cfg.ppo.updates, eval_judge)
            counters["evaluation_seconds"] += final[split]["seconds"]
        write_json(directory / "summary.json", {"arm": arm, "seed": seed, **counters, "final": final,
                   "training_judge": training_judge.summary() if training_judge else None,
                   "evaluation_judge": eval_judge.summary() if eval_judge else None})
    except BaseException as exc:
        write_json(directory / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                   "training_judge": training_judge.summary() if training_judge else None})
        raise
    finally:
        for current in (training_judge, eval_judge):
            if current:
                await current.close()


async def evaluate_checkpoint(checkpoint: Path, directory: Path, split: str, episodes: int,
                              provider: str | None) -> None:
    from .config import load_config
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Evaluation output directory must be empty.")
    directory.mkdir(parents=True, exist_ok=True)
    meta = json.loads((checkpoint / "metadata.json").read_text())
    # JSON is valid YAML; use the same strict configuration parser.
    config_path = directory / "config.json"
    write_json(config_path, meta["config"])
    cfg = load_config(config_path)
    policy = Policy(cfg.model, meta["seed"])
    policy.load_adapter(checkpoint)
    judge = Judge(provider, cfg.judge, directory / "judge") if provider else None
    try:
        result = await evaluate(policy, cfg, split, episodes, directory, meta["update"], judge)
        write_json(directory / "summary.json", {"checkpoint": str(checkpoint), **result,
                                               "judge": judge.summary() if judge else None})
    finally:
        if judge:
            await judge.close()
