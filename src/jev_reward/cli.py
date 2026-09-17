"""Small CLI; CPU demos/audits do not import or download the actor model."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

from .config import ARMS, PRIMARY_ARMS, canonical, load_config
from .env import SPLITS, FulfillmentEnv, audit_trace, expert_actions, make_task


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Jev reward comparisons on simulated fulfillment")
    commands = cli.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Print a scripted CPU-only episode; no API or model")
    demo.add_argument("--config", default="configs/default.yaml")
    demo.add_argument("--fault", choices=["address", "incomplete", "protected", "claim"])
    demo.add_argument("--split", choices=SPLITS, default="validation")
    demo.add_argument("--index", type=int, default=0)
    doctor = commands.add_parser("doctor", help="Check configuration and installed training imports, no weights/API")
    doctor.add_argument("--config", default="configs/default.yaml")
    for name in ("train", "sweep"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default="configs/default.yaml")
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--eval-judge", choices=["same", "none", "jev", "llm"], default="same")
        if name == "train":
            sub.add_argument("--arm", choices=ARMS, required=True)
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--resume", type=Path)
        else:
            sub.add_argument("--arms", choices=ARMS, nargs="+", default=list(PRIMARY_ARMS))
            sub.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    evaluation = commands.add_parser("evaluate", help="Evaluate a saved adapter using an independent oracle")
    evaluation.add_argument("--checkpoint", required=True, type=Path)
    evaluation.add_argument("--out", required=True, type=Path)
    evaluation.add_argument("--split", choices=SPLITS[1:], default="test")
    evaluation.add_argument("--episodes", type=int, default=100)
    evaluation.add_argument("--judge", choices=["none", "jev", "llm"], default="none")
    for name in ("audit", "score-traces"):
        sub = commands.add_parser(name, help="Paid judge evaluation only; no actor training")
        sub.add_argument("--config", default="configs/default.yaml")
        sub.add_argument("--provider", choices=["jev", "llm"], required=True)
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--cases", type=int, default=30)
        if name == "score-traces":
            sub.add_argument("--input", type=Path, required=True)
    analysis = commands.add_parser("report")
    analysis.add_argument("--runs", type=Path, required=True)
    analysis.add_argument("--out", type=Path, required=True)
    analysis.add_argument("--plots", action="store_true")
    analysis.add_argument("--gpu-hourly-usd", type=float)
    analysis.add_argument("--llm-server-hourly-usd", type=float)
    return cli


async def run_audit(args: argparse.Namespace) -> None:
    from .analysis import judge_metrics
    from .judges import Judge
    if args.cases < 1:
        raise ValueError("cases must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Audit output directory must be empty.")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    records = []
    if args.command == "audit":
        # Five controlled variants of each validation-only task; never use test to tune rubrics.
        for i in range(args.cases):
            task = make_task("validation", i, 20260919, cfg.task)
            for fault in (None, "incomplete", "address", "protected", "claim"):
                trajectory, oracle = audit_trace(task, fault)
                records.append({"episode_id": task.id, "variant": fault or "expert",
                                "trajectory": trajectory, "oracle": oracle})
    else:
        with args.input.open() as stream:
            for line in stream:
                if len(records) >= args.cases:
                    break
                row = json.loads(line)
                records.append({k: row[k] for k in ("episode_id", "trajectory", "oracle")})
    if not records:
        raise ValueError("No trajectories to score.")
    judge = Judge(args.provider, cfg.judge, args.out / "judge")
    try:
        judgments = await asyncio.gather(*(judge.evaluate(r["trajectory"]) for r in records))
        with (args.out / "judgments.jsonl").open("w") as stream:
            for record, judgment in zip(records, judgments):
                stream.write(canonical({**record, "judgment": asdict(judgment)}) + "\n")
        stats = judge_metrics([r["oracle"]["success"] for r in records], [j.success for j in judgments])
        (args.out / "summary.json").write_text(json.dumps({"provider": args.provider, **stats,
             "judge": judge.summary(), "note": "Calibration describes only this sampled trajectory mixture."}, indent=2) + "\n")
    finally:
        await judge.close()


def run_sweep(args: argparse.Namespace) -> None:
    from transformers import AutoConfig
    cfg = load_config(args.config)
    if len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate sweep arms or seeds.")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Sweep output directory must be empty.")
    args.out.mkdir(parents=True, exist_ok=True)
    # Resolve the actor ONCE so different arms cannot accidentally load different 'main's.
    resolved = AutoConfig.from_pretrained(cfg.model.name, revision=cfg.model.revision, trust_remote_code=False)
    revision = getattr(resolved, "_commit_hash", None)
    if not revision:
        raise RuntimeError("Could not resolve an immutable Hub revision for the sweep.")
    cfg.model.revision = revision
    config_path = args.out.resolve() / "resolved-config.json"
    config_path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    # Rotate provider order by seed to reduce (not eliminate) API time-of-day confounding.
    jobs = [(seed, arm) for i, seed in enumerate(args.seeds)
            for arm in (args.arms[i % len(args.arms):] + args.arms[:i % len(args.arms)])]
    (args.out / "plan.json").write_text(json.dumps({"jobs": jobs, "actor_revision": revision}, indent=2) + "\n")
    for seed, arm in jobs:
        subprocess.run([sys.executable, "-m", "jev_reward", "train", "--config", str(config_path),
                        "--arm", arm, "--seed", str(seed), "--out", str(args.out / f"{arm}-seed{seed}"),
                        "--eval-judge", args.eval_judge], check=True)


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "episodes", 1) < 1:
        raise ValueError("episodes must be positive")
    if args.command == "demo":
        cfg = load_config(args.config)
        task = make_task(args.split, args.index, 7, cfg.task)
        env = FulfillmentEnv(task)
        print(json.dumps(env.snapshot(), indent=2))
        for action in expert_actions(task, args.fault):
            if env.closed:
                break
            print(action, "->", canonical(env.step(action)))
        print("PRIVATE VERIFIER (never sent to a model):", canonical(env.verify()))
    elif args.command == "doctor":
        cfg = load_config(args.config)
        import torch
        from transformers import Qwen3_5ForConditionalGeneration
        from peft import get_peft_model
        print(canonical({"config_valid": True, "cuda": torch.cuda.is_available(),
                         "qwen_class": Qwen3_5ForConditionalGeneration.__name__,
                         "peft": get_peft_model.__name__, "requested_device": cfg.model.device}))
    elif args.command == "train":
        from .experiment import train
        asyncio.run(train(load_config(args.config), args.arm, args.seed, args.out, args.resume,
                          None if args.eval_judge == "none" else args.eval_judge))
    elif args.command == "sweep":
        run_sweep(args)
    elif args.command == "evaluate":
        from .experiment import evaluate_checkpoint
        asyncio.run(evaluate_checkpoint(args.checkpoint, args.out, args.split, args.episodes,
                                        None if args.judge == "none" else args.judge))
    elif args.command in ("audit", "score-traces"):
        asyncio.run(run_audit(args))
    else:
        from .analysis import report
        for value in (args.gpu_hourly_usd, args.llm_server_hourly_usd):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Hourly rates must be nonnegative finite numbers.")
        report(args.runs, args.out, args.gpu_hourly_usd, args.llm_server_hourly_usd, args.plots)


if __name__ == "__main__":
    main()
