from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

from .data import load
from .env import WorkflowEnv
from .rollout import generate_action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data", default="data/tasks.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--eval-n", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=8)
    args = p.parse_args()
    ckpt = Path(args.checkpoint) if args.checkpoint else sorted(args.run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))[-1]
    processor = AutoProcessor.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model = PeftModel.from_pretrained(base, ckpt)
    tasks = load(args.data)[-args.eval_n:]
    rows = []
    for task in tasks:
        env = WorkflowEnv(task, args.max_steps)
        while not env.state.done:
            action, _, _ = generate_action(model, processor, task, env.state.actions, 64)
            env.step(action)
        rows.append({"task": task.task_id, "success": env.state.success, "progress": env.state.progress, "violations": len(env.state.violations), "actions": env.state.actions})
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "eval.jsonl").open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    summary = {"n": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])), "violation_rate": float(np.mean([r["violations"] > 0 for r in rows])), "mean_progress": float(np.mean([r["progress"] for r in rows]))}
    (args.run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
