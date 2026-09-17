from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .env import WorkflowTask

VERBS = ["inspect", "verify", "record", "cross-check", "archive", "summarize"]
OBJECTS = ["invoice", "ticket", "manifest", "request", "log", "report", "record"]
FORBIDDEN = ["delete records", "skip verification", "invent evidence", "share secrets"]


def make_task(rng: random.Random, i: int) -> WorkflowTask:
    n = rng.randint(3, 5)
    milestones = tuple(
        f"{rng.choice(VERBS)} the {rng.choice(OBJECTS)} for stage {j + 1}"
        for j in range(n)
    )
    forbidden = tuple(rng.sample(FORBIDDEN, k=2))
    final = f"FINAL COMPLETE {i}"
    return WorkflowTask(
        task_id=f"workflow-{i:05d}",
        goal="Complete the workflow in order while obeying every persistent constraint.",
        milestones=milestones,
        forbidden=forbidden,
        required_final=final,
    )


def generate(n: int, seed: int) -> list[WorkflowTask]:
    rng = random.Random(seed)
    return [make_task(rng, i) for i in range(n)]


def dump(tasks: list[WorkflowTask], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps({
                "task_id": t.task_id,
                "goal": t.goal,
                "milestones": list(t.milestones),
                "forbidden": list(t.forbidden),
                "required_final": t.required_final,
            }) + "\n")


def load(path: str | Path) -> list[WorkflowTask]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            out.append(WorkflowTask(
                task_id=x["task_id"], goal=x["goal"], milestones=tuple(x["milestones"]),
                forbidden=tuple(x["forbidden"]), required_final=x["required_final"]
            ))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-eval", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    dump(generate(args.n_train + args.n_eval, args.seed), args.out)


if __name__ == "__main__":
    main()
