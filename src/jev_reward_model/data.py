"""Generate immutable, explicitly separated task splits and load them strictly."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from .env import WorkflowTask
from .utils import canonical, digest, write_json

TRAIN_FAMILIES = ("chain", "fork_join", "two_chains")
HELDOUT_FAMILIES = ("barrier", "overlapping_dependencies")


def make_task(rng: random.Random, split: str, i: int, family: str, n: int) -> WorkflowTask:
    nonce = f"{rng.getrandbits(48):012x}"
    domain = rng.choice(("invoice", "shipment", "report", "access_request"))
    verbs = ("inspect", "verify", "approve", "register", "reconcile", "archive")
    # Opaque random operation IDs avoid leaking a topological order through numeric suffixes.
    labels = rng.sample(range(1 << 24), n)
    names = tuple(f"{rng.choice(verbs)}_{domain}_{nonce}_{label:06x}" for label in labels)
    parents = []
    for j in range(n):
        if family == "chain":
            ids = [j - 1] if j else []
        elif family == "fork_join":
            ids = [] if j == 0 else list(range(1, n - 1)) if j == n - 1 else [0]
        elif family == "two_chains":
            ids = [n - 3, n - 2] if j == n - 1 else [j - 2] if j >= 2 else []
        elif family == "barrier":
            ids = list(range(n // 2)) if j >= n // 2 else []
        elif family == "overlapping_dependencies":
            ids = list(range(max(0, j - 2), j))
        else:
            raise ValueError(family)
        parents.append(tuple(names[k] for k in ids))
    order = list(range(n))
    rng.shuffle(order)  # Public catalog order is not a solution trace.
    return WorkflowTask(f"{split}-{i:05d}-{nonce}", split, family,
                        f"Complete the {domain} workflow while preserving all constraints.",
                        names, tuple(parents),
                        (f"DO bypass_review_{nonce}", f"DO delete_audit_log_{nonce}"),
                        f"FINISH {nonce}", n + 5, tuple(order))


def generate(n: int, seed: int, split: str, families: tuple[str, ...], sizes: tuple[int, ...]) -> list[WorkflowTask]:
    rng = random.Random(seed)
    return [make_task(rng, split, i, families[i % len(families)], sizes[(i // len(families)) % len(sizes)])
            for i in range(n)]


def dump(tasks: list[WorkflowTask], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(asdict(t)) + "\n" for t in tasks))


def load(path: str | Path, expected_split: str | None = None) -> list[WorkflowTask]:
    tasks = []
    for line in Path(path).read_text().splitlines():
        x = json.loads(line)
        for key in ("milestones", "forbidden", "display_order"):
            x[key] = tuple(x[key])
        x["prerequisites"] = tuple(tuple(p) for p in x["prerequisites"])
        tasks.append(WorkflowTask(**x))
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("Task file must be nonempty with unique IDs")
    if expected_split and any(t.split != expected_split for t in tasks):
        raise ValueError(f"{path} contains a task outside {expected_split}")
    return tasks


def assert_disjoint(train: list[WorkflowTask], evaluation: list[WorkflowTask]) -> None:
    if {t.task_id for t in train} & {t.task_id for t in evaluation}:
        raise ValueError("Train/evaluation ID overlap")
    if {digest(t.public_spec()) for t in train} & {digest(t.public_spec()) for t in evaluation}:
        raise ValueError("Train/evaluation content overlap")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-eval", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    if min(args.n_train, args.n_eval) < 1:
        p.error("Counts must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        p.error("Use an empty directory; datasets are not silently overwritten")
    splits = {"train": (args.n_train, TRAIN_FAMILIES, (4, 8, 12)),
              "validation": (args.n_eval, TRAIN_FAMILIES, (4, 8, 12)),
              "test_id": (args.n_eval, TRAIN_FAMILIES, (4, 8, 12)),
              "test_ood": (args.n_eval, HELDOUT_FAMILIES, (4, 8, 12)),
              "test_long": (args.n_eval, TRAIN_FAMILIES, (20, 28))}
    manifest = {}
    for j, (name, (count, families, sizes)) in enumerate(splits.items()):
        tasks = generate(count, args.seed + j * 100003, name, families, sizes)
        dump(tasks, args.out_dir / f"{name}.jsonl")
        manifest[name] = {"count": count, "families": families, "sizes": sizes,
                          "content_hash": digest([asdict(t) for t in tasks])}
    write_json(args.out_dir / "manifest.json", {"seed": args.seed, "splits": manifest})


if __name__ == "__main__":
    main()
