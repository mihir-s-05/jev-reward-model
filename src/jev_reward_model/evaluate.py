"""Independent deterministic evaluation: no judge is consulted and no weights change."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .config import ExperimentConfig
from .data import assert_disjoint, load
from .train import seed_all, summarize
from .utils import append_jsonl, provenance, write_json


def main() -> None:
    from .policy import Actor
    from .rollout import collect

    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--config", help="Evaluate the untrained base actor")
    p.add_argument("--model-revision", help="Pin the base model for an untrained baseline")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--tasks", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="0 means all tasks")
    args = p.parse_args()
    if args.limit < 0 or (args.checkpoint and not args.run_dir):
        p.error("Invalid limit or checkpoint without --run-dir")
    cfg = ExperimentConfig.load(args.run_dir / "resolved_config.yaml" if args.run_dir else args.config)
    if args.model_revision:
        if args.run_dir:
            p.error("Saved runs already pin their model; do not override their revision")
        cfg.model_revision = args.model_revision
    checkpoint_path = args.checkpoint
    if args.run_dir and checkpoint_path is None:
        checkpoint_path = Path(json.loads((args.run_dir / "latest.json").read_text())["checkpoint"])
    tasks = load(args.tasks)
    if any(t.split == "train" for t in tasks):
        p.error("Independent evaluation cannot use the training split")
    assert_disjoint(load(cfg.train_data, "train"), tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_all(cfg.seed)
    actor = Actor(cfg, checkpoint_path)
    groups = defaultdict(list)
    all_episodes = []
    for start in range(0, len(tasks), cfg.rollouts_per_update):
        episodes = collect(actor, tasks[start:start + cfg.rollouts_per_update], cfg, training=False)
        for episode in episodes:
            append_jsonl(args.output_dir / "traces.jsonl", episode.record())
            groups[f"family:{episode.task.family}"].append(episode)
            groups[f"operations:{len(episode.task.milestones)}"].append(episode)
        all_episodes.extend(episodes)
    write_json(args.output_dir / "summary.json", {
        "count": len(all_episodes), "checkpoint": str(checkpoint_path), "config": cfg.to_dict(),
        "resolved_actor_revision": actor.resolved_revision, **provenance([args.tasks]),
        "overall": summarize(all_episodes, cfg.gamma),
        "groups": {k: {"n": len(v), **summarize(v, cfg.gamma)} for k, v in groups.items()}})


if __name__ == "__main__":
    main()
