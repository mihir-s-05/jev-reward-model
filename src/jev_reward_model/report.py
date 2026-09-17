"""CSV and learning curves at common budgets; no fabricated prices or success results."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .config import ExperimentConfig
from .utils import write_json


def cost(row: dict, cfg: ExperimentConfig) -> float | None:
    # Reserved-device accounting includes collection, evaluation, API waits and startup.
    # A local judge is not 'free' just because it has no token invoice.
    if cfg.actor_gpu_usd_per_hour is None:
        return None
    total = cfg.actor_gpu_usd_per_hour * row["elapsed_seconds"] / 3600
    if cfg.reward.startswith("qwen_"):
        if cfg.qwen_judge_gpu_usd_per_hour is None:
            return None
        total += cfg.qwen_judge_gpu_usd_per_hour * row["elapsed_seconds"] / 3600
    else:
        total += row["judge"].get("api_usd") or 0.0
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("reports/comparison"))
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    runs = []
    seen = set()
    table, alignment = [], []
    comparison_signature = None
    for directory in args.runs:
        cfg = ExperimentConfig.load(directory / "resolved_config.yaml")
        manifest = json.loads((directory / "manifest.json").read_text())
        signature = {k: v for k, v in cfg.to_dict().items() if k not in
                     {"reward", "seed", "output_dir", "updates", "actor_gpu_usd_per_hour",
                      "qwen_judge_gpu_usd_per_hour"}}
        signature["data_sha256"] = manifest["data_sha256"]
        if comparison_signature is not None and signature != comparison_signature:
            raise ValueError("Nuisance configs/data differ between runs; compare matched arms only")
        comparison_signature = signature
        key = (cfg.reward, cfg.seed)
        if key in seen:
            raise ValueError("Provide one run segment per arm/seed; do not double-count resumed seeds")
        seen.add(key)
        # On an exact-checkpoint resume there can be two identical validation update IDs.
        by_update = {r["update"]: r for r in map(json.loads, (directory / "validation.jsonl").read_text().splitlines())}
        rows = [by_update[k] for k in sorted(by_update)]
        if not rows:
            raise ValueError(f"No validation records in {directory}")
        for row in rows:
            row["total_usd_estimate"] = cost(row, cfg)
            table.append({"run": str(directory), "arm": cfg.reward, "seed": cfg.seed,
                          **{k: row[k] for k in ("update", "success", "violation", "env_steps",
                             "action_tokens", "elapsed_seconds", "total_usd_estimate")}})
        runs.append((cfg.reward, cfg.seed, rows))
        trace_path = directory / "train_traces.jsonl"
        if trace_path.exists():
            grouped = defaultdict(list)
            with trace_path.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    grouped[record["update"]].append(record)
            for number, traces in sorted(grouped.items()):
                judged = [t for t in traces if t["judge_success"] is not None]
                alignment.append({"arm": cfg.reward, "seed": cfg.seed, "update": number,
                    "success": sum(t["oracle"]["success"] for t in traces) / len(traces),
                    "mean_terminal_judge": sum(t["judge_success"] for t in judged) / len(judged) if judged else None,
                    "episodes": len(traces)})
    with (args.out_dir / "checkpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    if alignment:
        with (args.out_dir / "reward_alignment.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(alignment[0]))
            writer.writeheader()
            writer.writerows(alignment)
    curves = []
    for axis in ("env_steps", "action_tokens", "elapsed_seconds", "total_usd_estimate"):
        if any(any(r[axis] is None for r in rows) for _, _, rows in runs):
            continue  # Unknown costs stay unknown rather than making local inference free.
        lower = max(rows[0][axis] for _, _, rows in runs)
        upper = min(rows[-1][axis] for _, _, rows in runs)
        if upper < lower:
            continue
        budgets = sorted({r[axis] for _, _, rows in runs for r in rows if lower <= r[axis] <= upper})
        by_arm = defaultdict(list)
        for budget in budgets:
            scores = defaultdict(list)
            for arm, _, rows in runs:
                index = bisect.bisect_right([r[axis] for r in rows], budget) - 1
                scores[arm].append(rows[index]["success"])
            for arm, values in scores.items():
                mean = sum(values) / len(values)
                se = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1) / len(values)) if len(values) > 1 else None
                record = {"axis": axis, "budget": budget, "arm": arm, "mean_success": mean,
                          "seed_se": se, "seeds": len(values)}
                curves.append(record)
                by_arm[arm].append(record)
        if not args.no_plots:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for arm, points in sorted(by_arm.items()):
                x, y = [r["budget"] for r in points], [r["mean_success"] for r in points]
                ax.step(x, y, where="post", label=f"{arm} (n={points[0]['seeds']})")
            ax.set(xlabel=axis, ylabel="Held-out exact success", ylim=(0, 1),
                   title="Latest completed validation at or below each budget")
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.out_dir / f"success_vs_{axis}.png", dpi=160)
            plt.close(fig)
    write_json(args.out_dir / "common_budget_curves.json", curves)
    write_json(args.out_dir / "notes.json", {
        "aggregation": "Step-held validation, not interpolation from future checkpoints. SE is across seeds.",
        "prices": "User-supplied device rates and configured API prices; retries may have unreported billing.",
        "limits": "No claim of statistical significance. Match configs/data/model revisions before interpreting."})


if __name__ == "__main__":
    main()
