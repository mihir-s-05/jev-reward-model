"""Judge diagnostics and paired-seed learning curves; no training dependencies."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from .config import digest


def judge_metrics(labels: list[int], predictions: list[float]) -> dict:
    y, p = np.asarray(labels), np.asarray(predictions, dtype=float)
    if len(y) == 0 or y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Need matching, nonempty, finite labels and predictions.")
    positive, negative = p[y == 1], p[y == 0]
    auc = None
    if len(positive) and len(negative):
        differences = positive[:, None] - negative[None, :]
        auc = float(np.mean((differences > 0) + 0.5 * (differences == 0)))
    bins, ece = [], 0.0
    for i in range(10):
        mask = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
        if mask.any():
            confidence, frequency = float(p[mask].mean()), float(y[mask].mean())
            ece += float(mask.mean()) * abs(confidence - frequency)
            bins.append({"lower": i / 10, "upper": (i + 1) / 10, "count": int(mask.sum()),
                         "predicted": confidence, "observed": frequency})
    return {"episodes": len(y), "positive_fraction": float(y.mean()),
            "brier": float(np.mean((p - y) ** 2)), "accuracy_at_half": float(np.mean((p >= 0.5) == y)),
            "auroc": auc, "ece_10_bins": ece, "calibration_bins": bins}


def bootstrap_mean(values: list[float]) -> tuple[float, float | None, float | None]:
    """Resample TRAINING SEEDS, not correlated episodes as independent trained policies."""
    x = np.asarray(values, dtype=float)
    if len(x) == 1:
        return float(x[0]), None, None
    rng = np.random.default_rng(1701)
    means = rng.choice(x, size=(2000, len(x)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(low), float(high)


def report(root: Path, out: Path, gpu_hourly_usd: float | None = None,
           llm_server_hourly_usd: float | None = None, plots: bool = False) -> None:
    """Refuse accidental mixing of different protocols/configs or duplicate arm/seed runs."""
    out.mkdir(parents=True, exist_ok=True)
    runs, seen, configurations = [], set(), set()
    for summary_path in sorted(root.rglob("summary.json")):
        directory = summary_path.parent
        if not (directory / "manifest.json").exists():
            continue  # Audit / standalone checkpoint evaluation is not a training run.
        summary = json.loads(summary_path.read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        key = (manifest["arm"], manifest["seed"])
        if key in seen:
            raise ValueError(f"Duplicate arm/seed {key}; report only one completed continuation.")
        seen.add(key)
        configurations.add(digest([config, manifest["protocol"], manifest["rubric_hash"]]))
        runs.append((directory, summary, manifest))
    if not runs:
        raise ValueError("No completed training runs found.")
    if len(configurations) != 1:
        raise ValueError("Mixed configurations/rubrics/model revisions. Report separate experiments.")
    curves, groups, seed_results = [], defaultdict(list), {}
    for directory, summary, manifest in runs:
        training = {}
        for line in (directory / "metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["phase"] == "training":
                training[row["update"]] = row
            if row["phase"] != "validation":
                continue
            stats = (training.get(row["update"], {}).get("judge") or {})
            prior_known = row.get("prior_api_known_cost_usd", 0)
            known = stats.get("known_cost_usd", 0) + prior_known
            unknown = stats.get("unknown_cost_requests", 0) + row.get("prior_api_unknown_cost_requests", 0)
            arm = manifest["arm"]
            # A self-hosted comparator needs its own cost assumption, not a free-GPU assumption.
            complete = (gpu_hourly_usd is not None and unknown == 0
                        and (not arm.startswith("llm") or llm_server_hourly_usd is not None))
            cost = None
            if complete:
                hours = row["training_seconds"] / 3600
                cost = known + hours * gpu_hourly_usd
                if arm.startswith("llm"):
                    cost += hours * llm_server_hourly_usd
            curves.append({"arm": arm, "seed": manifest["seed"], "update": row["update"],
                           "success": row["success"], "judge_success": row["mean_judge_success"],
                           "judge_brier": row["judge_brier"], "generated_tokens": row["generated_tokens"],
                           "environment_steps": row["environment_steps"],
                           "training_seconds": row["training_seconds"], "estimated_training_cost_usd": cost,
                           "known_judge_cost_usd": known, "unknown_cost_requests": unknown})
        for split, result in summary["final"].items():
            groups[(manifest["arm"], split)].append(result["success"])
            seed_results[(manifest["arm"], split, manifest["seed"])] = result["success"]
    with (out / "learning-curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    aggregate = []
    for (arm, split), values in sorted(groups.items()):
        mean, low, high = bootstrap_mean(values)
        aggregate.append({"arm": arm, "split": split, "training_seeds": len(values),
                          "success_mean": mean, "seed_bootstrap_95_low": low, "seed_bootstrap_95_high": high})
    (out / "final-results.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    paired = []
    for treatment, control in (("jev_terminal", "oracle_terminal"), ("jev_shaping", "oracle_terminal"),
                               ("jev_terminal", "llm_terminal"), ("jev_shaping", "llm_shaping")):
        for split in ("test", "composition", "long"):
            seeds = sorted({seed for arm, sp, seed in seed_results if arm == treatment and sp == split
                            and (control, split, seed) in seed_results})
            if not seeds:
                continue
            differences = [seed_results[treatment, split, seed] - seed_results[control, split, seed]
                           for seed in seeds]
            mean, low, high = bootstrap_mean(differences)
            paired.append({"treatment": treatment, "control": control, "split": split,
                           "paired_seeds": seeds, "success_difference": mean,
                           "seed_bootstrap_95_low": low, "seed_bootstrap_95_high": high})
    (out / "paired-differences.json").write_text(json.dumps(paired, indent=2) + "\n")
    if plots:
        import matplotlib.pyplot as plt
        for x, label in (("update", "PPO updates"), ("generated_tokens", "Training generated tokens"),
                         ("training_seconds", "Training wall seconds (including judge waits)"),
                         ("estimated_training_cost_usd", "Estimated training cost (USD)")):
            figure, axes = plt.subplots()
            plotted = False
            for arm in sorted({r["arm"] for r in curves}):
                rows = [r for r in curves if r["arm"] == arm and r[x] is not None]
                by_update = defaultdict(list)
                for row in rows:
                    by_update[row["update"]].append(row)
                points = sorted(by_update)
                if not points:
                    continue
                xx = [np.mean([r[x] for r in by_update[u]]) for u in points]
                yy = [np.mean([r["success"] for r in by_update[u]]) for u in points]
                axes.plot(xx, yy, label=arm, marker=".")
                plotted = True
            axes.set(xlabel=label, ylabel="Greedy validation task success", ylim=(-0.02, 1.02))
            if plotted:
                axes.legend()
                figure.tight_layout()
                figure.savefig(out / f"success-vs-{x}.png", dpi=160)
            plt.close(figure)
        # Per-arm surrogate-versus-ground-truth values; not a cross-provider reward ranking.
        figure, axes = plt.subplots()
        for arm in sorted({r["arm"] for r in curves}):
            rows = [r for r in curves if r["arm"] == arm and r["judge_success"] is not None]
            if rows:
                axes.scatter([r["judge_success"] for r in rows], [r["success"] for r in rows], label=arm)
        axes.set(xlabel="Mean evaluator success probability", ylabel="Independent task success")
        if axes.collections:
            axes.legend()
            figure.tight_layout()
            figure.savefig(out / "surrogate-vs-success.png", dpi=160)
        plt.close(figure)
