"""Reward arms and action-boundary potential shaping, separate from policy optimization."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import TYPE_CHECKING
import numpy as np

from .config import ARMS
if TYPE_CHECKING:
    from .judges import Judge


def potential_shaping(potentials: list[float], gamma: float, alpha: float) -> np.ndarray:
    """Exactly T+1 cached boundary values; final potential MUST be zero at closure."""
    p = np.asarray(potentials, dtype=np.float64)
    if len(p) < 2 or not np.isfinite(p).all() or p[-1] != 0:
        raise ValueError("Need finite T+1 potentials and zero terminal potential.")
    return alpha * (gamma * p[1:] - p[:-1])


async def score_episode(arm: str, snapshots: list[dict], oracle_success: int,
                        judge: Judge | None, gamma: float, alpha: float) -> tuple[list[float], dict]:
    """Snapshots are s_0 ... s_T; oracle is used ONLY by oracle and grounded-shaping arms."""
    if arm not in ARMS or len(snapshots) < 2 or not snapshots[-1]["closed"]:
        raise ValueError("Unknown arm or incomplete/empty episode.")
    rewards = np.zeros(len(snapshots) - 1, dtype=np.float64)
    if arm == "oracle_terminal":
        rewards[-1] = float(oracle_success)
        return rewards.tolist(), {"source": "oracle", "terminal": float(oracle_success)}
    if judge is None:
        raise ValueError("A learned-reward arm requires a judge.")
    if arm.endswith("terminal"):
        result = await judge.evaluate(snapshots[-1])
        rewards[-1] = result.success  # Joint probability, never product of marginals.
        return rewards.tolist(), {"source": arm, "terminal": result.success,
                                  "judgments": [asdict(result)]}
    # Evaluate each NONTERMINAL boundary once and reuse it for both neighboring rewards.
    results = await asyncio.gather(*(judge.evaluate(s) for s in snapshots[:-1]))
    potentials = [result.progress for result in results] + [0.0]
    rewards[:] = potential_shaping(potentials, gamma, alpha)
    rewards[-1] += float(oracle_success)
    # Terminal judge measurement is separate, NOT necessary for shaping and not charged here.
    return rewards.tolist(), {"source": arm, "terminal": float(oracle_success),
                              "potentials": potentials, "judgments": [asdict(r) for r in results]}
