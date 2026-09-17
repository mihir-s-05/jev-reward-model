"""Reward assignment is the only experimental difference between arms."""
from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig


def make_judge(cfg: ExperimentConfig, run_dir: Path):
    if cfg.reward.startswith("jev_"):
        from .jev import JevClient
        return JevClient(cfg, run_dir)
    if cfg.reward.startswith("qwen_"):
        from .judge import QwenJudge
        return QwenJudge(cfg, run_dir)
    return None


def potential_rewards(base: list[float], phi: list[float], gamma: float, alpha: float) -> list[float]:
    if len(phi) != len(base) + 1 or phi[-1] != 0:
        raise ValueError("Complete episodes require one potential per boundary and terminal Phi=0")
    return [r + alpha * (gamma * phi[t + 1] - phi[t]) for t, r in enumerate(base)]


def assign_rewards(episodes: list, cfg: ExperimentConfig, judge) -> None:
    """Judges receive only saved PUBLIC prefixes, never the oracle dictionary."""
    if cfg.reward in {"jev_terminal", "qwen_judge"}:
        answers = judge.evaluate_many([e.states[-1] for e in episodes], "terminal")
        for episode, answer in zip(episodes, answers):
            episode.judge_success = answer["success"]
            episode.rewards = [0.0] * (len(episode.turns) - 1) + [answer["success"]]
    else:
        for episode in episodes:
            episode.rewards = [0.0] * (len(episode.turns) - 1) + [episode.oracle["success"]]
        if cfg.reward in {"jev_shaping", "qwen_shaping"}:
            prefixes = [s for e in episodes for s in e.states[:-1]]
            answers = iter(judge.evaluate_many(prefixes, "progress"))
            for episode in episodes:
                episode.potentials = [next(answers)["progress"] for _ in episode.turns] + [0.0]
                episode.rewards = potential_rewards(episode.rewards, episode.potentials,
                                                     cfg.gamma, cfg.shaping_alpha)
