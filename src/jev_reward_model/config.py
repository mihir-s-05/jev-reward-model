from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    seed: int
    model: str
    train_data: str
    reward: str
    output_dir: str
    steps: int = 200
    rollouts_per_step: int = 8
    max_episode_steps: int = 8
    max_new_tokens: int = 64
    learning_rate: float = 1e-5
    ppo_epochs: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    shaping_alpha: float = 0.5
    lora_r: int = 16
    lora_alpha: int = 32

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))
