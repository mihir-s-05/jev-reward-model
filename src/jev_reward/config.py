"""Validated experiment configuration; one shared configuration for every arm."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import os
import yaml

ARMS = ("oracle_terminal", "jev_terminal", "jev_shaping", "llm_terminal", "llm_shaping")
PRIMARY_ARMS = ARMS[:4]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass
class TaskConfig:
    train_orders: list[int] = field(default_factory=lambda: [1, 3])
    long_orders: list[int] = field(default_factory=lambda: [5, 7])
    extra_steps: int = 6


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3.5-4B"
    revision: str = "main"  # Resolved to a commit before loading weights/tokenizer.
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    lora_rank: int = 16
    lora_alpha: int = 32
    max_context: int = 16384
    max_new_tokens: int = 128
    temperature: float = 1.0
    gradient_checkpointing: bool = True
    logits_chunk: int = 16
    logprob_tolerance: float = 0.15


@dataclass
class PPOConfig:
    updates: int = 100
    episodes_per_update: int = 8
    epochs: int = 2
    minibatch_turns: int = 8
    learning_rate: float = 1e-5
    critic_learning_rate: float = 1e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    kl_coefficient: float = 0.02
    target_kl: float = 0.03
    max_grad_norm: float = 1.0
    shaping_alpha: float = 0.5
    eval_every: int = 10
    eval_episodes: int = 24
    checkpoint_every: int = 10


@dataclass
class JudgeConfig:
    jev_url: str = "https://api.typesafe.ai/v1/systemone"
    jev_model: str = "jev-latest"
    jev_input_per_million: float = 0.042  # Configured estimate, not live billing.
    llm_base_url: str = "http://localhost:8001/v1"
    llm_model: str = "Qwen/Qwen3.5-9B"
    llm_input_per_million: float | None = None
    llm_output_per_million: float | None = None
    concurrency: int = 8
    timeout_seconds: float = 60.0
    attempts: int = 5
    context: str = "full"  # full / recent / ledger; policy always sees full history.
    recent_events: int = 4
    max_request_bytes: int = 100000
    # Byte guard is deliberately conservative, NOT an exact Jev token counter.
    llm_max_tokens: int = 384


@dataclass
class Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def validate(self) -> None:
        for pair in (self.task.train_orders, self.task.long_orders):
            if len(pair) != 2 or not (1 <= pair[0] <= pair[1]):
                raise ValueError("Order ranges must be [positive minimum, maximum].")
        positive = [self.task.extra_steps, self.model.lora_rank, self.model.lora_alpha,
                    self.model.max_context, self.model.max_new_tokens, self.model.logits_chunk,
                    self.model.logprob_tolerance, self.ppo.updates, self.ppo.episodes_per_update,
                    self.ppo.epochs, self.ppo.minibatch_turns, self.ppo.learning_rate,
                    self.ppo.critic_learning_rate, self.ppo.target_kl, self.ppo.max_grad_norm,
                    self.ppo.eval_every, self.ppo.eval_episodes, self.ppo.checkpoint_every,
                    self.judge.concurrency, self.judge.attempts, self.judge.timeout_seconds,
                    self.judge.max_request_bytes, self.judge.llm_max_tokens]
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Counts, rates, budgets, and tolerances must be positive and finite.")
        if not 0 < self.ppo.gamma <= 1 or not 0 <= self.ppo.gae_lambda <= 1:
            raise ValueError("gamma must be in (0,1], lambda in [0,1].")
        if not 0 < self.ppo.clip_ratio < 1 or self.ppo.value_clip <= 0:
            raise ValueError("Invalid PPO clipping parameters.")
        if not math.isfinite(self.model.temperature) or self.model.temperature <= 0:
            raise ValueError("Training temperature must be positive and finite.")
        for x in (self.ppo.kl_coefficient, self.ppo.value_coefficient, self.ppo.shaping_alpha,
                  self.judge.jev_input_per_million, self.judge.llm_input_per_million,
                  self.judge.llm_output_per_million):
            if x is not None and (not math.isfinite(x) or x < 0):
                raise ValueError("Coefficients/prices must be nonnegative finite numbers.")
        if self.model.max_context <= self.model.max_new_tokens + 256:
            raise ValueError("Context must leave room for a prompt and a complete action.")
        if self.model.dtype not in ("bfloat16", "float32"):
            raise ValueError("Use bfloat16 (CUDA) or float32; fp16 needs an unimplemented scaler.")
        if self.judge.context not in ("full", "recent", "ledger") or self.judge.recent_events < 1:
            raise ValueError("Invalid judge context mode/recent-events count.")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> Config:
    raw = {} if path is None else (yaml.safe_load(Path(path).read_text()) or {})
    classes = {"task": TaskConfig, "model": ModelConfig, "ppo": PPOConfig, "judge": JudgeConfig}
    if not isinstance(raw, dict) or set(raw) - set(classes):
        raise ValueError("Unknown configuration section.")
    sections = {}
    for name, cls in classes.items():
        section = raw.get(name, {})
        if not isinstance(section, dict) or set(section) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown configuration key in {name}.")
        sections[name] = cls(**section)
    cfg = Config(**sections)
    # Resolve public endpoint/model overrides once so manifests record the actual judge.
    cfg.judge.llm_base_url = os.environ.get("JUDGE_BASE_URL", cfg.judge.llm_base_url)
    cfg.judge.llm_model = os.environ.get("JUDGE_MODEL", cfg.judge.llm_model)
    cfg.validate()
    return cfg
