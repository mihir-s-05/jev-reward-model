"""Small, strict experiment config; paths are relative to the working directory."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

ARMS = ("grounded", "jev_terminal", "jev_shaping", "qwen_judge", "qwen_shaping")


def normalize_device(value: str) -> str:
    """Accept cpu / cpu:0 / cuda / cuda:<index>; store a canonical torch device string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("device must be a non-empty string")
    text = value.strip()
    lowered = text.lower()
    if lowered == "cpu" or lowered.startswith("cpu:"):
        if lowered != "cpu" and lowered.split(":", 1)[1] != "0":
            raise ValueError("CPU device must be 'cpu' or 'cpu:0'")
        return "cpu"
    if lowered == "cuda" or lowered.startswith("cuda:"):
        if lowered == "cuda":
            return "cuda:0"
        suffix = lowered.split(":", 1)[1]
        if not suffix.isdigit():
            raise ValueError("CUDA device must be cuda:<nonnegative integer>")
        return f"cuda:{int(suffix)}"
    raise ValueError("device must be cpu or cuda[:index]")


@dataclass
class ExperimentConfig:
    reward: str = "grounded"
    seed: int = 0
    model: str = "Qwen/Qwen3.5-4B"
    model_revision: str = "main"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    train_data: str = "data/train.jsonl"
    eval_data: str = "data/validation.jsonl"
    output_dir: str = "runs/grounded/seed-0"
    updates: int = 100
    rollouts_per_update: int = 8
    generation_batch_size: int = 2
    max_new_tokens: int = 48
    max_context_tokens: int = 8192
    temperature: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-5
    value_learning_rate: float = 1e-4
    ppo_epochs: int = 2
    minibatch_size: int = 16
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip: float = 0.2
    value_coef: float = 0.5
    target_kl: float = 0.03
    max_grad_norm: float = 1.0
    reference_kl_coef: float = 0.0
    shaping_alpha: float = 0.5
    judge_view: str = "full"
    recent_events: int = 4
    judge_workers: int = 8
    judge_timeout: float = 60.0
    judge_attempts: int = 5
    max_judge_bytes: int = 24000
    jev_model: str = "jev-latest"
    jev_endpoint: str = "https://api.typesafe.ai/v1/systemone"
    jev_input_usd_per_million: float = 0.042
    jev_output_usd_per_million: float = 0.0
    qwen_judge_model: str = "Qwen/Qwen3.5-4B"
    qwen_judge_endpoint: str = "http://localhost:8001/v1/chat/completions"
    qwen_judge_revision: str = "UNPINNED"
    qwen_judge_max_tokens: int = 256
    qwen_judge_gpu_usd_per_hour: float | None = None
    actor_gpu_usd_per_hour: float | None = None
    eval_every: int = 10
    eval_tasks: int = 32
    checkpoint_every: int = 10

    def __post_init__(self) -> None:
        self.device = normalize_device(self.device)

    def validate(self) -> None:
        self.device = normalize_device(self.device)
        if self.reward not in ARMS:
            raise ValueError(f"reward must be one of {ARMS}")
        if self.judge_view not in {"full", "recent", "ledger"}:
            raise ValueError("judge_view must be full, recent, or ledger")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("Only float32 and bfloat16 are supported (no fp16 scaler).")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU runs require dtype=float32; bfloat16 is not supported on CPU")
        for key in ("updates", "rollouts_per_update", "generation_batch_size", "max_new_tokens",
                    "max_context_tokens", "ppo_epochs", "minibatch_size", "lora_r", "lora_alpha",
                    "recent_events", "judge_workers", "judge_attempts", "max_judge_bytes",
                    "eval_every", "eval_tasks", "checkpoint_every", "qwen_judge_max_tokens"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid gamma / gae_lambda")
        for key in ("temperature", "learning_rate", "value_learning_rate", "target_kl",
                    "max_grad_norm", "clip_range", "value_clip", "judge_timeout"):
            if not 0 < getattr(self, key) < float("inf"):
                raise ValueError(f"{key} must be positive and finite")
        for key in ("reference_kl_coef", "shaping_alpha", "value_coef", "jev_input_usd_per_million",
                    "jev_output_usd_per_million", "actor_gpu_usd_per_hour", "qwen_judge_gpu_usd_per_hour"):
            value = getattr(self, key)
            if value is not None and not 0 <= value < float("inf"):
                raise ValueError(f"{key} must be nonnegative and finite")
        if self.max_new_tokens >= self.max_context_tokens:
            raise ValueError("max_new_tokens must leave room for the prompt")

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("Config must be a YAML object")
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys (old configs are not compatible): {unknown}")
        data = dict(raw)
        device = normalize_device(data["device"]) if "device" in data else normalize_device(cls.device)
        if "dtype" not in data and device == "cpu":
            data["dtype"] = "float32"
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
