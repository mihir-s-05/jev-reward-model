"""Qwen3.5 text-backbone LoRA policy and shared-backbone value head.

Load the official multimodal checkpoint through its native class, then retain only
its text backbone and tied LM head. This avoids guessing checkpoint key remappings.
Generation is deliberately plain categorical sampling: the distribution whose
log-probabilities PPO uses is exactly the distribution that generated the tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gc
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig, canonical


@dataclass
class Turn:
    prompt_ids: list[int]
    response_ids: list[int]
    text: str
    old_logprobs: list[float]
    old_values: list[float]
    ref_logprobs: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    ended_with_eos: bool = False


class ContextBudgetError(RuntimeError):
    pass


class Policy(nn.Module):
    def __init__(self, cfg: ModelConfig, seed: int):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("Install the train extra with Qwen3.5-capable Transformers/PEFT.") from exc
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. No silent CPU training.")
        if cfg.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(cfg.device))
        if cfg.dtype == "bfloat16" and (not cfg.device.startswith("cuda") or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("bfloat16 configuration requires a supporting CUDA GPU.")
        self.cfg = cfg
        torch.manual_seed(seed)
        full_config = AutoConfig.from_pretrained(cfg.name, revision=cfg.revision, trust_remote_code=False)
        self.revision = getattr(full_config, "_commit_hash", None) or cfg.revision
        if full_config.model_type != "qwen3_5":
            raise ValueError("This loader intentionally supports dense Qwen3.5 checkpoints only.")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name, revision=self.revision, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError("Checkpoint tokenizer must supply its official chat template.")
        # Native CPU loading prevents allocating the unused vision tower on the GPU.
        full, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg.name, revision=self.revision, config=full_config,
            dtype=getattr(torch, cfg.dtype), attn_implementation=cfg.attention,
            trust_remote_code=False, output_loading_info=True,
        )
        missing = [k for k in info.get("missing_keys", []) if "language_model" in k]
        if missing or info.get("mismatched_keys"):
            raise RuntimeError(f"Incomplete text checkpoint load: {missing}, {info.get('mismatched_keys')}")
        full.tie_weights()
        text = full.model.language_model
        self.lm_head = full.get_output_embeddings()
        eos = full.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.eos_ids.add(self.tokenizer.eos_token_id)
        self.eos_ids.discard(None)
        self.load_info = {k: list(info.get(k, [])) for k in ("missing_keys", "unexpected_keys")}
        tied = full.config.tie_word_embeddings
        # Select actual linear layers, including DeltaNet projections; not just q_proj/v_proj.
        targets = [name for name, layer in text.named_modules() if isinstance(layer, nn.Linear)]
        if not targets:
            raise RuntimeError("No text projection layers found for LoRA.")
        self.backbone = get_peft_model(text, LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
            bias="none", target_modules=targets,
        ))
        self.value_head = nn.Linear(text.config.hidden_size, 1, dtype=torch.float32)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        del full, text
        gc.collect()
        self.to(cfg.device)
        if tied:
            # Keep one embedding/LM-head parameter after device movement.
            self.lm_head.weight = self.backbone.get_input_embeddings().weight
        for parameter in self.lm_head.parameters():
            parameter.requires_grad_(False)
        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.backbone.enable_input_require_grads()
        # Stochastic dropout must not change rollout/update policy probabilities.
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        self.generator = torch.Generator(device=cfg.device).manual_seed(seed + 1009)
        self.eval()

    def prompt(self, snapshot: dict) -> list[int]:
        messages = [
            {"role": "system", "content":
             "You operate a simulated warehouse. Follow the goal and tool contract. "
             "Respond with one tool-call JSON object. Do not output explanations or reasoning."},
            {"role": "user", "content": canonical(snapshot)},
        ]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False)
        if len(ids) + self.cfg.max_new_tokens > self.cfg.max_context:
            raise ContextBudgetError("Full observed history exceeds the configured policy context.")
        return ids

    @torch.no_grad()
    def sample(self, snapshot: dict, greedy: bool = False) -> Turn:
        self.eval()
        prompt = self.prompt(snapshot)
        tokens = torch.tensor([prompt], device=self.cfg.device, dtype=torch.long)
        cache = None
        completion, logprobs, values = [], [], []
        offset = 0
        for _ in range(self.cfg.max_new_tokens):
            total = offset + tokens.shape[1]
            output = self.backbone(
                input_ids=tokens, attention_mask=torch.ones((1, total), device=tokens.device, dtype=torch.long),
                position_ids=torch.arange(offset, total, device=tokens.device).unsqueeze(0),
                past_key_values=cache, use_cache=True, return_dict=True,
            )
            hidden = output.last_hidden_state[:, -1, :]
            logits = self.lm_head(hidden).float() / self.cfg.temperature
            logs = F.log_softmax(logits, dim=-1)
            token = logits.argmax(dim=-1) if greedy else torch.multinomial(
                logs.exp(), 1, generator=self.generator).squeeze(-1)
            value = self.value_head(hidden.float()).squeeze()
            tid = token.item()
            completion.append(tid)
            logprobs.append(logs[0, tid].item())
            values.append(value.item())
            if tid in self.eos_ids:
                break
            cache, tokens, offset = output.past_key_values, token.reshape(1, 1), total
            if cache is None:
                raise RuntimeError("Qwen text backbone did not return a generation cache.")
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        return Turn(prompt, completion, text, logprobs, values,
                    ended_with_eos=completion[-1] in self.eos_ids)

    def token_statistics(self, turn: Turn) -> tuple[torch.Tensor, torch.Tensor]:
        """Only completion prediction positions contribute; never prompt/tool-result tokens."""
        ids = torch.tensor([turn.prompt_ids + turn.response_ids], device=self.cfg.device)
        output = self.backbone(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(ids.shape[1], device=ids.device).unsqueeze(0),
            use_cache=False, return_dict=True,
        )
        start = len(turn.prompt_ids) - 1  # Hidden state BEFORE each sampled token.
        hidden = output.last_hidden_state[0, start:-1]
        labels = ids[0, len(turn.prompt_ids):]
        values = self.value_head(hidden.float()).squeeze(-1)
        chunks = []

        def project(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            logits = self.lm_head(h).float() / self.cfg.temperature
            return -F.cross_entropy(logits, y, reduction="none")

        # Recompute each small LM-head chunk during backward instead of retaining an
        # entire sequence x 248k vocabulary tensor (including unused prompt logits).
        for h, y in zip(hidden.split(self.cfg.logits_chunk), labels.split(self.cfg.logits_chunk)):
            chunks.append(checkpoint(project, h, y, use_reentrant=False)
                          if torch.is_grad_enabled() else project(h, y))
        return torch.cat(chunks), values

    @torch.no_grad()
    def reference_statistics(self, turn: Turn) -> list[float]:
        self.eval()
        with self.backbone.disable_adapter():
            logprobs, _ = self.token_statistics(turn)
        return logprobs.cpu().tolist()

    @torch.no_grad()
    def check_rollout_consistency(self, turn: Turn) -> float:
        """Fail on cache/full-forward mismatch before PPO silently trains off-policy."""
        self.eval()
        logs, _ = self.token_statistics(turn)
        old = torch.tensor(turn.old_logprobs, device=logs.device)
        error = (logs - old).abs().max().item()
        if error > self.cfg.logprob_tolerance:
            raise RuntimeError(f"Cached versus teacher-forced logprob error {error:.4g} exceeds "
                               f"{self.cfg.logprob_tolerance}; inspect kernels before training.")
        return error

    def optimizer(self, learning_rate: float, critic_learning_rate: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW([
            {"params": [p for p in self.backbone.parameters() if p.requires_grad], "lr": learning_rate},
            {"params": self.value_head.parameters(), "lr": critic_learning_rate},
        ], weight_decay=0.0, eps=1e-8)

    def save_adapter(self, directory: Path) -> None:
        self.backbone.save_pretrained(directory / "adapter", safe_serialization=True, save_embedding_layers=False)
        torch.save(self.value_head.state_dict(), directory / "value_head.pt")

    def load_adapter(self, directory: Path) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        state = load_file(str(directory / "adapter" / "adapter_model.safetensors"))
        result = set_peft_model_state_dict(self.backbone, state)
        missing_lora = [key for key in result.missing_keys if "lora_" in key]
        if result.unexpected_keys or missing_lora:
            raise RuntimeError(f"Incomplete adapter load: missing={missing_lora}, unexpected={result.unexpected_keys}")
        self.value_head.load_state_dict(torch.load(directory / "value_head.pt", map_location=self.cfg.device,
                                                  weights_only=True), strict=True)

    def metadata(self) -> dict[str, Any]:
        return {"model": self.cfg.name, "revision": self.revision,
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "parameters": sum(p.numel() for p in self.parameters()), "load_info": self.load_info,
                "eos_ids": sorted(self.eos_ids), "device": self.cfg.device,
                "gpu": torch.cuda.get_device_name(self.cfg.device) if self.cfg.device.startswith("cuda") else None}
