from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class PPORollout:
    prompt_ids: torch.Tensor
    generated_ids: torch.Tensor
    old_logprob: torch.Tensor
    old_value: torch.Tensor
    reward: float
    done: bool
    advantage: torch.Tensor | None = None
    return_: torch.Tensor | None = None


class ValueHead(nn.Module):
    """Small scalar critic over the actor's final prompt hidden state."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).squeeze(-1)


def _text_backbone(model):
    """Return Qwen3.5's language backbone, preserving PEFT wrappers when possible."""
    base = model.base_model.model if hasattr(model, "base_model") else model
    return base.model if hasattr(base, "model") else base


def action_logprob(model, prompt_ids: torch.Tensor, generated_ids: torch.Tensor) -> torch.Tensor:
    full = torch.cat([prompt_ids, generated_ids], dim=1)
    attention = torch.ones_like(full)
    logits = model(input_ids=full, attention_mask=attention).logits[:, :-1]
    labels = full[:, 1:]
    token_lp = F.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = prompt_ids.shape[1] - 1
    return token_lp[:, start:].sum(dim=1)


def state_value(model, value_head: ValueHead, prompt_ids: torch.Tensor) -> torch.Tensor:
    out = _text_backbone(model)(input_ids=prompt_ids, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[-1][:, -1, :].float()
    return value_head(hidden)


def add_gae(rollout: list[PPORollout], gamma: float, lam: float) -> None:
    """Compute GAE on one complete episode using frozen rollout values."""
    gae = torch.zeros_like(rollout[0].old_value)
    next_value = torch.zeros_like(rollout[0].old_value)
    for item in reversed(rollout):
        nonterminal = 0.0 if item.done else 1.0
        delta = item.old_value.new_tensor(item.reward) + gamma * next_value * nonterminal - item.old_value
        gae = delta + gamma * lam * nonterminal * gae
        item.advantage = gae.detach()
        item.return_ = (gae + item.old_value).detach()
        next_value = item.old_value


def ppo_loss(
    model,
    value_head: ValueHead,
    item: PPORollout,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
) -> torch.Tensor:
    new_logprob = action_logprob(model, item.prompt_ids, item.generated_ids)
    ratio = torch.exp(new_logprob - item.old_logprob)
    unclipped = ratio * item.advantage
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * item.advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    value = state_value(model, value_head, item.prompt_ids)
    value_loss = 0.5 * F.mse_loss(value, item.return_)
    entropy_bonus = new_logprob.new_zeros(())
    return policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
