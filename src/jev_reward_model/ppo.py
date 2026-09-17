"""Turn-level PPO: a whole generated command is one macro-action, not a mean token score."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Turn:
    prompt_ids: torch.Tensor  # CPU, unpadded; preserve exact sampled tokens.
    action_ids: torch.Tensor
    old_logprob: float = 0.0
    old_value: float = 0.0
    ref_logprob: float = 0.0
    reward: float = 0.0
    advantage: float = 0.0
    return_: float = 0.0


def add_gae(turns: list[Turn], gamma: float, lam: float) -> None:
    """Every collection is a complete task episode; final bootstrap is exactly zero."""
    if not turns:
        raise ValueError("Empty episode")
    advantage, next_value = 0.0, 0.0
    for turn in reversed(turns):
        delta = turn.reward + gamma * next_value - turn.old_value
        advantage = delta + gamma * lam * advantage
        turn.advantage = advantage
        turn.return_ = advantage + turn.old_value
        next_value = turn.old_value


def clipped_loss(new_lp, value, old_lp, old_value, advantage, returns, cfg):
    log_ratio = new_lp - old_lp
    if not torch.isfinite(log_ratio).all() or log_ratio.abs().max().item() > 20:
        raise RuntimeError("Unstable macro-action ratio; lower LR/epochs or shorten commands")
    ratio = log_ratio.exp()
    policy = -torch.minimum(ratio * advantage,
                            ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * advantage).mean()
    clipped_value = old_value + (value - old_value).clamp(-cfg.value_clip, cfg.value_clip)
    value_loss = 0.5 * torch.maximum((value - returns).square(), (clipped_value - returns).square()).mean()
    kl = (ratio - 1 - log_ratio).mean()
    return policy + cfg.value_coef * value_loss, {
        "policy_loss": policy.detach(), "value_loss": value_loss.detach(),
        "approx_kl": kl.detach(), "clip_fraction": ((ratio - 1).abs() > cfg.clip_range).float().mean().detach()}


def update(actor, turns: list[Turn], optimizer, cfg, rng) -> dict:
    """Accumulate per-command microbatches into proper minibatch optimizer steps.

    No reference model copy, vocabulary-wide entropy allocation, or hidden reward
    normalization. The critic shares actor features and has its own learning rate.
    """
    advantages = torch.tensor([t.advantage for t in turns], dtype=torch.float32)
    mean, std = advantages.mean().item(), advantages.std(unbiased=False).clamp_min(1e-8).item()
    for turn in turns:
        turn.advantage = (turn.advantage - mean) / std
    actor.model.train()  # All backbone and LoRA dropout probabilities are explicitly zero.
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    totals = dict(policy_loss=0.0, value_loss=0.0, approx_kl=0.0, clip_fraction=0.0)
    count, updates, stopped = 0, 0, False
    for _ in range(cfg.ppo_epochs):
        order = list(range(len(turns)))
        rng.shuffle(order)
        for start in range(0, len(order), cfg.minibatch_size):
            mini = [turns[j] for j in order[start:start + cfg.minibatch_size]]
            optimizer.zero_grad(set_to_none=True)
            local = {k: 0.0 for k in totals}
            for turn in mini:
                lp, value = actor.score(turn.prompt_ids, turn.action_ids)
                scalar = lambda x: torch.tensor([x], device=lp.device, dtype=torch.float32)
                loss, metrics = clipped_loss(lp, value, scalar(turn.old_logprob), scalar(turn.old_value),
                                            scalar(turn.advantage), scalar(turn.return_), cfg)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite PPO loss")
                (loss / len(mini)).backward()
                for key, val in metrics.items():
                    local[key] += val.item() / len(mini)
            # Skip this minibatch entirely if drift is already too high.
            if local["approx_kl"] > cfg.target_kl:
                optimizer.zero_grad(set_to_none=True)
                stopped = True
                break
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            for key in totals:
                totals[key] += local[key] * len(mini)
            count += len(mini)
        if stopped:
            break
    actor.model.eval()
    return {**{k: v / max(1, count) for k, v in totals.items()},
            "optimizer_steps": updates, "kl_early_stop": stopped,
            "advantage_mean_raw": mean, "advantage_std_raw": std}
