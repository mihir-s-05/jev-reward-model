"""Token-level clipped PPO with tool-boundary discounting and generalized advantages.

Within a generated action, discount and GAE continuation are 1. At a tool boundary,
they are gamma and lambda. This does not artificially discount a long JSON argument
as though every token were an extra environment step. Episodes are always complete.
"""
from __future__ import annotations

import random
import numpy as np
import torch

from .config import PPOConfig
from .policy import Policy, Turn


def generalized_advantages(rewards: np.ndarray, values: np.ndarray, discounts: np.ndarray,
                           lambdas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not (rewards.shape == values.shape == discounts.shape == lambdas.shape) or rewards.ndim != 1:
        raise ValueError("GAE arrays must be equal one-dimensional shapes.")
    if len(rewards) == 0 or discounts[-1] != 0:
        raise ValueError("A complete episode with zero final bootstrap is required.")
    if not all(np.isfinite(x).all() for x in (rewards, values, discounts, lambdas)):
        raise ValueError("Non-finite GAE input.")
    advantages = np.empty_like(rewards, dtype=np.float64)
    accumulator, next_value = 0.0, 0.0
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + discounts[t] * next_value - values[t]
        accumulator = delta + discounts[t] * lambdas[t] * accumulator
        advantages[t], next_value = accumulator, values[t]
    return advantages, advantages + values


def prepare_episode(turns: list[Turn], env_rewards: list[float], cfg: PPOConfig) -> dict:
    if len(turns) != len(env_rewards) or not turns:
        raise ValueError("One environment reward per sampled action is required.")
    lengths = [len(t.response_ids) for t in turns]
    if any(n == 0 for n in lengths):
        raise ValueError("Cannot train an empty action.")
    old = np.concatenate([t.old_logprobs for t in turns]).astype(np.float64)
    ref = np.concatenate([t.ref_logprobs for t in turns]).astype(np.float64)
    values = np.concatenate([t.old_values for t in turns]).astype(np.float64)
    if old.shape != ref.shape or len(old) != sum(lengths) or values.shape != old.shape:
        raise ValueError("Incomplete token statistics.")
    rewards = -cfg.kl_coefficient * (old - ref)  # Sampled log-ratio KL shaping, frozen per batch.
    discounts, lambdas = np.ones(len(old)), np.ones(len(old))
    ends = np.cumsum(lengths) - 1
    rewards[ends] += np.asarray(env_rewards)
    discounts[ends], lambdas[ends] = cfg.gamma, cfg.gae_lambda
    discounts[-1] = 0.0  # finish, action budget and context budget are all task terminals.
    advantages, returns = generalized_advantages(rewards, values, discounts, lambdas)
    offset = 0
    for turn, length in zip(turns, lengths):
        turn.advantages = advantages[offset:offset + length].tolist()
        turn.returns = returns[offset:offset + length].tolist()
        offset += length
    return {"reference_kl_sample_mean": float(np.mean(old - ref)),
            "environment_return": float(sum(env_rewards)),
            "regularized_token_return": float(sum(rewards)), "generated_tokens": len(old)}


def normalize_advantages(turns: list[Turn]) -> None:
    values = np.concatenate([t.advantages for t in turns])
    mean, std = float(values.mean()), float(values.std())
    for turn in turns:
        turn.advantages = ((np.asarray(turn.advantages) - mean) / max(std, 1e-8)).tolist()


def clipped_losses(logprobs: torch.Tensor, values: torch.Tensor, old_logprobs: torch.Tensor,
                   old_values: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor,
                   cfg: PPOConfig) -> tuple[torch.Tensor, dict]:
    logratio = logprobs - old_logprobs
    if not torch.isfinite(logratio).all() or logratio.abs().max().item() > 20:
        raise FloatingPointError("PPO importance ratios are unstable; aborting rather than hiding them.")
    ratio = logratio.exp()
    policy = torch.maximum(-advantages * ratio,
                           -advantages * ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio))
    clipped_value = old_values + (values - old_values).clamp(-cfg.value_clip, cfg.value_clip)
    value = 0.5 * torch.maximum((values - returns).square(), (clipped_value - returns).square())
    loss = policy.sum() + cfg.value_coefficient * value.sum()
    metrics = {"policy_loss_sum": policy.detach().sum().item(),
               "value_loss_sum": value.detach().sum().item(),
               "approx_kl_sum": (ratio - 1 - logratio).detach().sum().item(),
               "clip_count": ((ratio - 1).abs() > cfg.clip_ratio).sum().item(),
               "tokens": len(logprobs)}
    return loss, metrics


def update_policy(policy: Policy, optimizer: torch.optim.Optimizer, turns: list[Turn],
                  cfg: PPOConfig, rng: random.Random) -> dict:
    normalize_advantages(turns)
    totals = {"policy_loss_sum": 0.0, "value_loss_sum": 0.0, "approx_kl_sum": 0.0,
              "clip_count": 0.0, "tokens": 0}
    updates, stop, last_norm = 0, False, 0.0
    for _ in range(cfg.epochs):
        indices = list(range(len(turns)))
        rng.shuffle(indices)
        for start in range(0, len(indices), cfg.minibatch_turns):
            batch = [turns[i] for i in indices[start:start + cfg.minibatch_turns]]
            denominator = sum(len(t.response_ids) for t in batch)
            optimizer.zero_grad(set_to_none=True)
            policy.train()  # Enables checkpointing; all dropout probabilities are zero.
            batch_kl = 0.0
            for turn in batch:
                logs, values = policy.token_statistics(turn)
                tensors = [torch.tensor(x, device=logs.device, dtype=torch.float32) for x in
                           (turn.old_logprobs, turn.old_values, turn.advantages, turn.returns)]
                loss, metrics = clipped_losses(logs, values, *tensors, cfg)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss.")
                (loss / denominator).backward()  # Token-weighted accumulation; no padded tokens.
                batch_kl += metrics["approx_kl_sum"]
                for key in totals:
                    totals[key] += metrics[key]
            if batch_kl / denominator > cfg.target_kl:
                optimizer.zero_grad(set_to_none=True)
                stop = True
                break  # Do not apply a batch already outside the trust-region target.
            norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],
                                                 cfg.max_grad_norm, error_if_nonfinite=True)
            last_norm = norm.item()
            optimizer.step()
            updates += 1
        if stop:
            break
    policy.eval()
    count = max(1, totals.pop("tokens"))
    return {key.removesuffix("_sum"): val / count for key, val in totals.items()} | {
        "optimizer_steps": updates, "early_stopped": stop, "gradient_norm": last_norm}
