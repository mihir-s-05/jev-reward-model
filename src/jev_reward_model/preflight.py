"""User-run actor compatibility check, NOT a training experiment or evaluator request."""
from __future__ import annotations

import argparse
import json

import torch

from .config import ExperimentConfig
from .data import load
from .env import WorkflowEnv
from .policy import Actor
from .train import seed_all


def inspect_actor(actor, states: list[dict], cfg: ExperimentConfig, max_logprob_error: float) -> dict:
    """Compare batched cached generation with uncached score(), then check finite nonzero grads.

    Works on the configured actor device (CUDA or CPU). Makes no optimizer step or API call.
    """
    prompts = [actor.prompt(s) for s in states]
    width = max(x.numel() for x in prompts)
    ids = torch.full((len(prompts), width), actor.tokenizer.pad_token_id,
                     dtype=torch.long, device=actor.device)
    mask = torch.zeros_like(ids)
    for i, prompt in enumerate(prompts):
        ids[i, -prompt.numel():], mask[i, -prompt.numel():] = prompt.to(actor.device), 1
    with torch.no_grad():
        sampled = actor.model.generate(input_ids=ids, attention_mask=mask,
            generation_config=actor.generation_config(max_new_tokens=min(16, cfg.max_new_tokens),
                                                       output_scores=True, return_dict_in_generate=True))
    errors, actions = [], []
    for i, sequence in enumerate(sampled.sequences[:, width:]):
        stop = next((j + 1 for j, token in enumerate(sequence.tolist()) if token in actor.eos_ids), len(sequence))
        action = sequence[:stop].cpu()
        actions.append(action)
        behavior = sum(torch.log_softmax(sampled.scores[j][i].float(), -1)[token]
                       for j, token in enumerate(action.tolist())).item()
        # Compare batched cached generation against the unpadded, uncached training path.
        actor.model.train()
        with torch.no_grad():
            rescored, _ = actor.score(prompts[i], action)
        errors.append(abs(behavior - rescored.item()))
    del sampled
    if max(errors) > max_logprob_error:
        raise RuntimeError(f"Generation/scoring mismatch {errors}; inspect dtype, kernels and padding before PPO")
    actor.model.zero_grad(set_to_none=True)
    lp, value = actor.score(prompts[0], actions[0])
    (-lp.mean() + (value - 1).square().mean()).backward()
    for name, parameters in (("actor", actor.model.parameters()), ("critic", actor.value_head.parameters())):
        grads = [x.grad for x in parameters if x.requires_grad and x.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):
            raise RuntimeError(f"Missing/nonfinite {name} gradients")
        if not any(torch.count_nonzero(g).item() for g in grads):
            raise RuntimeError(f"All {name} gradients are zero")
    return {"status": "Actor preflight passed; no optimizer step or API call",
            "joint_logprob_errors": errors, "revision": actor.resolved_revision,
            "device": str(actor.device), "dtype": cfg.dtype}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/grounded.yaml")
    p.add_argument("--model-revision")
    p.add_argument("--max-logprob-error", type=float, default=0.05,
                   help="Maximum absolute JOINT log-probability mismatch; do not loosen to hide a bug")
    args = p.parse_args()
    cfg = ExperimentConfig.load(args.config)
    if args.model_revision:
        cfg.model_revision = args.model_revision
        cfg.validate()
    seed_all(cfg.seed)
    actor = Actor(cfg)
    task = load(cfg.train_data, "train")[0]
    env = WorkflowEnv(task)
    states = [env.state.public_state()]
    env.step("DO " + task.milestones[0])
    states.append(env.state.public_state())
    print(json.dumps(inspect_actor(actor, states, cfg, args.max_logprob_error), indent=2))


if __name__ == "__main__":
    main()
