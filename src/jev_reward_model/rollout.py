"""Synchronous on-policy batches; asynchronous judges run only after policy collection."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .env import WorkflowEnv, WorkflowTask
from .ppo import Turn


@dataclass
class Episode:
    task: WorkflowTask
    turns: list[Turn] = field(default_factory=list)
    states: list[dict] = field(default_factory=list)
    oracle: dict = field(default_factory=dict)
    full_final: dict = field(default_factory=dict)
    rewards: list[float] = field(default_factory=list)
    potentials: list[float] = field(default_factory=list)
    judge_success: float | None = None

    def record(self, include_tokens: bool = False) -> dict:
        result = {"task_id": self.task.task_id, "family": self.task.family, "split": self.task.split,
                  "operations": len(self.task.milestones), "oracle": self.oracle,
                  "public_final": self.full_final, "judge_final": self.states[-1], "rewards": self.rewards,
                  "potentials": self.potentials, "judge_success": self.judge_success,
                  "action_tokens": sum(t.action_ids.numel() for t in self.turns),
                  "prompt_tokens": sum(t.prompt_ids.numel() for t in self.turns),
                  "old_logprobs": [t.old_logprob for t in self.turns],
                  "old_values": [t.old_value for t in self.turns],
                  "ref_logprobs": [t.ref_logprob for t in self.turns]}
        if include_tokens:
            result["tokens"] = [{"prompt": t.prompt_ids.tolist(), "action": t.action_ids.tolist()}
                                for t in self.turns]
        return result


@torch.no_grad()
def collect(actor, tasks: list[WorkflowTask], cfg, training: bool = True) -> list[Episode]:
    actor.model.eval()
    envs = [WorkflowEnv(t) for t in tasks]
    episodes = [Episode(t, states=[env.state.public_state(cfg.judge_view, cfg.recent_events)])
                for t, env in zip(tasks, envs)]
    active = list(range(len(tasks)))
    while active:
        for start in range(0, len(active), cfg.generation_batch_size):
            indices = active[start:start + cfg.generation_batch_size]
            # Actor always gets full evidence; judge context ablations do not change the policy task.
            prompts = [actor.prompt(envs[i].state.public_state()) for i in indices]
            actions = actor.generate(prompts, greedy=not training)
            for i, prompt, action in zip(indices, prompts, actions):
                turn = Turn(prompt, action)
                if training:
                    lp, value = actor.score(prompt, action)
                    turn.old_logprob, turn.old_value = lp.item(), value.item()
                    if cfg.reference_kl_coef:
                        ref, _ = actor.score(prompt, action, reference=True)
                        turn.ref_logprob = ref.item()
                    else:
                        turn.ref_logprob = turn.old_logprob
                envs[i].step(actor.decode(action))
                episodes[i].turns.append(turn)
                episodes[i].states.append(envs[i].state.public_state(cfg.judge_view, cfg.recent_events))
        active = [i for i in active if not envs[i].state.done]
    for episode, env in zip(episodes, envs):
        episode.oracle = env.oracle()
        episode.full_final = env.state.public_state()
    return episodes
