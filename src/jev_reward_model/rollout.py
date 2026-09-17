from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from .env import WorkflowEnv, WorkflowTask

SYSTEM = """You are an agent in a deterministic workflow environment.
Emit exactly one action per turn.
Use `DO <milestone text>` to perform a milestone.
When all milestones are complete, emit the exact completion string.
Obey all constraints for the full trajectory. Do not explain your action."""


@dataclass(slots=True)
class Transition:
    prompt_ids: torch.Tensor
    action_ids: torch.Tensor
    old_logprob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool


def actor_messages(task: WorkflowTask, actions: list[str]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": str(task.public_spec())}]
    for action in actions:
        messages.append({"role": "assistant", "content": action})
        messages.append({"role": "user", "content": "Continue with exactly one next action."})
    return messages


def clone_state(state):
    return copy.deepcopy(state)


def generate_action(model, processor, task: WorkflowTask, actions: list[str], max_new_tokens: int) -> tuple[str, Any, Any]:
    messages = actor_messages(task, actions)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_p=0.95, top_k=20
        )
    generated = output[:, inputs["input_ids"].shape[-1]:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip().splitlines()[0]
    return text, inputs, generated


def run_episode(model, processor, task: WorkflowTask, reward_source, max_steps: int, max_new_tokens: int):
    env = WorkflowEnv(task, max_steps=max_steps)
    reward_source.reset(env.state)
    records = []
    for _ in range(max_steps):
        action, inputs, generated = generate_action(model, processor, task, env.state.actions, max_new_tokens)
        before = clone_state(env.state)
        state, done = env.step(action)
        reward = reward_source.transition(before, state)
        if done:
            reward += reward_source.terminal(state)
        records.append({"inputs": inputs, "generated": generated, "action": action, "reward": reward, "done": done})
        if done:
            break
    return records, env.state
