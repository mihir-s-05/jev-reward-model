from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkflowTask:
    task_id: str
    goal: str
    milestones: tuple[str, ...]
    forbidden: tuple[str, ...]
    required_final: str

    def public_spec(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "milestones": list(self.milestones),
            "constraints": [f"Do not {x}." for x in self.forbidden],
            "completion_rule": f"Finish by emitting: {self.required_final}",
        }


@dataclass(slots=True)
class WorkflowState:
    task: WorkflowTask
    completed: set[int] = field(default_factory=set)
    violations: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    done: bool = False

    @property
    def success(self) -> bool:
        return self.done and len(self.completed) == len(self.task.milestones) and not self.violations

    @property
    def progress(self) -> float:
        if not self.task.milestones:
            return 1.0
        return len(self.completed) / len(self.task.milestones)

    def public_state(self) -> dict[str, Any]:
        # Do not expose parser state such as completed indices or exact success.
        return {"task": self.task.public_spec(), "actions_so_far": self.actions}


class WorkflowEnv:
    """Small deterministic environment with irreversible constraint violations.

    Actor actions are intentionally textual. A valid milestone action is
    `DO <exact milestone text>`. The terminal action is the task's required
    final string. Anything else is allowed but makes no simulator progress.
    """

    def __init__(self, task: WorkflowTask, max_steps: int = 8):
        self.state = WorkflowState(task=task)
        self.max_steps = max_steps

    def step(self, action: str) -> tuple[WorkflowState, bool]:
        if self.state.done:
            raise RuntimeError("episode already finished")

        action = action.strip()
        self.state.actions.append(action)
        lowered = action.lower()

        for forbidden in self.state.task.forbidden:
            if forbidden.lower() in lowered and forbidden not in self.state.violations:
                self.state.violations.append(forbidden)

        if action == self.state.task.required_final:
            self.state.done = True
        elif action.startswith("DO "):
            payload = action[3:].strip()
            for i, milestone in enumerate(self.state.task.milestones):
                if payload == milestone:
                    # Enforce order to create delayed dependencies.
                    if i == 0 or (i - 1) in self.state.completed:
                        self.state.completed.add(i)
                    break

        if len(self.state.actions) >= self.max_steps:
            self.state.done = True

        return self.state, self.state.done

    def grounded_terminal_reward(self) -> float:
        return 1.0 if self.state.success else 0.0
