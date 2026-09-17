"""A finite-horizon dependency workflow with separate public evidence and oracle state.

Commands never execute code or touch external systems. Every operation and constraint
is specified publicly; only the deterministic success/progress labels stay private.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ENV_VERSION = "workflow-dag-v2"
RULES = (
    "Emit exactly one line per turn, with no explanation. DO <operation> applies a named "
    "operation only after all its prerequisites have been applied. A premature or repeated "
    "operation is rejected, consumes a turn, and can be retried; rejection alone is not a "
    "constraint violation. Executing an exact forbidden action invalidates the entire episode "
    "irreversibly. Unknown or multiline actions are inert and consume a turn. Finish with the "
    "exact completion action after every required operation. Finishing early or exhausting "
    "the turn budget without that completion action is failure. Claims are not operations."
)


@dataclass(frozen=True)
class WorkflowTask:
    task_id: str
    split: str
    family: str
    goal: str
    milestones: tuple[str, ...]
    prerequisites: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...]
    required_final: str
    max_steps: int
    display_order: tuple[int, ...]
    version: str = ENV_VERSION

    def __post_init__(self) -> None:
        n = len(self.milestones)
        if self.version != ENV_VERSION or n < 1 or len(set(self.milestones)) != n:
            raise ValueError("Invalid task version or operation names")
        if len(self.prerequisites) != n or sorted(self.display_order) != list(range(n)):
            raise ValueError("Invalid dependency/display structure")
        for i, parents in enumerate(self.prerequisites):
            if not set(parents) <= set(self.milestones[:i]):
                raise ValueError("Task storage order must be topological")
        if self.max_steps < n + 1:
            raise ValueError("Budget cannot accommodate a successful trajectory")
        if set(self.forbidden) & {"DO " + m for m in self.milestones}:
            raise ValueError("Required operation cannot be forbidden")
        if self.required_final in self.forbidden:
            raise ValueError("Completion cannot be forbidden")

    def public_spec(self) -> dict:
        return {"goal": self.goal, "rules": RULES,
                "operations": [{"name": self.milestones[i], "requires": self.prerequisites[i]}
                               for i in self.display_order],
                "forbidden_actions": self.forbidden, "completion_action": self.required_final,
                "turn_budget": self.max_steps}


@dataclass
class WorkflowState:
    task: WorkflowTask
    completed: set[str] = field(default_factory=set)
    violations: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    done: bool = False
    finished: bool = False

    @property
    def success(self) -> bool:
        # Budget exhaustion is terminal but is never an implicit FINISH.
        return self.finished and len(self.completed) == len(self.task.milestones) and not self.violations

    @property
    def progress(self) -> float:
        return len(self.completed) / len(self.task.milestones)

    def public_state(self, view: str = "full", recent: int = 4) -> dict:
        events = self.events
        if view == "recent":
            events = events[-recent:]
        elif view == "ledger":
            # Preserve all actual DO/FINISH attempts, including forbidden and rejected actions.
            # This is syntactic evidence compression, NOT oracle progress/violation labels.
            events = [e for e in events if e["action"].startswith(("DO ", "FINISH "))]
        elif view != "full":
            raise ValueError(view)
        return {"task": self.task.public_spec(), "events": [dict(e) for e in events],
                "ended": self.done, "turns_used": len(self.events),
                "turns_remaining": self.task.max_steps - len(self.events),
                "history_view": view, "omitted_events": len(self.events) - len(events)}


class WorkflowEnv:
    def __init__(self, task: WorkflowTask):
        self.state = WorkflowState(task)
        self.requirements = dict(zip(task.milestones, task.prerequisites))

    def step(self, raw_action: str) -> dict:
        s = self.state
        if s.done:
            raise RuntimeError("Episode already ended")
        action = raw_action.strip()
        receipt = "inert"
        if "\n" not in action and "\r" not in action:
            if action == s.task.required_final:
                s.done = s.finished = True
                receipt = "finished"
            elif action in s.task.forbidden:
                s.violations.append(action)
                receipt = "applied"  # Do not hand the evaluator a 'violation' label.
            elif action.startswith("DO ") and action[3:] in self.requirements:
                name = action[3:]
                if name not in s.completed and set(self.requirements[name]) <= s.completed:
                    s.completed.add(name)
                    receipt = "applied"
                else:
                    receipt = "rejected"
        event = {"turn": len(s.events) + 1, "action": action, "receipt": receipt}
        s.events.append(event)
        if len(s.events) == s.task.max_steps:
            s.done = True  # The horizon is part of the task, not a collector truncation.
        return event

    def oracle(self) -> dict:
        s = self.state
        return {"success": float(s.success), "progress": s.progress,
                "violation": float(bool(s.violations)), "finished": s.finished,
                "steps": len(s.events), "terminal_reason": "finish" if s.finished else "budget"}
