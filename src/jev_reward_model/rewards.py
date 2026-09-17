from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .env import WorkflowState
from .jev import JevClient


class RewardSource(Protocol):
    def reset(self, state: WorkflowState) -> None: ...
    def transition(self, before: WorkflowState, after: WorkflowState) -> float: ...
    def terminal(self, state: WorkflowState) -> float: ...


@dataclass
class GroundedReward:
    def reset(self, state: WorkflowState) -> None:
        pass

    def transition(self, before: WorkflowState, after: WorkflowState) -> float:
        return 0.0

    def terminal(self, state: WorkflowState) -> float:
        return 1.0 if state.success else 0.0


class JevTerminalReward:
    def __init__(self, client: JevClient):
        self.client = client

    def reset(self, state: WorkflowState) -> None:
        pass

    def transition(self, before: WorkflowState, after: WorkflowState) -> float:
        return 0.0

    def terminal(self, state: WorkflowState) -> float:
        return self.client.terminal_success(state.public_state())


class JevPotentialShapingReward:
    """Grounded terminal reward plus gamma-consistent Jev potential shaping."""

    def __init__(self, client: JevClient, gamma: float, alpha: float):
        self.client, self.gamma, self.alpha = client, gamma, alpha
        self._phi = 0.0

    def reset(self, state: WorkflowState) -> None:
        self._phi = self.client.progress(state.public_state())

    def transition(self, before: WorkflowState, after: WorkflowState) -> float:
        next_phi = 0.0 if after.done else self.client.progress(after.public_state())
        shaped = self.alpha * (self.gamma * next_phi - self._phi)
        self._phi = next_phi
        return shaped

    def terminal(self, state: WorkflowState) -> float:
        return 1.0 if state.success else 0.0


class QwenJudgeReward:
    """Local alternative-judge baseline. The callable returns P(success|trajectory)."""

    def __init__(self, judge_fn):
        self.judge_fn = judge_fn

    def reset(self, state: WorkflowState) -> None:
        pass

    def transition(self, before: WorkflowState, after: WorkflowState) -> float:
        return 0.0

    def terminal(self, state: WorkflowState) -> float:
        return float(self.judge_fn(state.public_state()))
