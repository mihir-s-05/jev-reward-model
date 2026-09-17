import random

import pytest

from jev_reward_model.data import assert_disjoint, generate, make_task
from jev_reward_model.env import WorkflowEnv


def task():
    return make_task(random.Random(3), "train", 0, "chain", 4)


def test_terminal_requires_real_finish_and_irreversible_constraints():
    t = task()
    env = WorkflowEnv(t)
    for name in t.milestones:
        env.step("DO " + name)
    while not env.state.done:
        env.step("WAIT")
    assert not env.state.success  # Old scaffold incorrectly passed this at timeout.
    env = WorkflowEnv(t)
    env.step(t.forbidden[0])
    for name in t.milestones:
        env.step("DO " + name)
    env.step(t.required_final)
    assert env.state.finished and not env.state.success
    public = env.state.public_state()
    assert not {"success", "completed", "violations", "progress"} & public.keys()
    assert len(env.state.public_state("ledger")["events"]) == len(env.state.events)


def test_dependency_retry_and_full_output_semantics():
    t = task()
    env = WorkflowEnv(t)
    assert env.step("DO " + t.milestones[-1])["receipt"] == "rejected"
    assert env.step("DO " + t.milestones[0] + "\nextra text")["receipt"] == "inert"
    for name in t.milestones:
        env.step("DO " + name)
    env.step(t.required_final)
    assert env.state.success
    with pytest.raises(RuntimeError):
        env.step("WAIT")


def test_split_guard_and_heldout_graph_solvability():
    training = generate(3, 1, "train", ("chain",), (4,))
    with pytest.raises(ValueError):
        assert_disjoint(training, training)
    heldout = generate(4, 2, "test_ood", ("barrier", "overlapping_dependencies"), (4, 8))
    assert_disjoint(training, heldout)
    for t in heldout:
        env = WorkflowEnv(t)
        for name in t.milestones:
            env.step("DO " + name)
        env.step(t.required_final)
        assert env.state.success
