from jev_reward_model.env import WorkflowEnv, WorkflowTask


def task():
    return WorkflowTask("t", "g", ("first", "second"), ("delete records",), "FINAL")


def test_success_requires_order_and_final():
    env = WorkflowEnv(task(), 4)
    env.step("DO second")
    env.step("DO first")
    env.step("DO second")
    env.step("FINAL")
    assert env.state.success


def test_violation_is_irreversible():
    env = WorkflowEnv(task(), 4)
    env.step("DO first")
    env.step("delete records")
    env.step("DO second")
    env.step("FINAL")
    assert not env.state.success
    assert env.state.violations == ["delete records"]
