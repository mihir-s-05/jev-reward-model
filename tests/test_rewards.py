from jev_reward_model.rewards import JevPotentialShapingReward


class FakeJev:
    def __init__(self): self.values = iter([0.25, 0.5])
    def progress(self, _): return next(self.values)


class State:
    done = False
    def public_state(self): return {}


def test_potential_shaping_formula():
    reward = JevPotentialShapingReward(FakeJev(), gamma=0.9, alpha=2.0)
    s = State(); reward.reset(s)
    got = reward.transition(s, s)
    assert abs(got - 2.0 * (0.9 * 0.5 - 0.25)) < 1e-9
