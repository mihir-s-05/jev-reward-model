from types import SimpleNamespace

import pytest

from jev_reward_model.config import ExperimentConfig
from jev_reward_model.rewards import assign_rewards, potential_rewards


def test_discounted_shaping_telescopes():
    gamma, alpha = 0.91, 0.5
    base, phi = [0, 0, 1], [0.2, 0.7, 0.5, 0.0]
    shaped = potential_rewards(base, phi, gamma, alpha)
    difference = sum(gamma ** t * (s - b) for t, (s, b) in enumerate(zip(shaped, base)))
    assert difference == pytest.approx(-alpha * phi[0])
    with pytest.raises(ValueError):
        potential_rewards(base, [0, 0, 0, 1], gamma, alpha)


def test_terminal_arm_cannot_fall_back_to_oracle():
    class Judge:
        def evaluate_many(self, states, kind):
            assert states == [{"public": "evidence"}]
            return [{"success": 0.2}]
    episode = SimpleNamespace(states=[{}, {"public": "evidence"}], turns=[None], oracle={"success": 1.0})
    assign_rewards([episode], ExperimentConfig(reward="jev_terminal"), Judge())
    assert episode.rewards == [0.2]
