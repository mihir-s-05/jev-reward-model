"""CPU-only PPO algebra. Skipped when the optional torch dependency is absent."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from jev_reward.config import PPOConfig
from jev_reward.policy import Turn
from jev_reward.ppo import clipped_losses, generalized_advantages, prepare_episode


def test_gae_matches_complete_returns():
    rewards = np.array([0., 0., 1.])
    values = np.array([.2, .4, .6])
    adv, returns = generalized_advantages(rewards, values, np.array([1., .9, 0.]), np.ones(3))
    assert returns == pytest.approx([.9, .9, 1.])
    assert adv == pytest.approx(returns - values)


def test_only_tool_boundaries_discount():
    turns = [Turn([9], [1, 2], "{}", [-1., -1.], [0., 0.], [-1., -1.]),
             Turn([9, 8], [3, 4, 5], "{}", [-1.] * 3, [0.] * 3, [-1.] * 3)]
    cfg = PPOConfig(gamma=.5, gae_lambda=1., kl_coefficient=0.)
    prepare_episode(turns, [0., 1.], cfg)
    assert turns[0].returns == pytest.approx([.5, .5])
    assert turns[1].returns == pytest.approx([1., 1., 1.])


def test_ppo_ratio_one_and_gradient():
    logs = torch.tensor([-1., -2.], requires_grad=True)
    values = torch.zeros(2, requires_grad=True)
    old = logs.detach().clone()
    advantages = torch.tensor([1., -1.])
    loss, metrics = clipped_losses(logs, values, old, torch.zeros(2), advantages,
                                  torch.zeros(2), PPOConfig())
    assert metrics["approx_kl_sum"] == pytest.approx(0.)
    loss.backward()
    assert logs.grad.tolist() == pytest.approx([-1., 1.])
