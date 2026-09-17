import pytest
import torch

from jev_reward_model.config import ExperimentConfig
from jev_reward_model.ppo import Turn, add_gae, clipped_loss


def test_gae_lambda_one_matches_monte_carlo():
    turns = [Turn(torch.tensor([1]), torch.tensor([2]), old_value=v, reward=r)
             for v, r in [(0.2, 0), (0.4, 0), (0.8, 1)]]
    add_gae(turns, gamma=0.9, lam=1.0)
    assert [t.return_ for t in turns] == pytest.approx([0.81, 0.9, 1.0])


def test_ppo_joint_logprob_ratio_and_clipping():
    scalar = lambda x: torch.tensor([x], dtype=torch.float32)
    cfg = ExperimentConfig(value_coef=0.0)
    new = scalar(-3.0).requires_grad_()
    loss, stats = clipped_loss(new, scalar(0), scalar(-3.0), scalar(0), scalar(1), scalar(0), cfg)
    assert loss.item() == pytest.approx(-1)
    assert stats["approx_kl"].item() == pytest.approx(0)
    loss.backward()
    assert new.grad.item() < 0
    loss, _ = clipped_loss(scalar(-2.0), scalar(0), scalar(-3.0), scalar(0), scalar(1), scalar(0), cfg)
    assert loss.item() == pytest.approx(-1.2)


def test_response_alignment_temperature_and_prefix_only_value():
    """Tiny causal stand-in: checks the production scoring path without HF weights."""
    from types import SimpleNamespace
    from torch import nn
    from jev_reward_model.policy import Actor

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(9, 4)
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=self.embed(input_ids).cumsum(dim=1))

    torch.manual_seed(42)
    actor = Actor.__new__(Actor)
    actor.cfg = ExperimentConfig(temperature=0.7)
    actor.device = torch.device("cpu")
    core = SimpleNamespace(model=Backbone(), lm_head=nn.Linear(4, 9, bias=False))
    actor.model = SimpleNamespace(get_base_model=lambda: core)
    actor.value_head = nn.Linear(4, 1)
    prompt, action = torch.tensor([1, 2]), torch.tensor([3, 4, 5])
    lp, value = actor.score(prompt, action)
    hidden = core.model(torch.tensor([[1, 2, 3, 4]])).last_hidden_state
    expected = -torch.nn.functional.cross_entropy(core.lm_head(hidden[0, 1:]) / 0.7,
                                                   action, reduction="sum")
    assert lp.item() == pytest.approx(expected.item())
    changed_lp, changed_value = actor.score(prompt, torch.tensor([7, 8]))
    assert value.item() == pytest.approx(changed_value.item())
    assert lp.item() != changed_lp.item()
    (-lp + value.square()).sum().backward()
    assert core.model.embed.weight.grad is not None
