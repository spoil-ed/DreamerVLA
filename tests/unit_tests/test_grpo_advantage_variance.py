import torch

from dreamervla.algorithms.ppo.grpo import _group_advantage


def test_varying_outcomes_give_nonzero_advantage():
    score = torch.tensor([1.0, 0.0, 1.0, 0.0])

    advantage = _group_advantage(score, group_size=4, eps=1e-6)

    assert advantage.abs().sum() > 0
    assert torch.isfinite(advantage).all()


def test_constant_outcomes_give_zero_advantage():
    for score in (torch.zeros(4), torch.ones(4)):
        advantage = _group_advantage(score, group_size=4, eps=1e-6)

        assert advantage.abs().max() < 1e-5
