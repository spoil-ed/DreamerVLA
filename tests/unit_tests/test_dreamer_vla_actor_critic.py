import torch
from omegaconf import OmegaConf

from dreamer_vla.algorithms.dreamer_vla import (
    _actor_action_for_world_model,
    _actor_action_to_env_scale,
    compute_replay_lambda_returns,
    normalize_returns_for_actor_critic,
)
from dreamer_vla.models.critic.twohot_critic import (
    ReturnPercentileTracker,
    TwohotCritic,
    symexp,
)


def test_twohot_critic_predicts_softmax_bin_expectation():
    critic = TwohotCritic(
        hidden_dim=2, num_bins=3, bin_min=-1.0, bin_max=1.0, critic_layers=0
    )
    final = next(
        module for module in critic.backbone if isinstance(module, torch.nn.Linear)
    )
    with torch.no_grad():
        final.weight.zero_()
        final.bias.copy_(torch.tensor([-1.0, 0.0, 1.0]))

    hidden = torch.zeros(1, 2)
    probs = torch.softmax(torch.tensor([-1.0, 0.0, 1.0]), dim=0)
    expected_symlog = (probs * torch.tensor([-1.0, 0.0, 1.0])).sum()

    pred = critic({"mode": "value", "hidden": hidden})

    assert torch.allclose(pred, symexp(expected_symlog).reshape(1))


def test_dreamerv3_return_normalization_keeps_value_targets_raw():
    returns = torch.tensor([[0.0, 10.0], [2.0, 8.0]])
    values = torch.tensor([[1.0, 3.0], [4.0, 6.0]])
    cfg = OmegaConf.create(
        {
            "return_normalization": {
                "mode": "dreamerv3",
                "low": 0.0,
                "high": 1.0,
                "eps": 1.0e-6,
            }
        }
    )

    out = normalize_returns_for_actor_critic(returns, values, cfg)

    assert torch.equal(out.returns, returns)
    assert torch.equal(out.values, values)
    assert out.enabled
    assert out.low == 0.0
    assert out.high == 10.0
    assert out.scale == 10.0


def test_return_tracker_reports_dreamerv3_offset_and_scale():
    tracker = ReturnPercentileTracker(decay=0.0, low=0.0, high=1.0)
    low, high = tracker.update(torch.tensor([[2.0, 5.0], [8.0, 12.0]]))

    assert low == 2.0
    assert high == 12.0
    assert tracker.offset() == 2.0
    assert tracker.scale() == 10.0
    assert tracker.stats() == (2.0, 10.0)


def test_replay_lambda_returns_respect_terminal_and_last_flags():
    rewards = torch.tensor([[0.0, 1.0, 2.0]])
    values = torch.zeros_like(rewards)
    boot = torch.tensor([[5.0, 7.0, 11.0]])
    terminal = torch.tensor([[0.0, 1.0, 0.0]])
    last = torch.zeros_like(rewards)

    returns = compute_replay_lambda_returns(
        last=last,
        terminal=terminal,
        rewards=rewards,
        values=values,
        boot=boot,
        disc=0.9,
        lam=0.5,
    )

    assert torch.allclose(returns, torch.tensor([[1.0, 11.9]]))


def test_actor_actions_default_to_libero_env_scale_for_world_model():
    raw_action = torch.tensor([[-1.0, 0.0, 1.0, 0.0, 0.5, -0.5, 1.0]])

    rssm_action = _actor_action_for_world_model(raw_action, OmegaConf.create({}))

    expected = torch.tensor(
        [
            [
                -0.9375,
                0.0,
                0.9375,
                0.053035713732242584,
                0.1875,
                -0.17946428060531616,
                1.0,
            ]
        ]
    )
    assert torch.allclose(rssm_action, expected, atol=1.0e-6)


def test_actor_env_scale_helper_can_report_unclipped_and_clipped_drift():
    raw_action = torch.tensor([[2.0, -2.0, 0.0, 1.0, -1.0, 0.0, 3.0]])
    cfg = OmegaConf.create({"rssm_action_clip": True})

    unclipped = _actor_action_to_env_scale(raw_action, cfg, clip=False)
    clipped = _actor_action_to_env_scale(raw_action, cfg, clip=True)

    assert unclipped[0, 0] > 0.9375
    assert unclipped[0, 1] < -0.9375
    assert unclipped[0, 6] > 1.0
    assert torch.all(
        clipped
        <= torch.tensor([[0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0]])
    )
    assert torch.all(
        clipped
        >= torch.tensor(
            [[-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0]]
        )
    )
