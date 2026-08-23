from __future__ import annotations

import math

import pytest
import torch

from dreamervla.algorithms.critic.twohot_critic import (
    ReturnPercentileTracker,
    RMSNorm,
    TwohotCritic,
)
from dreamervla.algorithms.ppo.grpo import (
    _group_advantage,
    _ppo_clip_term,
    _ppo_ratio,
    group_variance_mask,
)
from dreamervla.algorithms.reward import get_reward_model
from dreamervla.algorithms.tdmpc_mpc import TDMPCMPCConfig


def test_group_advantage_rejects_empty_scores() -> None:
    with pytest.raises(ValueError, match="score must be non-empty"):
        _group_advantage(torch.empty(0), group_size=1, eps=1.0e-6)


@pytest.mark.parametrize("eps", [0.0, -1.0, math.inf, math.nan])
def test_group_advantage_requires_positive_finite_epsilon(eps: float) -> None:
    score = torch.tensor([0.0, 1.0])

    with pytest.raises(ValueError, match="eps must be finite and > 0"):
        _group_advantage(score, group_size=2, eps=eps)


@pytest.mark.parametrize("eps", [-1.0, math.inf, math.nan])
def test_group_variance_mask_requires_non_negative_finite_epsilon(eps: float) -> None:
    score = torch.tensor([0.0, 1.0])

    with pytest.raises(ValueError, match="eps must be finite and >= 0"):
        group_variance_mask(score, group_size=2, eps=eps)


def test_group_advantage_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="score must contain only finite values"):
        _group_advantage(torch.tensor([0.0, math.nan]), group_size=2, eps=1.0e-6)


@pytest.mark.parametrize("clip_log_ratio", [0.0, -1.0, math.inf, math.nan])
def test_ppo_ratio_rejects_invalid_log_ratio_clip(clip_log_ratio: float) -> None:
    with pytest.raises(ValueError, match="clip_log_ratio must be finite and > 0"):
        _ppo_ratio(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            clip_log_ratio=clip_log_ratio,
        )


def test_ppo_ratio_rejects_misaligned_or_nonfinite_log_probs() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        _ppo_ratio(torch.zeros(2), torch.zeros(1))
    with pytest.raises(ValueError, match="finite"):
        _ppo_ratio(torch.tensor([math.nan]), torch.zeros(1))


@pytest.mark.parametrize(
    ("clip_low", "clip_high", "clip_ratio_c", "message"),
    [
        (-0.1, 0.2, None, "clip_low"),
        (1.0, 0.2, None, "clip_low"),
        (0.2, -0.1, None, "clip_high"),
        (0.2, 0.3, 1.0, "clip_ratio_c"),
    ],
)
def test_ppo_clip_term_rejects_invalid_geometry(
    clip_low: float,
    clip_high: float,
    clip_ratio_c: float | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ppo_clip_term(
            torch.ones(2),
            torch.ones(2),
            clip_low,
            clip_high,
            clip_ratio_c=clip_ratio_c,
        )


def test_ppo_clip_term_rejects_misaligned_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        _ppo_clip_term(torch.ones(2), torch.ones(1), 0.2, 0.2)
    with pytest.raises(ValueError, match="finite"):
        _ppo_clip_term(torch.tensor([math.inf]), torch.ones(1), 0.2, 0.2)


@pytest.mark.parametrize("reward_name", ["sparse_outcome", "probability_outcome"])
def test_reward_models_require_positive_geometry(reward_name: str) -> None:
    model = get_reward_model(reward_name)

    with pytest.raises(ValueError, match="batch must be > 0"):
        model.build_reward(
            batch=0,
            max_steps=4,
            chunk_size=1,
            finish_step=torch.empty(0),
            complete=torch.empty(0, dtype=torch.bool),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="max_steps must be > 0"):
        model.build_reward(
            batch=1,
            max_steps=0,
            chunk_size=1,
            finish_step=torch.tensor([0]),
            complete=torch.tensor([False]),
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("reward_name", ["sparse_outcome", "probability_outcome"])
def test_reward_models_require_batch_aligned_outcomes(reward_name: str) -> None:
    model = get_reward_model(reward_name)

    with pytest.raises(ValueError, match=r"finish_step.*batch=2"):
        model.build_reward(
            batch=2,
            max_steps=4,
            chunk_size=1,
            finish_step=torch.tensor([0]),
            complete=torch.tensor([False, True]),
            device=torch.device("cpu"),
        )


def test_probability_reward_rejects_non_finite_scores() -> None:
    model = get_reward_model("probability_outcome")

    with pytest.raises(ValueError, match="score must contain only finite values"):
        model.build_reward(
            batch=1,
            max_steps=4,
            chunk_size=1,
            finish_step=torch.tensor([0]),
            complete=torch.tensor([False]),
            score=torch.tensor([math.nan]),
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("finish_step", torch.tensor([0.0]), "integer"),
        ("finish_step", torch.tensor([[0]]), r"\[B\]"),
        ("complete", torch.tensor([1]), "bool"),
        ("score_step", torch.tensor([0.5]), "integer"),
    ],
)
def test_reward_models_reject_malformed_outcome_types(
    field: str, value: torch.Tensor, message: str
) -> None:
    model = get_reward_model("probability_outcome")
    kwargs = {
        "batch": 1,
        "max_steps": 4,
        "chunk_size": 1,
        "finish_step": torch.tensor([0]),
        "complete": torch.tensor([False]),
        "score": torch.tensor([0.5]),
        "score_step": torch.tensor([0]),
        "device": torch.device("cpu"),
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        model.build_reward(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_bins": 1}, "num_bins"),
        ({"bin_min": 1.0, "bin_max": 1.0}, "bin_min"),
        ({"critic_layers": -1}, "critic_layers"),
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"critic_hidden_dim": 0}, "critic_hidden_dim"),
        ({"outscale": math.nan}, "outscale"),
    ],
)
def test_twohot_critic_rejects_invalid_geometry(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TwohotCritic(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"decay": -0.1}, "decay"),
        ({"decay": 1.1}, "decay"),
        ({"low": -0.1}, "low"),
        ({"high": 1.1}, "high"),
        ({"low": 0.8, "high": 0.2}, "low"),
    ],
)
def test_return_percentile_tracker_rejects_invalid_geometry(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ReturnPercentileTracker(**kwargs)


def test_return_percentile_tracker_rejects_empty_and_non_finite_samples() -> None:
    tracker = ReturnPercentileTracker()

    with pytest.raises(ValueError, match="returns must be non-empty"):
        tracker.update(torch.empty(0))
    with pytest.raises(ValueError, match="returns must contain only finite values"):
        tracker.update(torch.tensor([0.0, math.inf]))


@pytest.mark.parametrize(("dim", "eps"), [(0, 1.0e-6), (2, 0.0), (2, math.nan)])
def test_rms_norm_rejects_invalid_geometry(dim: int, eps: float) -> None:
    with pytest.raises(ValueError, match="dim|eps"):
        RMSNorm(dim, eps=eps)


def test_return_percentile_tracker_rejects_invalid_checkpoint_state() -> None:
    tracker = ReturnPercentileTracker()

    with pytest.raises(TypeError, match="mapping"):
        tracker.load_state_dict([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="low_ema"):
        tracker.load_state_dict({"low_ema": 2.0, "high_ema": 1.0})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"horizon": 0}, "horizon"),
        ({"num_samples": 0}, "num_samples"),
        ({"num_samples": 4, "num_elites": 5}, "num_elites"),
        ({"num_samples": 4, "num_elites": 2, "num_pi_trajs": 5}, "num_pi_trajs"),
        ({"min_std": 2.0, "max_std": 1.0}, "min_std"),
        ({"horizon": 2, "execute_steps": 3}, "execute_steps"),
    ],
)
def test_tdmpc_config_rejects_invalid_geometry(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TDMPCMPCConfig(**kwargs)


@pytest.mark.parametrize(
    "values",
    [
        torch.tensor([-1.0e6, -1.0, 0.0, 1.0, 1.0e6]),
        torch.linspace(-100.0, 100.0, 257),
    ],
)
def test_twohot_targets_are_finite_probability_distributions(values: torch.Tensor) -> None:
    critic = TwohotCritic(num_bins=255, bin_min=-20.0, bin_max=20.0)

    targets = critic.twohot_targets(values)

    assert torch.isfinite(targets).all()
    assert torch.all(targets >= 0.0)
    assert torch.allclose(targets.sum(dim=-1), torch.ones_like(values))


def test_twohot_targets_reject_nonfinite_values() -> None:
    critic = TwohotCritic(num_bins=5)

    with pytest.raises(ValueError, match="values must contain only finite"):
        critic.twohot_targets(torch.tensor([0.0, math.nan]))


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_group_advantage_is_zero_mean_per_nonconstant_group(group_size: int) -> None:
    score = torch.arange(group_size * 3, dtype=torch.float32)

    advantage = _group_advantage(score, group_size=group_size, eps=1.0e-6)
    groups = advantage.reshape(-1, group_size)

    assert torch.allclose(groups.mean(dim=1), torch.zeros(3), atol=1.0e-6)
    assert torch.allclose(groups.std(dim=1, unbiased=False), torch.ones(3), atol=1.0e-6)
