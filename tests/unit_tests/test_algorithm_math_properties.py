from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf

from dreamervla.algorithms.dreamervla import (
    _actor_action_to_env_scale,
    _flatten_strided_steps,
    compute_lambda_returns,
    compute_replay_lambda_returns,
    normalize_returns_for_actor_critic,
)


def _reference_lambda_return(
    rewards: torch.Tensor,
    live: torch.Tensor,
    trace: torch.Tensor,
    boot: torch.Tensor,
) -> torch.Tensor:
    result = torch.empty_like(live)
    carry = boot[:, -1]
    for step in range(live.shape[1] - 1, -1, -1):
        carry = (
            rewards[:, step + 1]
            + live[:, step] * (1.0 - trace[:, step]) * boot[:, step + 1]
            + live[:, step] * trace[:, step] * carry
        )
        result[:, step] = carry
    return result


@pytest.mark.parametrize("disc", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("lam", [0.0, 0.4, 1.0])
def test_imagined_lambda_return_matches_explicit_recurrence(disc: float, lam: float) -> None:
    generator = torch.Generator().manual_seed(1000 + int(100 * disc) + int(10 * lam))
    rewards = torch.randn(3, 6, generator=generator)
    continues = torch.rand(3, 6, generator=generator)
    boot = torch.randn(3, 6, generator=generator)
    live = continues[:, 1:] * disc
    trace = torch.full_like(live, lam)

    actual = compute_lambda_returns(rewards, continues, boot, disc=disc, lam=lam)
    expected = _reference_lambda_return(rewards, live, trace, boot)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("disc", [0.0, 0.9, 1.0])
@pytest.mark.parametrize("lam", [0.0, 0.7, 1.0])
def test_replay_lambda_return_matches_explicit_recurrence(disc: float, lam: float) -> None:
    generator = torch.Generator().manual_seed(2000 + int(100 * disc) + int(10 * lam))
    rewards = torch.randn(2, 7, generator=generator)
    boot = torch.randn(2, 7, generator=generator)
    terminal = torch.randint(0, 2, (2, 7), generator=generator).float()
    last = torch.randint(0, 2, (2, 7), generator=generator).float()
    live = (1.0 - terminal[:, 1:]) * disc
    trace = (1.0 - last[:, 1:]) * lam

    actual = compute_replay_lambda_returns(last, terminal, rewards, boot, disc, lam)
    expected = _reference_lambda_return(rewards, live, trace, boot)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("shape", "disc", "lam", "message"),
    [
        ((3,), 0.9, 0.5, r"\[B,H\+1\]"),
        ((2, 1), 0.9, 0.5, "at least two time"),
        ((2, 3), -0.1, 0.5, "disc"),
        ((2, 3), 0.9, 1.1, "lam"),
        ((2, 3), math.nan, 0.5, "disc"),
    ],
)
def test_imagined_lambda_return_rejects_invalid_contracts(
    shape: tuple[int, ...], disc: float, lam: float, message: str
) -> None:
    tensor = torch.zeros(shape)

    with pytest.raises(ValueError, match=message):
        compute_lambda_returns(tensor, tensor, tensor, disc, lam)


def test_action_scale_maps_policy_endpoints_to_configured_bounds() -> None:
    cfg = OmegaConf.create(
        {
            "rssm_action_low": [-2.0, 1.0, 4.0],
            "rssm_action_high": [2.0, 3.0, 10.0],
        }
    )
    action = torch.tensor([[-1.0, 0.0, 1.0]])

    mapped = _actor_action_to_env_scale(action, cfg)

    torch.testing.assert_close(mapped, torch.tensor([[-2.0, 2.0, 10.0]]))


def test_action_scale_rejects_action_width_mismatch() -> None:
    with pytest.raises(ValueError, match="action width"):
        _actor_action_to_env_scale(torch.zeros(2, 3), OmegaConf.create({}))


@pytest.mark.parametrize(
    ("action", "cfg"),
    [
        (
            torch.tensor([[math.nan]]),
            {"rssm_action_low": [-1.0], "rssm_action_high": [1.0]},
        ),
        (
            torch.zeros(1, 1),
            {"rssm_action_low": [0.0], "rssm_action_high": [0.0]},
        ),
    ],
)
def test_action_scale_rejects_nonfinite_or_unordered_geometry(
    action: torch.Tensor, cfg: dict
) -> None:
    with pytest.raises(ValueError, match="finite|smaller"):
        _actor_action_to_env_scale(action, OmegaConf.create(cfg))


@pytest.mark.parametrize(
    "cfg",
    [
        {"return_normalization": {"mode": "dreamerv3", "low": -0.1}},
        {"return_normalization": {"mode": "dreamerv3", "high": 1.1}},
        {
            "return_normalization": {
                "mode": "dreamerv3",
                "low": 0.9,
                "high": 0.1,
            }
        },
        {"return_normalization": {"mode": "dreamerv3", "eps": 0.0}},
    ],
)
def test_return_normalization_rejects_invalid_quantile_geometry(cfg: dict) -> None:
    with pytest.raises(ValueError, match="return_normalization"):
        normalize_returns_for_actor_critic(
            torch.ones(2, 2),
            torch.zeros(2, 2),
            OmegaConf.create(cfg),
        )


@pytest.mark.parametrize("which", ["returns", "values"])
def test_return_normalization_rejects_nonfinite_inputs(which: str) -> None:
    returns = torch.ones(2, 2)
    values = torch.zeros(2, 2)
    (returns if which == "returns" else values)[0, 0] = math.nan

    with pytest.raises(ValueError, match="finite"):
        normalize_returns_for_actor_critic(
            returns,
            values,
            OmegaConf.create({"return_normalization": {"mode": "dreamerv3"}}),
        )


def test_flatten_strided_steps_selects_requested_evenly_spaced_states() -> None:
    latent = torch.arange(10).reshape(1, 10, 1)

    selected = _flatten_strided_steps(latent, num_starts=3, min_start=2)

    assert selected[:, 0].tolist() == [2, 6, 9]


@pytest.mark.parametrize(("num_starts", "min_start"), [(0, 0), (-1, 0), (1, -1), (1, 4)])
def test_flatten_strided_steps_rejects_invalid_selection_geometry(
    num_starts: int, min_start: int
) -> None:
    with pytest.raises(ValueError, match="num_starts|min_start"):
        _flatten_strided_steps(torch.zeros(1, 4, 2), num_starts, min_start)
