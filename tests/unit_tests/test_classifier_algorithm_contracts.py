from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from dreamervla.algorithms.critic.latent_success_classifier import (
    LatentSuccessClassifier,
    LatentSuccessClassifierConfig,
)
from dreamervla.algorithms.tdmpc_mpc import (
    TDMPCMPCConfig,
    TDMPCMPCPlanner,
    _repeat_latent,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"latent_dim": 0, "head_type": "linear"}, "latent_dim"),
        ({"token_dim": 0, "head_type": "spatial_tf"}, "token_dim"),
        ({"window": 0, "head_type": "linear"}, "window"),
        ({"hidden_dim": 0, "head_type": "mlp2"}, "hidden_dim"),
        ({"output_dim": 0, "head_type": "linear"}, "output_dim"),
        ({"proprio_dim": -1, "head_type": "linear"}, "proprio_dim"),
        ({"lang_dim": -1, "head_type": "linear"}, "lang_dim"),
        ({"num_proprio_repeat": 0, "head_type": "linear"}, "num_proprio_repeat"),
        ({"num_lang_repeat": 0, "head_type": "linear"}, "num_lang_repeat"),
        (
            {
                "latent_dim": 4,
                "hidden_dim": 4,
                "num_layers": 0,
                "num_heads": 1,
                "head_type": "transformer",
            },
            "num_layers",
        ),
        (
            {
                "latent_dim": 4,
                "hidden_dim": 4,
                "num_layers": 1,
                "num_heads": 0,
                "head_type": "transformer",
            },
            "num_heads",
        ),
        (
            {
                "latent_dim": 4,
                "hidden_dim": 5,
                "num_layers": 1,
                "num_heads": 2,
                "head_type": "transformer",
            },
            "divisible",
        ),
        (
            {
                "latent_dim": 4,
                "hidden_dim": 4,
                "num_layers": 1,
                "num_heads": 1,
                "mlp_ratio": 0.0,
                "head_type": "transformer",
            },
            "mlp_ratio",
        ),
        ({"latent_dim": 4, "dropout": math.nan, "head_type": "linear"}, "dropout"),
        ({"latent_dim": 4, "dropout": 1.0, "head_type": "linear"}, "dropout"),
    ],
)
def test_success_classifier_rejects_invalid_model_geometry(
    kwargs: dict[str, object], message: str
) -> None:
    cfg = LatentSuccessClassifierConfig(**kwargs)

    with pytest.raises(ValueError, match=message):
        LatentSuccessClassifier(cfg)


def _tiny_linear_classifier(window: int = 2) -> LatentSuccessClassifier:
    return LatentSuccessClassifier(
        LatentSuccessClassifierConfig(
            latent_dim=3,
            token_dim=3,
            token_count=1,
            window=window,
            hidden_dim=4,
            num_layers=1,
            num_heads=1,
            head_type="linear",
            granularity="action",
        )
    )


def test_success_classifier_forward_requires_window_tensor_rank() -> None:
    classifier = _tiny_linear_classifier()

    with pytest.raises(ValueError, match=r"\[B,W"):
        classifier(torch.zeros(2, 3))


@pytest.mark.parametrize(("stride", "min_steps"), [(0, 0), (-1, 0), (1, -1)])
def test_success_classifier_scan_rejects_invalid_scan_geometry(stride: int, min_steps: int) -> None:
    classifier = _tiny_linear_classifier()

    with pytest.raises(ValueError, match="stride|min_steps"):
        classifier.predict_success(
            torch.zeros(1, 3, 3),
            threshold=0.5,
            stride=stride,
            min_steps=min_steps,
        )


def test_success_classifier_scan_requires_at_least_one_window() -> None:
    classifier = _tiny_linear_classifier(window=3)

    with pytest.raises(ValueError, match="shorter than classifier window"):
        classifier.predict_success(torch.zeros(1, 2, 3), threshold=0.5)


def test_success_classifier_rejects_conditioning_batch_mismatch() -> None:
    classifier = LatentSuccessClassifier(
        LatentSuccessClassifierConfig(
            latent_dim=5,
            token_dim=4,
            token_count=1,
            window=2,
            hidden_dim=4,
            num_layers=1,
            num_heads=1,
            head_type="linear",
            granularity="action",
            proprio_dim=2,
            proprio_emb_dim=1,
        )
    )

    with pytest.raises(ValueError, match="proprio batch"):
        classifier(
            torch.zeros(2, 2, 4),
            proprio=torch.zeros(1, 2, 2),
        )


@pytest.mark.parametrize(
    "task_ids",
    [torch.tensor([0]), torch.tensor([0.0, 1.0]), torch.tensor([0, 2])],
)
def test_success_classifier_rejects_invalid_task_ids(task_ids: torch.Tensor) -> None:
    classifier = LatentSuccessClassifier(
        LatentSuccessClassifierConfig(
            latent_dim=3,
            token_dim=3,
            token_count=1,
            window=2,
            hidden_dim=4,
            num_layers=1,
            num_heads=1,
            head_type="linear",
            granularity="action",
            task_conditioning={"enabled": True, "num_tasks": 2, "embedding_dim": 3},
        )
    )

    with pytest.raises(ValueError, match="task_ids"):
        classifier(torch.zeros(2, 2, 3), task_ids=task_ids)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"terminal_value_scale": math.nan}, "terminal_value_scale"),
        ({"reward_scale": math.inf}, "reward_scale"),
        ({"value_mode": "mystery"}, "value_mode"),
    ],
)
def test_tdmpc_config_rejects_invalid_value_contracts(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TDMPCMPCConfig(**kwargs)


class _NaNRewardWorldModel(nn.Module):
    def forward(self, batch: dict[str, object]) -> torch.Tensor:
        mode = batch["mode"]
        latent = batch["latent"]
        assert isinstance(latent, torch.Tensor)
        if mode == "predict_next":
            return latent
        if mode == "reward":
            return torch.full((latent.shape[0], 1), math.nan, device=latent.device)
        raise AssertionError(f"unexpected mode {mode!r}")


class _UnusedPolicy(nn.Module):
    def forward(self, batch: dict[str, object]):
        raise AssertionError("policy trajectories are disabled in this test")


def test_tdmpc_planner_rejects_nonfinite_objective_values() -> None:
    planner = TDMPCMPCPlanner(
        TDMPCMPCConfig(
            horizon=1,
            iterations=1,
            num_samples=2,
            num_elites=1,
            num_pi_trajs=0,
            action_dim=1,
        )
    )

    with pytest.raises(ValueError, match="non-finite objective"):
        planner.plan(
            policy=_UnusedPolicy(),
            world_model=_NaNRewardWorldModel(),
            latent=torch.zeros(1, 2),
            device=torch.device("cpu"),
        )


class _LinearRewardWorldModel(nn.Module):
    def forward(self, batch: dict[str, object]) -> torch.Tensor:
        mode = batch["mode"]
        latent = batch["latent"]
        assert isinstance(latent, torch.Tensor)
        if mode == "predict_next":
            actions = batch["actions"]
            assert isinstance(actions, torch.Tensor)
            return latent + actions[:, :1]
        if mode == "reward":
            return latent
        if mode == "actor_input":
            return latent
        raise AssertionError(f"unexpected mode {mode!r}")


def test_tdmpc_planner_is_seeded_finite_and_action_bounded() -> None:
    cfg = TDMPCMPCConfig(
        horizon=2,
        iterations=3,
        num_samples=64,
        num_elites=8,
        num_pi_trajs=0,
        action_dim=1,
        execute_steps=1,
        seed=123,
    )
    kwargs = {
        "policy": _UnusedPolicy(),
        "world_model": _LinearRewardWorldModel(),
        "latent": torch.zeros(1, 1),
        "device": torch.device("cpu"),
    }

    first = TDMPCMPCPlanner(cfg).plan(**kwargs)
    second = TDMPCMPCPlanner(cfg).plan(**kwargs)

    assert first.raw_actions.shape == (1, 1)
    assert torch.isfinite(first.raw_actions).all()
    assert torch.all(first.raw_actions.abs() <= 1.0)
    torch.testing.assert_close(first.raw_actions, second.raw_actions)
    torch.testing.assert_close(first.best_value, second.best_value)


def test_tdmpc_repeat_latent_rejects_nonpositive_count() -> None:
    with pytest.raises(ValueError, match="repeats"):
        _repeat_latent(torch.zeros(1, 2), 0)


class _NarrowPolicy(nn.Module):
    def forward(self, batch: dict[str, object]):
        hidden = batch["hidden"]
        assert isinstance(hidden, torch.Tensor)
        action = torch.zeros(hidden.shape[0], 1, device=hidden.device)
        return action, torch.zeros(hidden.shape[0]), {}


def test_tdmpc_policy_trajectory_rejects_too_narrow_actions() -> None:
    planner = TDMPCMPCPlanner(
        TDMPCMPCConfig(
            horizon=1,
            iterations=1,
            num_samples=2,
            num_elites=1,
            num_pi_trajs=1,
            action_dim=2,
        )
    )

    with pytest.raises(ValueError, match="action_dim"):
        planner.plan(
            policy=_NarrowPolicy(),
            world_model=_LinearRewardWorldModel(),
            latent=torch.zeros(1, 2),
            device=torch.device("cpu"),
        )
