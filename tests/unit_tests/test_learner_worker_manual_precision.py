from __future__ import annotations

import contextlib
from typing import Any

import pytest
import torch

from dreamervla.workers.actor.learner_worker import LearnerWorker, _resolve_precision


def test_resolve_precision_defaults_to_fp32_no_autocast() -> None:
    precision = _resolve_precision({}, torch.device("cpu"))

    assert precision.name == "fp32"
    assert precision.autocast_dtype is None
    assert isinstance(precision.context(), contextlib.AbstractContextManager)


def test_resolve_precision_accepts_manual_bf16_on_cpu() -> None:
    precision = _resolve_precision({"precision": "bf16"}, torch.device("cpu"))

    assert precision.name == "bf16"
    assert precision.autocast_dtype is torch.bfloat16


def test_resolve_precision_rejects_auto() -> None:
    with pytest.raises(ValueError, match="precision"):
        _resolve_precision({"precision": "auto"}, torch.device("cpu"))


class _DirectReplay:
    def __init__(self) -> None:
        self.sample_calls = 0
        self.classifier_calls = 0

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        self.sample_calls += 1
        return {
            "obs_embedding": torch.ones(int(batch_size), 3, 4),
            "actions": torch.zeros(int(batch_size), 3, 7),
            "current_actions": torch.ones(int(batch_size), 3, 7),
            "rewards": torch.zeros(int(batch_size), 3),
            "dones": torch.zeros(int(batch_size), 3),
            "is_first": torch.zeros(int(batch_size), 3, dtype=torch.bool),
            "is_terminal": torch.zeros(int(batch_size), 3),
            "is_last": torch.zeros(int(batch_size), 3),
        }

    def sample_classifier_windows(
        self,
        batch_size: int,
        *,
        window: int,
        chunk_size: int,
        chunk_pool: str,
        early_neg_stride: int,
    ) -> dict[str, torch.Tensor]:
        del chunk_pool, early_neg_stride
        self.classifier_calls += 1
        return {
            "windows": torch.ones(int(batch_size), int(window), 4),
            "labels": torch.zeros(int(batch_size), dtype=torch.long),
        }

    def classifier_window_count(self, *, window: int, chunk_size: int) -> int:
        del window, chunk_size
        return 3


def _cotrain_model_cfg() -> dict[str, Any]:
    return {
        "policy": {
            "target": "dreamervla.workers.actor._test_models:TinySharedPolicy",
            "kwargs": {"hidden_dim": 4, "action_dim": 7},
        },
        "world_model": {
            "target": "dreamervla.workers.actor._test_models:TinyTrainableWorldModel",
            "kwargs": {"hidden_dim": 4},
        },
        "critic": {
            "target": "dreamervla.workers.actor._test_models:TinyValueCritic",
            "kwargs": {"hidden_dim": 4},
        },
        "classifier": {
            "target": "dreamervla.workers.actor._test_models:TinySuccessClassifier",
            "kwargs": {"hidden_dim": 4, "window": 3},
        },
    }


def _cotrain_train_cfg() -> dict[str, Any]:
    return {
        "mode": "dreamervla_cotrain",
        "device": "cpu",
        "batch_size": 2,
        "classifier_batch_size": 2,
        "classifier_early_neg_stride": 2,
        "classifier_threshold": 0.5,
        "lr": 0.01,
        "optim_cfg": {"grad_clip_norm": 1.0, "zero_grad_set_to_none": True},
        "algorithm_cfg": {
            "wmpo": {"chunk_size": 1, "episode_max_steps": 2},
            "ppo_rollouts_per_start": 1,
        },
        "syncer": {"store_name": "learner_worker_cotrain_store"},
    }


def test_dreamervla_cotrain_mode_routes_real_update_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    import dreamervla.workers.actor.learner_worker as mod

    calls: list[str] = []

    def fake_wm_step(**kwargs):
        calls.append("wm")
        assert kwargs["world_model"] is learner.world_model
        assert kwargs["optimizer"] is learner.world_model_optimizer
        assert "obs_embedding" in kwargs["batch"]
        return {"loss": 1.25}

    def fake_classifier_step(**kwargs):
        calls.append("classifier")
        assert kwargs["classifier"] is learner.classifier
        assert kwargs["optimizer"] is learner.classifier_optimizer
        assert kwargs["replay"].classifier_window_count(window=3, chunk_size=1) == 3
        return {"loss": 0.5, "acc": 1.0, "f1": 1.0}

    def fake_rl_step(**kwargs):
        calls.append("rl")
        assert kwargs["policy"] is learner.policy
        assert kwargs["chunk_world_model"] is learner.world_model
        assert kwargs["actor_optimizer"] is learner.policy_optimizer
        assert kwargs["classifier_threshold"] == 0.5
        assert set(kwargs["obs"]) >= {
            "obs_embedding",
            "actions",
            "rewards",
            "dones",
            "is_first",
            "is_terminal",
            "is_last",
        }
        return {"actor_loss": 0.75, "returns_mean": 0.25, "actor_grad_norm": 0.125}

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)
    monkeypatch.setattr(mod, "online_classifier_update_step", fake_classifier_step)
    monkeypatch.setattr(mod, "dino_wmpo_outcome_step", fake_rl_step)

    replay = _DirectReplay()
    learner = LearnerWorker(_cotrain_model_cfg(), {}, _cotrain_train_cfg(), replay)
    learner.init()

    assert learner.update("wm", 1) == {"wm/loss": 1.25}
    assert learner.update("classifier", 1) == {
        "cls/loss": 0.5,
        "cls/acc": 1.0,
        "cls/f1": 1.0,
    }
    assert learner.update("rl", 1) == {
        "rl/actor_loss": 0.75,
        "rl/returns_mean": 0.25,
        "rl/policy_grad_norm": 0.125,
    }

    metrics = learner.update("cotrain", 1)

    assert metrics["wm/loss"] == 1.25
    assert metrics["cls/loss"] == 0.5
    assert metrics["rl/actor_loss"] == 0.75
    assert calls == ["wm", "classifier", "rl", "wm", "classifier", "rl"]


def test_dreamervla_cotrain_mode_requires_components() -> None:
    replay = _DirectReplay()
    cfg = _cotrain_model_cfg()
    cfg.pop("world_model")
    learner = LearnerWorker(cfg, {}, _cotrain_train_cfg(), replay)

    with pytest.raises(ValueError, match="world_model"):
        learner.init()
