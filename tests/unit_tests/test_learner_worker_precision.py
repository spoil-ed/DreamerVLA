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


def test_learner_world_model_uses_configured_fp32_adamw_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dummy_syncer(monkeypatch)
    train_cfg = _wm_classifier_only_train_cfg()
    train_cfg.update(
        {
            "precision": "bf16",
            "param_precision": "fp32",
            "optimizers": {
                "world_model": {
                    "name": "adamw",
                    "lr": 1.0e-4,
                    "betas": [0.9, 0.999],
                    "eps": 1.0e-8,
                    "weight_decay": 0.0,
                }
            },
        }
    )
    learner = LearnerWorker(_wm_classifier_only_model_cfg(), {}, train_cfg, replay=None)

    learner.init()

    assert learner.world_model is not None
    assert all(p.dtype is torch.float32 for p in learner.world_model.parameters())
    assert isinstance(learner.world_model_optimizer, torch.optim.AdamW)
    parameter = next(learner.world_model.parameters())
    parameter.square().mean().backward()
    learner.world_model_optimizer.step()
    state = learner.world_model_optimizer.state[parameter]
    assert state["exp_avg"].dtype is torch.float32
    assert state["exp_avg_sq"].dtype is torch.float32


class _DirectReplay:
    def __init__(self) -> None:
        self.sample_calls = 0
        self.classifier_calls = 0

    def sample(
        self,
        batch_size: int,
        *,
        include_images: bool = True,
        staleness_threshold: int | None = None,
    ) -> dict[str, torch.Tensor]:
        del include_images, staleness_threshold
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
            "proprio": torch.full((int(batch_size), 3, 8), 0.5),
            "lang_emb": torch.full((int(batch_size), 6), 0.25),
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


class _ImageSensitiveReplay(_DirectReplay):
    def __init__(self) -> None:
        super().__init__()
        self.include_images_calls: list[bool] = []

    def sample(
        self,
        batch_size: int,
        *,
        include_images: bool = True,
        staleness_threshold: int | None = None,
    ) -> dict[str, torch.Tensor]:
        del staleness_threshold
        self.include_images_calls.append(bool(include_images))
        if include_images:
            raise AssertionError("latent learner updates must not request replay images")
        return super().sample(batch_size)


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
            "lumos": {"chunk_size": 1, "episode_max_steps": 2},
            "ppo_rollouts_per_start": 1,
        },
        "syncer": {"store_name": "learner_worker_cotrain_store"},
    }


def _patch_dummy_syncer(monkeypatch: pytest.MonkeyPatch) -> None:
    import dreamervla.workers.actor.learner_worker as mod

    class _DummySyncer:
        def __init__(self, **kwargs):
            del kwargs

        def push(self, *args, **kwargs):
            del args, kwargs

    monkeypatch.setattr(mod, "ObjectStoreWeightSyncer", _DummySyncer)


def test_dreamervla_cotrain_mode_routes_real_update_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    import dreamervla.workers.actor.learner_worker as mod

    calls: list[str] = []

    def fake_wm_step(**kwargs):
        calls.append("wm")
        assert kwargs["world_model"] is learner.world_model
        assert kwargs["optimizer"] is learner.world_model_optimizer
        assert "obs_embedding" in kwargs["batch"]
        assert kwargs["batch"]["proprio"].shape == (2, 3, 8)
        assert kwargs["batch"]["lang_emb"].shape == (2, 6)
        return {
            "loss": 1.25,
            "hidden_rec_loss": 0.25,
            "hidden_cosine_loss": 0.125,
        }

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
            "proprio",
            "lang_emb",
        }
        return {"actor_loss": 0.75, "returns_mean": 0.25, "actor_grad_norm": 0.125}

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)
    monkeypatch.setattr(mod, "online_classifier_update_step", fake_classifier_step)
    monkeypatch.setattr(mod, "dino_lumos_step", fake_rl_step)
    _patch_dummy_syncer(monkeypatch)

    replay = _DirectReplay()
    learner = LearnerWorker(_cotrain_model_cfg(), {}, _cotrain_train_cfg(), replay)
    learner.init()

    assert learner.update("wm", 1) == {
        "wm/loss": 1.25,
        "wm/hidden_rec_loss": 0.25,
        "wm/hidden_cosine_loss": 0.125,
    }
    assert learner.update("classifier", 1) == {
        "cls/loss": 0.5,
        "cls/acc": 1.0,
        "cls/f1": 1.0,
        "cls/skipped_single_class_batch": 0.0,
        "cls/updated": 1.0,
        "cls/updates": 1.0,
    }
    rl_metrics = learner.update("rl", 1)
    assert rl_metrics["rl/actor_loss"] == 0.75
    assert rl_metrics["rl/returns_mean"] == 0.25
    assert rl_metrics["rl/policy_grad_norm"] == 0.125
    assert rl_metrics["rl/skipped_no_signal"] == 0.0

    metrics = learner.update("cotrain", 1)

    assert metrics["wm/loss"] == 1.25
    assert metrics["wm/hidden_rec_loss"] == 0.25
    assert metrics["wm/hidden_cosine_loss"] == 0.125
    assert metrics["cls/loss"] == 0.5
    assert metrics["rl/actor_loss"] == 0.75
    assert calls == ["wm", "classifier", "rl", "wm", "classifier", "rl"]


def test_learner_state_dicts_include_classifier_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dummy_syncer(monkeypatch)
    train_cfg = _cotrain_train_cfg()
    train_cfg["classifier_threshold"] = 0.875
    learner = LearnerWorker(_cotrain_model_cfg(), {}, train_cfg, _DirectReplay())
    learner.init()

    state_dicts = learner.state_dicts()

    assert state_dicts["classifier_threshold"] == 0.875


def test_dreamervla_rl_update_uses_init_classifier_threshold_when_cfg_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dreamervla.workers.actor.learner_worker as mod

    seen_thresholds: list[float] = []

    def fake_rl_step(**kwargs):
        seen_thresholds.append(float(kwargs["classifier_threshold"]))
        return {"actor_loss": 0.75, "returns_mean": 0.25, "actor_grad_norm": 0.125}

    monkeypatch.setattr(mod, "dino_lumos_step", fake_rl_step)
    _patch_dummy_syncer(monkeypatch)

    train_cfg = _cotrain_train_cfg()
    train_cfg["classifier_threshold"] = None
    learner = LearnerWorker(
        _cotrain_model_cfg(),
        {"classifier_threshold": 0.125},
        train_cfg,
        _DirectReplay(),
    )
    learner.init()

    metrics = learner.update("rl", 1)

    assert metrics["rl/actor_loss"] == 0.75
    assert seen_thresholds == [0.125]


def test_dreamervla_wm_update_samples_replay_without_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dreamervla.workers.actor.learner_worker as mod

    def fake_wm_step(**kwargs):
        assert "images" not in kwargs["batch"]
        return {"loss": 1.25}

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)
    _patch_dummy_syncer(monkeypatch)

    replay = _ImageSensitiveReplay()
    learner = LearnerWorker(_cotrain_model_cfg(), {}, _cotrain_train_cfg(), replay)
    learner.init()

    metrics = learner.update("wm", 1)

    assert metrics["wm/loss"] == 1.25
    assert replay.include_images_calls == [False]


def test_dreamervla_rl_update_samples_replay_without_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dreamervla.workers.actor.learner_worker as mod

    def fake_rl_step(**kwargs):
        assert "images" not in kwargs["obs"]
        return {
            "actor_loss": 0.75,
            "returns_mean": 0.25,
            "returns_std": 0.5,
            "actor_grad_norm": 0.125,
            "ppo_step_applied": 1.0,
        }

    monkeypatch.setattr(mod, "dino_lumos_step", fake_rl_step)
    _patch_dummy_syncer(monkeypatch)

    train_cfg = _cotrain_train_cfg()
    train_cfg["algorithm_cfg"]["rollout_epoch"] = 3
    replay = _ImageSensitiveReplay()
    learner = LearnerWorker(_cotrain_model_cfg(), {}, train_cfg, replay)
    learner.init()

    metrics = learner.update("rl", 1)

    assert metrics["rl/actor_loss"] == 0.75
    assert replay.include_images_calls == [False, False, False]


def test_dreamervla_cotrain_rl_rollout_epoch_concatenates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dreamervla.workers.actor.learner_worker as mod

    calls = 0

    def fake_rl_step(**kwargs):
        nonlocal calls
        calls += 1
        obs = kwargs["obs"]
        assert obs["obs_embedding"].shape == (32, 3, 4)
        assert obs["actions"].shape == (32, 3, 7)
        assert obs["proprio"].shape == (32, 3, 8)
        assert obs["lang_emb"].shape == (32, 6)
        return {
            "actor_loss": 0.75,
            "returns_mean": 0.25,
            "returns_std": 0.5,
            "actor_grad_norm": 0.125,
            "ppo_step_applied": 1.0,
            "ppo_update_epochs": 1.0,
        }

    monkeypatch.setattr(mod, "dino_lumos_step", fake_rl_step)
    _patch_dummy_syncer(monkeypatch)

    replay = _DirectReplay()
    train_cfg = _cotrain_train_cfg()
    train_cfg["algorithm_cfg"]["rollout_epoch"] = 16
    learner = LearnerWorker(_cotrain_model_cfg(), {}, train_cfg, replay)
    learner.init()

    metrics = learner.update("rl", 1)

    assert calls == 1
    assert replay.sample_calls == 16
    assert metrics["rl/actor_loss"] == 0.75
    assert metrics["rl/ppo_step_applied"] == 1.0


def test_dreamervla_cotrain_actor_signal_gate_delays_rl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dreamervla.workers.actor.learner_worker as mod

    calls: list[str] = []
    f1_values = iter([0.25, 0.75])

    def fake_wm_step(**kwargs):
        del kwargs
        calls.append("wm")
        return {"loss": 1.0}

    def fake_classifier_step(**kwargs):
        del kwargs
        calls.append("classifier")
        return {"loss": 0.5, "acc": 0.5, "f1": next(f1_values)}

    def fake_rl_step(**kwargs):
        del kwargs
        calls.append("rl")
        return {
            "actor_loss": 0.75,
            "returns_mean": 0.25,
            "returns_std": 0.5,
            "actor_grad_norm": 0.125,
            "ppo_step_applied": 1.0,
        }

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)
    monkeypatch.setattr(mod, "online_classifier_update_step", fake_classifier_step)
    monkeypatch.setattr(mod, "dino_lumos_step", fake_rl_step)
    _patch_dummy_syncer(monkeypatch)

    train_cfg = _cotrain_train_cfg()
    train_cfg["actor_signal_gate"] = {
        "enabled": True,
        "min_classifier_f1": 0.6,
        "min_classifier_updates": 1,
    }
    learner = LearnerWorker(_cotrain_model_cfg(), {}, train_cfg, _DirectReplay())
    learner.init()

    first = learner.update("cotrain", 1)
    second = learner.update("cotrain", 1)

    assert calls == ["wm", "classifier", "wm", "classifier", "rl"]
    assert first["rl/skipped_no_signal"] == 1.0
    assert first["rl/actor_signal_ready"] == 0.0
    assert first["rl/actor_loss"] == 0.0
    assert second["rl/skipped_no_signal"] == 0.0
    assert second["rl/actor_signal_ready"] == 1.0
    assert second["rl/actor_loss"] == 0.75


def test_dreamervla_cotrain_mode_requires_components() -> None:
    replay = _DirectReplay()
    cfg = _cotrain_model_cfg()
    cfg.pop("world_model")
    learner = LearnerWorker(cfg, {}, _cotrain_train_cfg(), replay)

    with pytest.raises(ValueError, match="world_model"):
        learner.init()


def _wm_classifier_only_model_cfg() -> dict[str, Any]:
    return {
        "world_model": {
            "target": "dreamervla.workers.actor._test_models:TinyLumosWorldModel",
            "kwargs": {"hidden_dim": 4, "action_dim": 7},
        },
        "classifier": {
            "target": "dreamervla.workers.actor._test_models:TinySuccessClassifier",
            "kwargs": {"hidden_dim": 4, "window": 3},
        },
    }


def _wm_classifier_only_train_cfg() -> dict[str, Any]:
    return {
        "mode": "wm_classifier_only",
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 2,
        "classifier_batch_size": 2,
        "classifier_early_neg_stride": 2,
        "lr": 0.01,
        "optim_cfg": {"grad_clip_norm": 1.0, "zero_grad_set_to_none": True},
    }


def test_wm_classifier_only_does_not_require_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dummy_syncer(monkeypatch)
    learner = LearnerWorker(
        _wm_classifier_only_model_cfg(),
        {},
        _wm_classifier_only_train_cfg(),
        replay=None,
    )

    learner.init()

    assert learner.policy is None
    assert "policy" not in learner.optimizers
    assert learner.policy_optimizer is None


def test_wm_classifier_only_rejects_policy_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dummy_syncer(monkeypatch)
    cfg = _wm_classifier_only_model_cfg()
    cfg["policy"] = {
        "target": "dreamervla.workers.actor._test_models:TinySharedPolicy",
        "kwargs": {"hidden_dim": 4, "action_dim": 7},
    }
    learner = LearnerWorker(cfg, {}, _wm_classifier_only_train_cfg(), replay=None)

    with pytest.raises(ValueError, match="wm_classifier_only.*policy"):
        learner.init()


def test_wm_classifier_only_rejects_fsdp_train_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dummy_syncer(monkeypatch)
    train_cfg = _wm_classifier_only_train_cfg()
    train_cfg["fsdp"] = {"strategy": "none"}
    learner = LearnerWorker(_wm_classifier_only_model_cfg(), {}, train_cfg, replay=None)

    with pytest.raises(ValueError, match="wm_classifier_only.*FSDP"):
        learner.init()


@pytest.mark.parametrize("missing", ["world_model", "classifier"])
def test_wm_classifier_only_rejects_missing_required_components(
    missing: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dummy_syncer(monkeypatch)
    cfg = _wm_classifier_only_model_cfg()
    cfg.pop(missing)
    learner = LearnerWorker(cfg, {}, _wm_classifier_only_train_cfg(), replay=None)

    with pytest.raises(ValueError, match=missing):
        learner.init()


def test_wm_classifier_only_rejects_rl_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dummy_syncer(monkeypatch)
    learner = LearnerWorker(
        _wm_classifier_only_model_cfg(),
        {},
        _wm_classifier_only_train_cfg(),
        _DirectReplay(),
    )
    learner.init()

    with pytest.raises(ValueError, match="wm_classifier_only"):
        learner.update("rl", 1)


def test_wm_classifier_only_cotrain_updates_wm_then_classifier_without_rl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dreamervla.workers.actor.learner_worker as mod

    calls: list[str] = []

    def fake_wm_step(**kwargs):
        calls.append("wm")
        assert kwargs["policy"] is None
        assert kwargs["world_model"] is learner.world_model
        assert kwargs["optimizer"] is learner.world_model_optimizer
        return {"loss": 1.0}

    def fake_classifier_step(**kwargs):
        calls.append("classifier")
        assert kwargs["classifier"] is learner.classifier
        assert kwargs["optimizer"] is learner.classifier_optimizer
        return {"loss": 0.5, "acc": 1.0, "f1": 1.0}

    def fake_rl_step(**kwargs):
        del kwargs
        calls.append("rl")
        raise AssertionError("wm_classifier_only must not run dino_lumos_step")

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)
    monkeypatch.setattr(mod, "online_classifier_update_step", fake_classifier_step)
    monkeypatch.setattr(mod, "dino_lumos_step", fake_rl_step)
    _patch_dummy_syncer(monkeypatch)

    learner = LearnerWorker(
        _wm_classifier_only_model_cfg(),
        {},
        _wm_classifier_only_train_cfg(),
        _DirectReplay(),
    )
    learner.init()

    metrics = learner.update("cotrain", 1)

    assert calls == ["wm", "classifier"]
    assert metrics["wm/loss"] == 1.0
    assert metrics["cls/loss"] == 0.5
