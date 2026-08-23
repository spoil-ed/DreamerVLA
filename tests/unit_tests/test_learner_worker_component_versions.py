from __future__ import annotations

from typing import Any

from dreamervla.workers.actor.learner_worker import LearnerWorker


class _FakeSyncer:
    def __init__(self) -> None:
        self.pushes: list[tuple[str, int, tuple[str, ...]]] = []

    def push(self, key: str, state_dict: dict[str, Any], version: int) -> None:
        self.pushes.append((str(key), int(version), tuple(sorted(state_dict))))


def test_learner_worker_syncs_independent_component_versions():
    model_cfg = {
        "policy": {
            "target": "dreamervla.workers.actor._test_models:TinySharedPolicy",
            "kwargs": {"hidden_dim": 4, "action_dim": 7},
        },
        "world_model": {
            "target": "dreamervla.workers.actor._test_models:TinyTrainableWorldModel",
            "kwargs": {"hidden_dim": 4},
        },
        "classifier": {
            "target": "dreamervla.workers.actor._test_models:TinySuccessClassifier",
            "kwargs": {"hidden_dim": 4, "window": 3},
        },
    }
    learner = LearnerWorker(
        model_cfg,
        {},
        {"mode": "synthetic_ppo", "device": "cpu", "syncer": {"store_name": "unused"}},
        replay=None,
    )
    learner.init()
    fake = _FakeSyncer()
    learner.syncer = fake

    learner.sync_weights("world_model", 3)
    learner.sync_weights("classifier", 4)
    learner.sync_weights("policy", 5)

    assert [(key, version) for key, version, _ in fake.pushes] == [
        ("world_model", 3),
        ("classifier", 4),
        ("policy", 5),
    ]


def test_learner_worker_checkpoint_round_trips_component_optimizers() -> None:
    model_cfg = {
        "world_model": {
            "target": "dreamervla.workers.actor._test_models:TinyTrainableWorldModel",
            "kwargs": {"hidden_dim": 4},
        },
        "classifier": {
            "target": "dreamervla.workers.actor._test_models:TinySuccessClassifier",
            "kwargs": {"hidden_dim": 4, "window": 3},
        },
    }
    train_cfg = {
        "mode": "wm_classifier_only",
        "device": "cpu",
        "classifier_threshold": 0.5,
        "syncer": {"store_name": "unused"},
        "optimizers": {
            "world_model": {"lr": 1.0e-3},
            "classifier": {"lr": 2.0e-3},
        },
    }
    first = LearnerWorker(model_cfg, {}, train_cfg, replay=None)
    first.init()
    first.optimizers["world_model"].param_groups[0]["lr"] = 0.123
    first.optimizers["classifier"].param_groups[0]["lr"] = 0.456

    sync_payload = first.state_dicts()
    payload = first.state_dicts(include_optimizers=True)
    second = LearnerWorker(model_cfg, payload, train_cfg, replay=None)
    second.init()

    assert "world_model_optimizer" not in sync_payload
    assert "classifier_optimizer" not in sync_payload
    assert "world_model_optimizer" in payload
    assert "classifier_optimizer" in payload
    assert second.optimizers["world_model"].param_groups[0]["lr"] == 0.123
    assert second.optimizers["classifier"].param_groups[0]["lr"] == 0.456
