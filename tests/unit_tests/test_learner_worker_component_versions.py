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
