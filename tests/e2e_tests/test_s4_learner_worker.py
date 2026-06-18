from __future__ import annotations

import numpy as np
import ray
import torch

from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import NodePlacementStrategy
from dreamervla.scheduler.worker_group import WorkerGroup
from dreamervla.workers.replay.replay_worker import ReplayWorker


def _episode(length: int = 5) -> list[dict]:
    episode = []
    for t in range(length):
        action = np.full((7,), float(t), dtype=np.float32)
        episode.append(
            {
                "image": np.zeros((4, 4, 3), dtype=np.uint8),
                "state": np.zeros((2,), dtype=np.float32),
                "action": action,
                "wm_action": action,
                "policy_action": action,
                "reward": np.float32(1.0 if t == length - 1 else 0.0),
                "done": np.float32(t == length - 1),
                "discount": np.float32(0.0 if t == length - 1 else 1.0),
                "is_first": bool(t == 0),
                "is_terminal": bool(t == length - 1),
                "is_last": bool(t == length - 1),
                "task_id": 0,
                "step": t,
                "task_description": "synthetic",
                "obs_embedding": np.full((4,), float(t), dtype=np.float32),
            }
        )
    return episode


def test_learner_worker_runs_synthetic_ppo_and_syncs_policy_weights() -> None:
    try:
        from dreamervla.hybrid_engines.weight_syncer.objectstore import (
            ObjectStoreWeightSyncer,
        )
        from dreamervla.workers.actor.learner_worker import LearnerWorker
    except ModuleNotFoundError as exc:
        raise AssertionError("LearnerWorker and syncer modules should exist") from exc

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        replay_group = WorkerGroup(
            ReplayWorker,
            {"capacity": 100, "sequence_length": 3, "task_ids": (0,), "rank": 0},
        ).launch(cluster, NodePlacementStrategy(1))
        replay = replay_group.workers[0]
        for _ in range(4):
            replay_group.add_episode(_episode()).wait()

        model_cfg = {
            "policy": {
                "target": "dreamervla.workers.actor._test_models:TinyTrainablePolicy",
                "kwargs": {"hidden_dim": 4, "action_dim": 7},
            }
        }
        train_cfg = {
            "mode": "synthetic_ppo",
            "batch_size": 2,
            "lr": 0.05,
            "device": "cpu",
            "syncer": {"store_name": "learner_worker_weight_store"},
        }
        learner = WorkerGroup(LearnerWorker, model_cfg, {}, train_cfg, replay).launch(
            cluster, NodePlacementStrategy(1)
        )

        before = learner.state_dicts().wait()[0]["policy"]["bias"].clone()
        metrics = learner.update("rl", 4).wait()[0]
        after_state = learner.state_dicts().wait()[0]["policy"]

        assert metrics["train/rl_loss"] >= 0.0
        assert not torch.allclose(before, after_state["bias"])

        learner.sync_weights("policy", version=3).wait()
        target = torch.nn.Linear(4, 7)
        syncer = ObjectStoreWeightSyncer(store_name="learner_worker_weight_store")
        assert syncer.pull("policy", target, local_version=0) == 3
        assert torch.allclose(target.bias, after_state["bias"])
    finally:
        cluster.shutdown()
