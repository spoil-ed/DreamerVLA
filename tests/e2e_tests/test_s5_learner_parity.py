"""P0 parity: the Ray actor learner must match the direct update implementation
for the same components and batch.

The equivalence that matters — and that this guards — is that the Ray
``LearnerWorker`` computes the same training math as the update called directly.
Tiny test models are zero-initialised, so the only
non-determinism is the sampled batch; we remove it by feeding both learners one
captured fixed batch.
"""

from __future__ import annotations

import numpy as np
import ray
import torch

from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import NodePlacementStrategy
from dreamervla.scheduler.worker_group import WorkerGroup
from dreamervla.workers.actor.learner_worker import LearnerWorker
from dreamervla.workers.replay._test_replays import FixedBatchReplay
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


def _model_cfg() -> dict:
    return {
        "policy": {
            "target": "dreamervla.workers.actor._test_models:TinyLumosPolicy",
            "kwargs": {"hidden_dim": 4, "action_dim": 7, "chunk_size": 1},
        },
        "world_model": {
            "target": "dreamervla.workers.actor._test_models:TinyLumosWorldModel",
            "kwargs": {"hidden_dim": 4, "action_dim": 7},
        },
        "classifier": {
            "target": "dreamervla.workers.actor._test_models:TinySuccessClassifier",
            "kwargs": {"hidden_dim": 4, "window": 3},
        },
    }


def _train_cfg(store_name: str) -> dict:
    return {
        "mode": "dreamervla_cotrain",
        "batch_size": 2,
        "classifier_threshold": 0.5,
        "lr": 0.01,
        "device": "cpu",
        "precision": "fp32",
        "algorithm_cfg": {
            "ppo_rollouts_per_start": 1,
            "ppo_update_epochs": 1,
            "entropy_coef": 0.0,
            "lumos": {
                "chunk_size": 1,
                "episode_max_steps": 2,
                "classifier_min_steps": 1,
                "filter_zero_variance_groups": False,
            },
        },
        "optim_cfg": {"grad_clip_norm": 1.0, "zero_grad_set_to_none": True},
        "syncer": {"store_name": store_name},
    }


_RL_KEYS = ("rl/actor_loss", "rl/returns_mean", "rl/policy_grad_norm")


def test_ray_actor_learner_matches_in_process_learner_on_fixed_batch() -> None:
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        # Capture one deterministic batch from a populated replay.
        replay = ReplayWorker(
            {"capacity": 100, "sequence_length": 3, "task_ids": (0,), "rank": 0}
        )
        replay.init()
        for _ in range(4):
            replay.add_episode(_episode())
        torch.manual_seed(0)
        np.random.seed(0)
        fixed_batch = replay.sample(2)

        # In-process learner (single-machine path).
        local = LearnerWorker(
            _model_cfg(), {}, _train_cfg("parity_local_store"), FixedBatchReplay(fixed_batch)
        )
        local.init()
        local_metrics = local.update("rl", 1)

        # Ray-actor learner (distributed backend path), identical inputs.
        actor = WorkerGroup(
            LearnerWorker,
            _model_cfg(),
            {},
            _train_cfg("parity_ray_store"),
            FixedBatchReplay(fixed_batch),
        ).launch(cluster, NodePlacementStrategy(1))
        ray_metrics = actor.update("rl", 1).wait()[0]

        for key in _RL_KEYS:
            assert key in local_metrics and key in ray_metrics
            assert np.isclose(local_metrics[key], ray_metrics[key], rtol=1e-5, atol=1e-6), (
                f"{key}: ray={ray_metrics[key]} vs local={local_metrics[key]}"
            )
    finally:
        cluster.shutdown()
