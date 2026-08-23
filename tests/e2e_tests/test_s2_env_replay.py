from __future__ import annotations

import numpy as np
import ray

from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import NodePlacementStrategy
from dreamervla.scheduler.worker_group import WorkerGroup
from dreamervla.workers.replay.replay_worker import ReplayWorker


def test_env_workers_push_completed_episodes_to_replay() -> None:
    try:
        from dreamervla.workers.env.env_worker import EnvWorker
    except ModuleNotFoundError as exc:
        raise AssertionError("EnvWorker module should exist") from exc

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        replay_group = WorkerGroup(
            ReplayWorker,
            {"capacity": 100, "sequence_length": 3, "task_ids": (5,), "rank": 0},
        ).launch(cluster, NodePlacementStrategy(1))
        replay = replay_group.workers[0]
        env_cfg = {
            "target": "dreamervla.workers.env._test_envs:CounterEnv",
            "kwargs": {"horizon": 3, "image_shape": (4, 4, 3), "embedding_dim": 6},
        }
        envs = WorkerGroup(EnvWorker, env_cfg, task_id=5, replay=replay).launch(
            cluster, NodePlacementStrategy(2)
        )

        initial = envs.current_obs().wait()
        assert [obs["step"] for obs in initial] == [0, 0]

        for t in range(3):
            action = np.full((7,), t, dtype=np.float32)
            hidden = np.full((6,), t, dtype=np.float32)
            results = envs.step(action, hidden).wait()

        assert [done for _, done, _ in results] == [True, True]
        assert [obs["step"] for obs, _, _ in results] == [0, 0]
        assert replay_group.size().wait() == [2]
        assert replay_group.ready(2).wait() == [True]

        batch = replay_group.sample(2).wait()[0]
        assert tuple(batch["images"].shape) == (2, 3, 4, 4, 3)
        assert tuple(batch["obs_embedding"].shape) == (2, 3, 6)
        assert batch["task_ids"].tolist() == [5, 5]
        assert batch["episode_success"].tolist() == [True, True]
    finally:
        cluster.shutdown()
