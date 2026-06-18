from __future__ import annotations

import numpy as np
import ray

from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import NodePlacementStrategy
from dreamervla.scheduler.worker_group import WorkerGroup


def _online_replay_cls():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "dreamervla" / "runners" / "online_replay.py"
    spec = importlib.util.spec_from_file_location("dreamervla_online_replay_for_s2_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OnlineReplay


def _episode(task_id: int = 3, length: int = 5) -> list[dict]:
    return [
        {
            "image": np.full((4, 4, 3), t, dtype=np.uint8),
            "state": np.full((2,), t, dtype=np.float32),
            "action": np.full((7,), t, dtype=np.float32),
            "wm_action": np.full((7,), t, dtype=np.float32),
            "policy_action": np.full((7,), t, dtype=np.float32),
            "reward": np.float32(1.0 if t == length - 1 else 0.0),
            "done": np.float32(1.0 if t == length - 1 else 0.0),
            "discount": np.float32(0.0 if t == length - 1 else 1.0),
            "is_first": bool(t == 0),
            "is_terminal": np.float32(1.0 if t == length - 1 else 0.0),
            "is_last": np.float32(1.0 if t == length - 1 else 0.0),
            "task_id": task_id,
            "step": t,
            "task_description": f"task {task_id}",
            "obs_embedding": np.full((6,), t, dtype=np.float32),
        }
        for t in range(length)
    ]


def test_replay_worker_matches_online_replay_record_and_sample_schema() -> None:
    try:
        from dreamervla.workers.replay.replay_worker import ReplayWorker
    except ModuleNotFoundError as exc:
        raise AssertionError("ReplayWorker module should exist") from exc

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        replay_cfg = {
            "capacity": 100,
            "sequence_length": 3,
            "task_ids": (3,),
            "rank": 0,
        }
        group = WorkerGroup(ReplayWorker, replay_cfg).launch(cluster, NodePlacementStrategy(1))
        episode = _episode()

        actor_record = group.add_episode(episode).wait()[0]
        local_record = _online_replay_cls()(**replay_cfg).add_episode(episode)

        assert actor_record is not None
        assert local_record is not None
        for key in ("task_id", "success", "length", "finish_step", "rank"):
            assert actor_record[key] == local_record[key]
        assert group.size().wait() == [1]
        assert group.ready(1).wait() == [True]

        batch = group.sample(2).wait()[0]
        assert set(
            [
                "images",
                "obs_embedding",
                "actions",
                "current_actions",
                "rewards",
                "dones",
                "task_ids",
                "episode_success",
            ]
        ).issubset(batch)
        assert tuple(batch["images"].shape) == (2, 3, 4, 4, 3)
        assert tuple(batch["obs_embedding"].shape) == (2, 3, 6)
        assert batch["task_ids"].tolist() == [3, 3]
        assert batch["episode_success"].tolist() == [True, True]
    finally:
        cluster.shutdown()
