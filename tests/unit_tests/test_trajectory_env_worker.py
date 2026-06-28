from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dreamervla.workers.cotrain.messages import (
    ObservationMsg,
    RolloutResultMsg,
    TrajectoryShard,
)
import dreamervla.workers.env.trajectory_env_worker as trajectory_env_worker
from dreamervla.workers.env.trajectory_env_worker import (
    BaseTrajectoryEnvWorker,
    RealEnvWorker,
    WMEnvWorker,
)


class _MemoryChannel:
    def __init__(self, initial: list[Any] | None = None) -> None:
        self.queue = list(initial or [])
        self.puts: list[tuple[str, Any]] = []

    def put(self, item: Any, *, key: str = "default") -> None:
        self.puts.append((str(key), item))

    def get(self, *, key: str = "default") -> Any:
        del key
        assert self.queue
        return self.queue.pop(0)


class _MemoryReplay:
    def __init__(self) -> None:
        self.episodes: list[list[dict[str, Any]]] = []

    def add_episode(self, episode: list[dict[str, Any]], source: str = "online") -> None:
        del source
        self.episodes.append(list(episode))


def _counter_env_cfg() -> dict[str, Any]:
    return {
        "target": "dreamervla.workers.env._test_envs:CounterEnv",
        "kwargs": {"horizon": 2, "embedding_dim": 4},
    }


def _batched_counter_env_cfg() -> dict[str, Any]:
    return {
        "target": "dreamervla.workers.env._test_envs:BatchedCounterEnv",
        "kwargs": {"num_envs": 3, "horizon": 2, "embedding_dim": 4},
    }


def _short_horizon_counter_env_cfg() -> dict[str, Any]:
    return {
        "target": "dreamervla.workers.env._test_envs:CounterEnv",
        "kwargs": {"horizon": 1, "embedding_dim": 4},
    }


def _no_sidecar_env_cfg() -> dict[str, Any]:
    return {
        "target": "dreamervla.workers.env._test_envs:NoSidecarTrainEnv",
        "kwargs": {"horizon": 1, "state_dim": 2},
    }


def _no_sidecar_oft_env_cfg() -> dict[str, Any]:
    cfg = _no_sidecar_env_cfg()
    cfg["action_postprocess"] = "openvla_oft"
    return cfg


def _rollout_result() -> RolloutResultMsg:
    return RolloutResultMsg(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_id=0,
        step=0,
        actions=np.zeros((2, 3), dtype=np.float32),
        prev_logprobs=np.array([0.25], dtype=np.float32),
        prev_values=None,
        forward_inputs={
            "hidden": np.ones((1, 4), dtype=np.float32),
            "action": np.zeros((1, 2, 3), dtype=np.float32),
        },
        versions={"policy": 1},
    )


def _sidecar_rollout_result(action: np.ndarray | None = None) -> RolloutResultMsg:
    return RolloutResultMsg(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_id=0,
        step=0,
        actions=np.asarray(
            action if action is not None else np.zeros((1, 7), dtype=np.float32),
            dtype=np.float32,
        ),
        prev_logprobs=np.array([0.25], dtype=np.float32),
        prev_values=None,
        forward_inputs={
            "hidden": np.full((1, 4), 7.0, dtype=np.float32),
            "lang_emb": np.full((2,), 3.0, dtype=np.float32),
        },
        versions={"policy": 4},
    )


def test_real_env_worker_buffers_rollout_result_into_trajectory() -> None:
    worker = BaseTrajectoryEnvWorker(
        role="real_env",
        env_cfg=_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
    )
    try:
        worker.init()
        obs = worker.bootstrap_obs()[0]

        assert isinstance(obs, ObservationMsg)
        assert obs.obs["step"] == 0

        shard = worker.apply_rollout_result(_rollout_result())

        assert isinstance(shard, TrajectoryShard)
        assert shard.actions.shape == (1, 1, 2, 3)
        assert shard.forward_inputs["hidden"].shape == (1, 1, 4)
        assert shard.forward_inputs["action"].shape == (1, 1, 2, 3)
        assert shard.versions["policy"].shape == (1, 1)
        assert shard.prev_logprobs.shape == (1, 1)
        assert shard.rewards.shape == (1, 1, 2)
        assert shard.dones.shape == (1, 1, 2)
        assert shard.rewards[0, 0].tolist() == [0.0, 1.0]
        assert shard.dones[0, 0].tolist() == [False, True]
    finally:
        worker.close()


def test_real_env_worker_replay_transition_has_obs_embedding() -> None:
    replay = _MemoryReplay()
    worker = BaseTrajectoryEnvWorker(
        role="real_env",
        env_cfg=_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
        replay=replay,
    )
    try:
        worker.init()
        worker.bootstrap_obs()
        worker.apply_rollout_result(_rollout_result())

        assert len(replay.episodes) == 1
        assert replay.episodes[0][0]["obs_embedding"].shape == (4,)
        assert replay.episodes[0][0]["episode_id"] == 0
    finally:
        worker.close()


def test_real_env_worker_attaches_rollout_sidecars_to_no_embedding_env_records() -> None:
    replay = _MemoryReplay()
    worker = RealEnvWorker(
        env_cfg=_no_sidecar_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
        replay=replay,
    )
    try:
        worker.init()
        worker.bootstrap_obs()

        worker.apply_rollout_result(_sidecar_rollout_result())

        assert len(replay.episodes) == 1
        step = replay.episodes[0][0]
        assert step["obs_embedding"].tolist() == [7.0, 7.0, 7.0, 7.0]
        assert step["lang_emb"].tolist() == [3.0, 3.0]
        assert step["policy_version"] == 4
    finally:
        worker.close()


def test_real_env_worker_postprocesses_openvla_oft_env_action_without_overwriting_policy_action() -> None:
    replay = _MemoryReplay()
    worker = RealEnvWorker(
        env_cfg=_no_sidecar_oft_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
        replay=replay,
    )
    policy_action = np.array(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.25]],
        dtype=np.float32,
    )
    try:
        worker.init()
        worker.bootstrap_obs()

        worker.apply_rollout_result(_sidecar_rollout_result(policy_action))

        env = worker.envs[0]
        np.testing.assert_allclose(
            env.received_actions[0],
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0], dtype=np.float32),
        )
        step = replay.episodes[0][0]
        np.testing.assert_allclose(step["policy_action"], policy_action.reshape(-1))
        np.testing.assert_allclose(step["wm_action"], env.received_actions[0])
    finally:
        worker.close()


def test_apply_rollout_result_stops_chunk_at_episode_boundary() -> None:
    replay = _MemoryReplay()
    worker = BaseTrajectoryEnvWorker(
        role="real_env",
        env_cfg=_short_horizon_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
        replay=replay,
    )
    try:
        worker.init()
        worker.bootstrap_obs()

        shard = worker.apply_rollout_result(_rollout_result())

        assert shard.actions.shape == (1, 1, 2, 3)
        assert shard.rewards[0, 0].tolist() == [1.0, 0.0]
        assert shard.dones[0, 0].tolist() == [True, True]
        assert worker._last_apply_completed_episodes == 1
        assert worker._last_apply_physical_steps == 1
        assert len(replay.episodes) == 1
        assert len(replay.episodes[0]) == 1
        assert worker._obs_by_slot[0]["episode_id"] == 1
        assert worker._obs_by_slot[0]["step"] == 0
    finally:
        worker.close()


def test_apply_rollout_result_rejects_empty_or_oversized_chunks() -> None:
    worker = BaseTrajectoryEnvWorker(
        role="real_env",
        env_cfg=_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
    )
    try:
        worker.init()
        worker.bootstrap_obs()
        empty = replace(_rollout_result(), actions=np.zeros((0, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="non-empty"):
            worker.apply_rollout_result(empty)

        oversized = replace(
            _rollout_result(),
            actions=np.zeros((3, 3), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="num_action_chunks"):
            worker.apply_rollout_result(oversized)
    finally:
        worker.close()


def test_real_and_wm_worker_classes_are_distinct_roles() -> None:
    assert RealEnvWorker.role_name == "real_env"
    assert WMEnvWorker.role_name == "wm_env"


def test_trajectory_env_worker_uses_single_batched_env_when_available() -> None:
    worker = WMEnvWorker(
        env_cfg=_batched_counter_env_cfg(),
        num_slots=3,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
    )
    try:
        worker.init()
        assert len(worker.envs) == 1
        messages = worker.bootstrap_obs()
        assert [msg.slot_id for msg in messages] == [0, 1, 2]
    finally:
        worker.close()


def test_trajectory_env_worker_applies_pending_state_sync_after_init() -> None:
    worker = WMEnvWorker(
        env_cfg=_batched_counter_env_cfg(),
        num_slots=3,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
    )
    try:
        worker.load_world_model_state({"weight": 1}, version=4)
        worker.load_classifier_state({"weight": 2}, version=5)
        worker.init()

        env = worker.envs[0]
        assert env.wm_loaded_version == 4
        assert env.classifier_loaded_version == 5
        assert worker.bootstrap_obs()[0].versions == {
            "world_model": 4,
            "classifier": 5,
        }
    finally:
        worker.close()


def test_wm_env_worker_requires_component_state_loaders() -> None:
    worker = WMEnvWorker(
        env_cfg=_no_sidecar_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )
    try:
        worker.init()

        with pytest.raises(TypeError, match="load_world_model_state"):
            worker.load_world_model_state({"weight": 1}, version=4)
        with pytest.raises(TypeError, match="load_classifier_state"):
            worker.load_classifier_state({"weight": 2}, version=5)
    finally:
        worker.close()


def test_interact_routes_observations_rollout_results_and_trajectory(
    monkeypatch,
) -> None:
    worker = RealEnvWorker(
        env_cfg=_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
    )
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel([_rollout_result(), _rollout_result()]),
        "actor": _MemoryChannel(),
    }
    monkeypatch.setattr(
        trajectory_env_worker.Channel,
        "connect",
        staticmethod(lambda name: channels[str(name)]),
    )

    try:
        worker.init()
        metrics = worker.interact("env", "rollout", "actor")

        assert channels["env"].puts[0][0] == "default"
        assert isinstance(channels["env"].puts[0][1], ObservationMsg)
        assert channels["env"].puts[-1][1].obs["_final_bootstrap"] is True
        assert channels["actor"].puts[0][0] == "default"
        assert isinstance(channels["actor"].puts[0][1], TrajectoryShard)
        assert metrics["env/chunk_steps"] == 1.0
        assert metrics["env/physical_steps"] == 2.0
        assert metrics["env/steps"] == 2.0
        assert metrics["env/trajectory_shards"] == 1.0
        assert metrics["env/episodes_completed"] == 1.0
        assert metrics["env/final_bootstrap_requests"] == 1.0
    finally:
        worker.close()


def test_interact_flushes_partial_episode_at_rollout_epoch_boundary(
    monkeypatch,
) -> None:
    replay = _MemoryReplay()
    worker = RealEnvWorker(
        env_cfg={
            "target": "dreamervla.workers.env._test_envs:CounterEnv",
            "kwargs": {"horizon": 10, "embedding_dim": 4},
        },
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
        replay=replay,
    )
    one_step = replace(
        _rollout_result(),
        actions=np.zeros((1, 3), dtype=np.float32),
    )
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel([one_step, one_step]),
        "actor": _MemoryChannel(),
    }
    monkeypatch.setattr(
        trajectory_env_worker.Channel,
        "connect",
        staticmethod(lambda name: channels[str(name)]),
    )

    try:
        worker.init()
        metrics = worker.interact("env", "rollout", "actor")

        assert metrics["env/episodes_flushed"] == 1.0
        assert len(replay.episodes) == 1
        assert len(replay.episodes[0]) == 1
        last_step = replay.episodes[0][-1]
        assert bool(last_step["is_last"]) is True
        assert bool(last_step["is_terminal"]) is False
        assert float(last_step["discount"]) == 1.0
        assert worker._episode_ids_by_slot[0] == 1
        assert worker._episodes_by_slot[0] == []
    finally:
        worker.close()
