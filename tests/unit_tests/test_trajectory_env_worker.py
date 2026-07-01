from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import dreamervla.workers.env.trajectory_env_worker as trajectory_env_worker
from dreamervla.workers.cotrain.messages import (
    ObservationMsg,
    RolloutResultMsg,
    TrajectoryShard,
)
from dreamervla.workers.env.trajectory_env_worker import (
    BaseTrajectoryEnvWorker,
    RealEnvWorker,
    WMEnvWorker,
)


class _MemoryChannel:
    def __init__(self, initial: list[Any] | None = None) -> None:
        self.queue = list(initial or [])
        self.puts: list[tuple[str, Any]] = []
        self.put_no_wait_calls: list[tuple[str, Any]] = []
        self.gets: list[str] = []

    def put(self, item: Any, *, key: str = "default") -> None:
        self.puts.append((str(key), item))

    def put_no_wait(self, item: Any, *, key: str = "default"):
        self.put_no_wait_calls.append((str(key), item))
        self.put(item, key=key)
        return _ReadyPut()

    def get(self, *, key: str = "default") -> Any:
        self.gets.append(str(key))
        assert self.queue
        return self.queue.pop(0)


class _ReadyPut:
    def wait(self) -> None:
        return None


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


def _long_horizon_counter_env_cfg() -> dict[str, Any]:
    return {
        "target": "dreamervla.workers.env._test_envs:CounterEnv",
        "kwargs": {"horizon": 99, "embedding_dim": 4},
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


def _rollout_result_for_slot(slot_id: int) -> RolloutResultMsg:
    return replace(_rollout_result(), slot_id=int(slot_id), episode_id=int(slot_id))


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


def test_real_env_worker_uses_full_component_version_schema_without_models() -> None:
    worker = RealEnvWorker(
        env_cfg=_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
    )
    try:
        worker.init()
        worker.set_global_step(2)
        obs = worker.bootstrap_obs()[0]

        assert obs.versions == {
            "world_model_version": 0,
            "wm_version": 0,
            "classifier_version": 0,
            "reward_or_classifier_version": 0,
            "global_step": 2,
        }
    finally:
        worker.close()


def test_env_worker_propagates_component_and_step_versions_to_replay() -> None:
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
        worker.set_global_step(11)
        worker.load_component_states(
            {
                "world_model": {},
                "classifier": {},
            },
            version=3,
        )
        obs = worker.bootstrap_obs()[0]

        assert obs.versions["global_step"] == 11
        assert obs.versions["world_model_version"] == 3
        assert obs.versions["wm_version"] == 3
        assert obs.versions["classifier_version"] == 3
        assert obs.versions["reward_or_classifier_version"] == 3

        result = replace(
            _rollout_result(),
            versions={
                "policy": 5,
                "actor_policy_version": 5,
                "rollout_policy_version": 5,
                "global_step": 11,
                "wm_version": 3,
                "classifier_version": 3,
                "reward_or_classifier_version": 3,
            },
        )
        worker.apply_rollout_result(result)

        step = replay.episodes[0][0]
        assert step["policy_version"] == 5
        assert step["actor_policy_version"] == 5
        assert step["rollout_policy_version"] == 5
        assert step["global_step"] == 11
        assert step["wm_version"] == 3
        assert step["classifier_version"] == 3
        assert step["reward_or_classifier_version"] == 3
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
        assert step["proprio"].tolist() == step["state"].tolist()
        assert step["policy_version"] == 4
    finally:
        worker.close()


def test_real_env_worker_respects_explicit_spawn_slots_false_for_egl_backend(monkeypatch) -> None:
    cfg = dict(_counter_env_cfg())
    cfg["render_backend"] = "egl"
    cfg["spawn_env_slots"] = False
    worker = RealEnvWorker(
        env_cfg=cfg,
        num_slots=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )
    build_calls: list[dict[str, Any]] = []
    original_build_env_from_cfg = trajectory_env_worker._build_env_from_cfg

    def build_inproc(cfg_arg: dict[str, Any]) -> Any:
        build_calls.append(dict(cfg_arg))
        return original_build_env_from_cfg(cfg_arg)

    monkeypatch.setattr(trajectory_env_worker, "_build_env_from_cfg", build_inproc)

    try:
        worker.init()

        assert len(build_calls) == 2
        assert worker._spawned_env is False
        assert len(worker.envs) == 2
    finally:
        worker.close()


def test_real_env_worker_pins_osmesa_for_inproc_non_egl_backend(monkeypatch) -> None:
    cfg = dict(_counter_env_cfg())
    cfg["render_backend"] = "osmesa"
    cfg["spawn_env_slots"] = False
    worker = RealEnvWorker(
        env_cfg=cfg,
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)

    try:
        worker.init()

        assert os.environ["MUJOCO_GL"] == "osmesa"
        assert os.environ["PYOPENGL_PLATFORM"] == "osmesa"
        assert worker._spawned_env is False
    finally:
        worker.close()


def test_real_env_worker_uses_spawned_slots_when_explicitly_enabled_for_egl_backend(monkeypatch) -> None:
    cfg = dict(_counter_env_cfg())
    cfg["render_backend"] = "egl"
    cfg["spawn_env_slots"] = True
    worker = RealEnvWorker(
        env_cfg=cfg,
        num_slots=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )
    calls: list[int] = []

    def fail_inproc_build(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("explicitly spawned EGL real env slots must not be built in the Ray actor")

    def fake_init_spawn_slot(slot_id: int, *, task_id: int | None = None, start_episode_id: int = 0) -> None:
        del task_id
        calls.append(int(slot_id))
        worker._spawned_env = True
        worker._spawn_procs[int(slot_id)] = object()
        worker._spawn_conns[int(slot_id)] = object()
        worker._obs_by_slot[int(slot_id)] = {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "state": np.zeros((2,), dtype=np.float32),
            "obs_embedding": np.zeros((4,), dtype=np.float32),
            "task_id": 0,
            "episode_id": int(start_episode_id),
            "step": 0,
            "task_description": "task 0",
            "is_first": True,
        }

    monkeypatch.setattr(trajectory_env_worker, "_build_env_from_cfg", fail_inproc_build)
    monkeypatch.setattr(worker, "_init_spawn_slot", fake_init_spawn_slot, raising=False)

    worker.init()

    assert calls == [0, 1]
    assert worker._spawned_env is True
    assert worker.envs == []


def test_spawned_real_env_worker_records_child_transition() -> None:
    replay = _MemoryReplay()
    worker = RealEnvWorker(
        env_cfg={"render_backend": "egl"},
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
        replay=replay,
    )
    worker._spawned_env = True
    worker._spawn_procs[0] = object()
    worker._spawn_conns[0] = object()
    worker._obs_by_slot[0] = {"step": 0, "state": np.zeros((2,), dtype=np.float32)}
    transition = {
        "state": np.zeros((2,), dtype=np.float32),
        "action": np.zeros((7,), dtype=np.float32),
        "reward": np.float32(1.0),
        "done": np.float32(1.0),
        "discount": np.float32(0.0),
        "is_first": True,
        "is_terminal": True,
        "is_last": True,
        "task_id": 0,
        "episode_id": 0,
        "step": 0,
        "task_description": "task 0",
        "obs_embedding": np.ones((4,), dtype=np.float32),
        "policy_version": 4,
    }

    def fake_rpc(
        cmd: str,
        payload: Any = None,
        *,
        slot_id: int = 0,
        timeout_s: float | None = None,
    ) -> Any:
        assert timeout_s == 120.0
        assert cmd == "step"
        assert slot_id == 0
        env_action, policy_action, sidecars = payload
        np.testing.assert_allclose(env_action, np.zeros((7,), dtype=np.float32))
        np.testing.assert_allclose(policy_action, np.zeros((7,), dtype=np.float32))
        assert sidecars["policy_version"] == 4
        return transition, {"step": 1}, 1.0, True, {"success": True}

    worker._spawn_rpc_with_timeout = fake_rpc  # type: ignore[method-assign]

    try:
        worker._step_slot(
            0,
            np.zeros((7,), dtype=np.float32),
            transition_sidecars={"obs_embedding": np.ones((4,), dtype=np.float32), "policy_version": 4},
        )

        assert len(replay.episodes) == 1
        assert replay.episodes[0][0]["policy_version"] == 4
        assert worker._episode_ids_by_slot[0] == 1
    finally:
        worker.close()


def test_spawned_real_env_worker_respawns_dead_child_when_enabled(monkeypatch) -> None:
    class _DeadConn:
        def send(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def poll(self, timeout: float | None = None) -> bool:
            del timeout
            return True

        def recv(self) -> Any:
            raise EOFError

    class _Proc:
        def __init__(self) -> None:
            self.terminated = False

        def is_alive(self) -> bool:
            return False

        def terminate(self) -> None:
            self.terminated = True

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

    worker = RealEnvWorker(
        env_cfg={"render_backend": "egl", "egl_max_respawns": 1},
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )
    old_proc = _Proc()
    worker._spawned_env = True
    worker._spawn_procs[0] = old_proc
    worker._spawn_conns[0] = _DeadConn()
    worker._obs_by_slot[0] = {"step": 0}
    worker._episodes_by_slot[0] = [{"partial": 1}]
    calls: list[tuple[int, int]] = []

    def fake_init_spawn_slot(slot_id: int, *, task_id: int | None = None, start_episode_id: int = 0) -> None:
        del task_id
        calls.append((int(slot_id), int(start_episode_id)))
        worker._spawn_procs[int(slot_id)] = _Proc()
        worker._spawn_conns[int(slot_id)] = object()
        worker._obs_by_slot[int(slot_id)] = {"step": 0, "episode_id": int(start_episode_id)}
        worker._episode_ids_by_slot[int(slot_id)] = int(start_episode_id)

    monkeypatch.setattr(worker, "_init_spawn_slot", fake_init_spawn_slot, raising=False)

    obs, reward, done, info = worker._step_slot(
        0,
        np.zeros((7,), dtype=np.float32),
        transition_sidecars={},
    )

    assert obs["episode_id"] == 1
    assert reward == 0.0
    assert done is True
    assert info["env_crash"] is True
    assert info["respawned"] is True
    assert info["respawn_count"] == 1
    assert calls == [(0, 1)]
    assert worker._episodes_by_slot[0] == []
    assert old_proc.terminated is True


def test_spawned_real_env_worker_respawns_hung_child_when_step_timeout_enabled(
    monkeypatch,
) -> None:
    class _HungConn:
        def __init__(self) -> None:
            self.poll_timeout: float | None = None
            self.closed = False
            self.sent: list[Any] = []

        def send(self, payload: Any) -> None:
            self.sent.append(payload)

        def poll(self, timeout: float | None = None) -> bool:
            self.poll_timeout = timeout
            return False

        def recv(self) -> Any:
            raise AssertionError("step response recv should be guarded by poll timeout")

        def close(self) -> None:
            self.closed = True

    class _Proc:
        def __init__(self) -> None:
            self.terminated = False

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            self.terminated = True

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

    worker = RealEnvWorker(
        env_cfg={
            "render_backend": "egl",
            "egl_max_respawns": 1,
            "egl_step_timeout_s": 0.25,
        },
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )
    old_proc = _Proc()
    old_conn = _HungConn()
    worker._spawned_env = True
    worker._spawn_procs[0] = old_proc
    worker._spawn_conns[0] = old_conn
    worker._obs_by_slot[0] = {"step": 0}
    worker._episodes_by_slot[0] = [{"partial": 1}]
    calls: list[tuple[int, int]] = []

    def fake_init_spawn_slot(
        slot_id: int,
        *,
        task_id: int | None = None,
        start_episode_id: int = 0,
    ) -> None:
        del task_id
        calls.append((int(slot_id), int(start_episode_id)))
        worker._spawn_procs[int(slot_id)] = _Proc()
        worker._spawn_conns[int(slot_id)] = object()
        worker._obs_by_slot[int(slot_id)] = {
            "step": 0,
            "episode_id": int(start_episode_id),
        }
        worker._episode_ids_by_slot[int(slot_id)] = int(start_episode_id)

    monkeypatch.setattr(worker, "_init_spawn_slot", fake_init_spawn_slot, raising=False)

    obs, reward, done, info = worker._step_slot(
        0,
        np.zeros((7,), dtype=np.float32),
        transition_sidecars={},
    )

    assert obs["episode_id"] == 1
    assert reward == 0.0
    assert done is True
    assert info["env_crash"] is True
    assert info["env_timeout"] is True
    assert info["respawned"] is True
    assert info["respawn_count"] == 1
    assert calls == [(0, 1)]
    assert worker._episodes_by_slot[0] == []
    assert old_conn.poll_timeout == 0.25
    assert old_conn.closed is True
    assert old_proc.terminated is True


def test_spawned_real_env_worker_resets_respawn_budget_after_success(
    monkeypatch,
) -> None:
    class _OkConn:
        def send(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def poll(self, timeout: float | None = None) -> bool:
            del timeout
            return True

        def recv(self) -> Any:
            return (
                "ok",
                (
                    {"policy_action": np.zeros((7,), dtype=np.float32)},
                    {"step": 1},
                    0.0,
                    False,
                    {},
                ),
            )

    class _DeadConn:
        def send(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def poll(self, timeout: float | None = None) -> bool:
            del timeout
            return True

        def recv(self) -> Any:
            raise EOFError

    class _Proc:
        def terminate(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

    worker = RealEnvWorker(
        env_cfg={"render_backend": "egl", "egl_max_respawns": 1},
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )
    worker._spawned_env = True
    worker._spawn_procs[0] = _Proc()
    worker._spawn_conns[0] = _OkConn()
    worker._obs_by_slot[0] = {"step": 0}
    worker._episodes_by_slot[0] = []
    worker._egl_respawns_by_slot[0] = 1
    calls: list[tuple[int, int]] = []

    def fake_init_spawn_slot(
        slot_id: int,
        *,
        task_id: int | None = None,
        start_episode_id: int = 0,
    ) -> None:
        del task_id
        calls.append((int(slot_id), int(start_episode_id)))
        worker._spawn_procs[int(slot_id)] = _Proc()
        worker._spawn_conns[int(slot_id)] = object()
        worker._obs_by_slot[int(slot_id)] = {"step": 0}
        worker._episode_ids_by_slot[int(slot_id)] = int(start_episode_id)

    monkeypatch.setattr(worker, "_init_spawn_slot", fake_init_spawn_slot, raising=False)

    _obs, _reward, done, _info = worker._step_slot(
        0,
        np.zeros((7,), dtype=np.float32),
        transition_sidecars={},
    )
    assert done is False
    assert worker._egl_respawns_by_slot[0] == 0

    worker._spawn_conns[0] = _DeadConn()
    obs, reward, done, info = worker._step_slot(
        0,
        np.zeros((7,), dtype=np.float32),
        transition_sidecars={},
    )

    assert obs["step"] == 0
    assert reward == 0.0
    assert done is True
    assert info["env_crash"] is True
    assert info["respawned"] is True
    assert info["respawn_count"] == 1
    assert calls == [(0, 1)]


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
            "world_model_version": 4,
            "wm_version": 4,
            "classifier_version": 5,
            "reward_or_classifier_version": 5,
            "global_step": 0,
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
    traces: list[str] = []
    monkeypatch.setattr(trajectory_env_worker, "_hs_trace", traces.append)
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

        assert channels["env"].puts[0][0] == "0:0"
        assert isinstance(channels["env"].puts[0][1], ObservationMsg)
        assert channels["env"].puts[-1][1].obs["_final_bootstrap"] is True
        assert channels["env"].puts[-1][0] == "0:0"
        assert channels["rollout"].gets == ["0:0", "0:0"]
        assert channels["actor"].puts[0][0] == "default"
        assert channels["actor"].put_no_wait_calls[0][0] == "default"
        assert isinstance(channels["actor"].puts[0][1], TrajectoryShard)
        assert metrics["env/chunk_steps"] == 1.0
        assert metrics["env/physical_steps"] == 2.0
        assert metrics["env/steps"] == 2.0
        assert metrics["env/real_env/chunk_steps"] == 1.0
        assert metrics["env/real_env/steps"] == 2.0
        assert metrics["env/trajectory_shards"] == 1.0
        assert metrics["env/episodes_completed"] == 1.0
        assert metrics["env/final_bootstrap_requests"] == 1.0
        assert metrics["env/channel_put_obs_s"] >= 0.0
        assert metrics["env/rollout_get_s"] >= 0.0
        assert metrics["env/apply_step_s"] >= 0.0
        assert metrics["env/actor_put_s"] >= 0.0
        assert metrics["env/actor_put_flush_s"] >= 0.0
        assert metrics["env/interact_loop_s"] >= 0.0
        assert metrics["env/real_env/channel_put_obs_s"] >= 0.0
        assert metrics["env/real_env/rollout_get_s"] >= 0.0
        assert metrics["env/real_env/apply_step_s"] >= 0.0
        assert metrics["env/real_env/actor_put_s"] >= 0.0
        assert metrics["env/real_env/actor_put_flush_s"] >= 0.0
        assert metrics["env/real_env/interact_loop_s"] >= 0.0
        assert any("[env rank=0 role=real_env] reset start" in line for line in traces)
        assert any("[env rank=0 role=real_env] reset done" in line for line in traces)
        assert any(
            "[env rank=0 role=real_env] send action request batch_size=1 key=0:0"
            in line
            for line in traces
        )
        assert any(
            "[env rank=0 role=real_env] recv action response batch_size=1 key=0:0"
            in line
            for line in traces
        )
        assert any(
            "[env rank=0 role=real_env] step 0 start key=0:0" in line
            for line in traces
        )
        assert any(
            "[env rank=0 role=real_env] step 0 done key=0:0" in line
            for line in traces
        )
    finally:
        worker.close()


def test_interact_buffers_chunks_until_complete_trajectory(monkeypatch) -> None:
    worker = RealEnvWorker(
        env_cfg=_long_horizon_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=4,
        num_action_chunks=2,
        task_id=0,
        request_final_bootstrap=False,
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

        assert len(channels["actor"].puts) == 1
        shard = channels["actor"].puts[0][1]
        assert isinstance(shard, TrajectoryShard)
        assert shard.actions.shape == (2, 1, 2, 3)
        assert shard.prev_logprobs.shape == (2, 1)
        assert shard.rewards.shape == (2, 1, 2)
        assert metrics["env/chunk_steps"] == 2.0
        assert metrics["env/trajectory_shards"] == 1.0
        assert metrics["env/final_bootstrap_requests"] == 0.0
    finally:
        worker.close()


def test_wm_env_worker_does_not_write_imagined_rollouts_to_replay_by_default(
    monkeypatch,
) -> None:
    replay = _MemoryReplay()
    worker = WMEnvWorker(
        env_cfg=_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
        replay=replay,
        request_final_bootstrap=False,
    )
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel([_rollout_result()]),
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

        assert metrics["env/episodes_completed"] == 1.0
        assert len(replay.episodes) == 0
        assert len(channels["actor"].puts) == 1
    finally:
        worker.close()


def test_wm_env_worker_batches_slots_with_step_batch(monkeypatch) -> None:
    class _BatchOnlyWMEnv:
        num_envs = 2

        def __init__(self) -> None:
            self.step_i = [0, 0]
            self.batch_calls: list[tuple[list[int], tuple[int, ...]]] = []

        def reset_slot(self, slot_id: int, *, task_id: int = 0, episode_id: int = 0):
            self.step_i[int(slot_id)] = 0
            return self._obs(int(slot_id), task_id, episode_id, is_first=True), {}

        def step_slot(self, slot_id: int, action):
            raise AssertionError("WMEnvWorker should call step_batch for batched WM envs")

        def step_batch(self, actions, env_ids=None):
            slots = [int(v) for v in env_ids]
            action_arr = np.asarray(actions, dtype=np.float32)
            self.batch_calls.append((slots, tuple(action_arr.shape)))
            observations, rewards, terminations, truncations, infos = [], [], [], [], []
            for slot_id in slots:
                self.step_i[slot_id] += 1
                done = self.step_i[slot_id] >= 2
                observations.append(
                    self._obs(
                        slot_id,
                        task_id=0,
                        episode_id=slot_id,
                        is_first=False,
                    )
                )
                rewards.append(float(done))
                terminations.append(bool(done))
                truncations.append(False)
                infos.append(
                    {
                        "success": bool(done),
                        "wm_action": action_arr[slots.index(slot_id)],
                    }
                )
            return observations, rewards, terminations, truncations, infos

        def make_transition(
            self,
            obs,
            action,
            reward,
            terminated,
            truncated,
            info,
        ):
            done = bool(terminated or truncated)
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            return {
                "image": np.zeros((1, 1, 3), dtype=np.uint8),
                "state": np.asarray(obs["state"], dtype=np.float32),
                "obs_embedding": np.asarray(obs["obs_embedding"], dtype=np.float32),
                "action": action_arr,
                "wm_action": np.asarray(info["wm_action"], dtype=np.float32),
                "policy_action": action_arr,
                "reward": np.float32(reward),
                "done": np.float32(done),
                "discount": np.float32(0.0 if terminated else 1.0),
                "is_first": bool(obs.get("is_first", False)),
                "is_terminal": bool(terminated),
                "is_last": bool(done),
                "task_id": int(obs["task_id"]),
                "episode_id": int(obs["episode_id"]),
                "step": int(obs["step"]),
                "task_description": str(obs["task_description"]),
                "success": bool(info.get("success", False)),
            }

        def get_metrics(self, *, reset: bool = False):
            del reset
            return {
                "model_forwards": len(self.batch_calls),
                "wm_forward_calls": len(self.batch_calls),
            }

        def _obs(self, slot_id: int, task_id: int, episode_id: int, *, is_first: bool):
            value = self.step_i[slot_id]
            return {
                "image": np.zeros((1, 1, 3), dtype=np.uint8),
                "state": np.full((2,), value, dtype=np.float32),
                "obs_embedding": np.full((4,), value, dtype=np.float32),
                "task_id": int(task_id),
                "episode_id": int(episode_id),
                "step": int(value),
                "task_description": f"task {task_id}",
                "is_first": bool(is_first),
            }

    env = _BatchOnlyWMEnv()
    monkeypatch.setattr(
        trajectory_env_worker,
        "_build_env_from_cfg",
        lambda _cfg: env,
    )
    worker = WMEnvWorker(
        env_cfg={"target": "unused"},
        num_slots=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
    )
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel(
            [
                _rollout_result_for_slot(0),
                _rollout_result_for_slot(1),
                _rollout_result_for_slot(0),
                _rollout_result_for_slot(1),
            ]
        ),
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

        assert env.batch_calls == [([0, 1], (2, 3)), ([0, 1], (2, 3))]
        assert metrics["env/wm_env/chunk_steps"] == 2.0
        assert metrics["env/wm_env/steps"] == 4.0
        assert metrics["env/wm_env/model_forwards"] == 2.0
        assert metrics["env/wm_env/channel_put_obs_s"] >= 0.0
        assert metrics["env/wm_env/rollout_get_s"] >= 0.0
        assert metrics["env/wm_env/apply_step_s"] >= 0.0
        assert metrics["env/wm_env/actor_put_s"] >= 0.0
        assert metrics["env/wm_env/actor_put_flush_s"] >= 0.0
        assert metrics["env/wm_env/interact_loop_s"] >= 0.0
        assert len(channels["actor"].puts) == 2
        assert len(channels["actor"].put_no_wait_calls) == 2
        assert all(isinstance(item, TrajectoryShard) for _key, item in channels["actor"].puts)
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
