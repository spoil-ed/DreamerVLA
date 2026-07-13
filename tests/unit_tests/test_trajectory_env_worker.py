from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

import dreamervla.workers.env.trajectory_env_worker as trajectory_env_worker
from dreamervla.workers.cotrain.messages import (
    ObservationBatchMsg,
    ObservationMsg,
    RolloutResultBatchMsg,
    RolloutResultMsg,
    TrajectoryShard,
)
from dreamervla.workers.env._test_envs import CounterEnv
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


def _tiny_wm_env_cfg() -> dict[str, Any]:
    return {
        "target": "dreamervla.envs.world_model.latent_world_model_env:LatentWorldModelEnv",
        "kwargs": {
            "world_model": {
                "target": "dreamervla.workers.actor._test_models:TinyLumosWorldModel",
                "kwargs": {"hidden_dim": 4, "action_dim": 3},
            },
            "classifier": {
                "target": "dreamervla.workers.actor._test_models:TinySuccessClassifier",
                "kwargs": {"hidden_dim": 4, "window": 3},
            },
            "latent_dim": 4,
            "action_dim": 3,
            "success_threshold": 0.5,
            "num_envs": 1,
        },
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


def _rollout_batch(*results: RolloutResultMsg, env_rank: int = 0) -> RolloutResultBatchMsg:
    return RolloutResultBatchMsg(env_rank=int(env_rank), results=list(results))


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

        result = _rollout_result()
        result = replace(
            result,
            forward_inputs={
                **result.forward_inputs,
                "lang_emb": np.full((2,), 3.0, dtype=np.float32),
            },
        )
        shard = worker.apply_rollout_result(result)

        assert isinstance(shard, TrajectoryShard)
        assert shard.actions.shape == (1, 1, 2, 3)
        assert shard.forward_inputs["hidden"].shape == (1, 1, 4)
        assert shard.forward_inputs["lang_emb"].shape == (1, 1, 2)
        assert shard.forward_inputs["lang_emb"][0, 0].tolist() == [3.0, 3.0]
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


def test_wm_env_worker_applies_classifier_threshold_from_component_state() -> None:
    worker = WMEnvWorker(
        env_cfg=_tiny_wm_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=1,
        task_id=0,
    )
    try:
        worker.init()

        assert worker.envs[0].success_threshold == 0.5

        worker.load_component_states(
            {
                "classifier": {},
                "classifier_threshold": 0.95,
            },
            version=3,
        )

        assert worker.envs[0].success_threshold == 0.95
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


def test_real_env_worker_pins_egl_for_inproc_slots(monkeypatch) -> None:
    cfg = dict(_counter_env_cfg())
    cfg["render_backend"] = "egl"
    cfg["render_devices"] = [0, 1]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
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

    monkeypatch.setattr(
        trajectory_env_worker,
        "_build_env_from_cfg",
        lambda cfg_arg: build_calls.append(dict(cfg_arg)) or original_build_env_from_cfg(cfg_arg),
    )

    try:
        worker.init()

        assert len(build_calls) == 2
        assert os.environ["MUJOCO_GL"] == "egl"
        assert os.environ["PYOPENGL_PLATFORM"] == "egl"
        assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "1"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert len(worker.envs) == 2
        assert not hasattr(worker, "_spawned_env")
    finally:
        worker.close()


def test_real_env_worker_spawn_env_slots_uses_child_process() -> None:
    cfg = dict(_counter_env_cfg())
    cfg["render_backend"] = "osmesa"
    cfg["spawn_env_slots"] = True
    replay = _MemoryReplay()
    worker = RealEnvWorker(
        env_cfg=cfg,
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
        replay=replay,
    )

    try:
        worker.init()
        messages = worker.bootstrap_obs()
        assert len(messages) == 1
        assert type(worker.envs[0]).__name__ == "_SpawnedTrajectoryEnvSlot"

        worker.apply_rollout_result(
            replace(_rollout_result(), actions=np.zeros((1, 3), dtype=np.float32))
        )

        assert len(replay.episodes) == 0
    finally:
        worker.close()


def test_real_env_worker_pins_osmesa_for_inproc_non_egl_backend(monkeypatch) -> None:
    cfg = dict(_counter_env_cfg())
    cfg["render_backend"] = "osmesa"
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
        assert not hasattr(worker, "_spawned_env")
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


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_real_env_worker_rejects_nonfinite_policy_actions_before_env_step(
    bad_value: float,
) -> None:
    worker = RealEnvWorker(
        env_cfg=_no_sidecar_oft_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )
    policy_action = np.array(
        [[0.1, 0.2, 0.3, bad_value, 0.5, 0.6, 0.25]],
        dtype=np.float32,
    )
    try:
        worker.init()
        worker.bootstrap_obs()

        with pytest.raises(ValueError, match="non-finite policy action"):
            worker.apply_rollout_result(_sidecar_rollout_result(policy_action))

        assert worker.envs[0].received_actions == []
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
        assert worker._last_apply_successful_episodes == 1
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
        assert worker.envs[0].reset_batch_calls == 1
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


def test_observation_batch_msg_moves_hidden_to_batched_payload() -> None:
    worker = WMEnvWorker(
        env_cfg=_no_sidecar_env_cfg(),
        num_slots=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )
    messages = [
        ObservationMsg(
            env_rank=0,
            slot_id=0,
            task_id=0,
            episode_id=10,
            step=0,
            obs={
                "obs_embedding": np.ones(4, dtype=np.float32),
                "lang_emb": np.full(2, 3.0, dtype=np.float32),
                "task_description": "task 0",
            },
            versions={"policy": 0},
        ),
        ObservationMsg(
            env_rank=0,
            slot_id=1,
            task_id=0,
            episode_id=11,
            step=0,
            obs={
                "obs_embedding": np.full(4, 2.0, dtype=np.float32),
                "lang_emb": np.full(2, 4.0, dtype=np.float32),
                "task_description": "task 0",
            },
            versions={"policy": 0},
        ),
    ]

    batch = worker._observation_batch_msg(messages)

    assert batch.batched_obs is not None
    assert batch.batched_obs["obs_embedding"].shape == (2, 4)
    assert batch.batched_obs["obs_embedding"].tolist() == [
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
    ]
    assert batch.batched_obs["lang_emb"].shape == (2, 2)
    assert batch.batched_obs["lang_emb"].tolist() == [
        [3.0, 3.0],
        [4.0, 4.0],
    ]
    assert all("obs_embedding" not in msg.obs for msg in batch.observations)
    assert all("lang_emb" not in msg.obs for msg in batch.observations)
    assert batch.observations[0].obs["task_description"] == "task 0"


def test_observation_batch_msg_preserves_bfloat16_tensor_payload() -> None:
    worker = WMEnvWorker(
        env_cfg=_no_sidecar_env_cfg(),
        num_slots=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )
    messages = [
        ObservationMsg(
            env_rank=0,
            slot_id=0,
            task_id=0,
            episode_id=10,
            step=0,
            obs={
                "obs_embedding": torch.ones(4, dtype=torch.bfloat16),
                "task_description": "task 0",
            },
            versions={"policy": 0},
        ),
        ObservationMsg(
            env_rank=0,
            slot_id=1,
            task_id=0,
            episode_id=11,
            step=0,
            obs={
                "obs_embedding": torch.full((4,), 2.0, dtype=torch.bfloat16),
                "task_description": "task 0",
            },
            versions={"policy": 0},
        ),
    ]

    batch = worker._observation_batch_msg(messages)

    assert batch.batched_obs is not None
    assert isinstance(batch.batched_obs["obs_embedding"], torch.Tensor)
    assert batch.batched_obs["obs_embedding"].dtype == torch.bfloat16
    assert batch.batched_obs["obs_embedding"].shape == (2, 4)


def test_same_storage_observation_rows_reuse_batched_tensor_view() -> None:
    source = torch.arange(12, dtype=torch.bfloat16).reshape(2, 2, 3)

    batched = trajectory_env_worker._batch_same_shape_obs_values(
        [source[0], source[1]]
    )

    assert isinstance(batched, torch.Tensor)
    assert batched.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    torch.testing.assert_close(batched, source)


def test_real_env_interact_routes_observations_and_replay_without_actor_trajectory(
    monkeypatch,
) -> None:
    traces: list[str] = []
    monkeypatch.setattr(trajectory_env_worker, "_hs_trace", traces.append)
    replay = _MemoryReplay()
    worker = RealEnvWorker(
        env_cfg=_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
        replay=replay,
    )
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel([
            _rollout_batch(_rollout_result()),
            _rollout_batch(_rollout_result()),
        ]),
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

        assert channels["env"].puts[0][0] == "0"
        assert isinstance(channels["env"].puts[0][1], ObservationBatchMsg)
        assert len(channels["env"].puts[0][1].observations) == 1
        assert channels["env"].puts[-1][1].observations[0].obs["_final_bootstrap"] is True
        assert channels["env"].puts[-1][0] == "0"
        assert channels["rollout"].gets == ["0", "0"]
        assert channels["actor"].puts == []
        assert channels["actor"].put_no_wait_calls == []
        assert len(replay.episodes) == 1
        assert metrics["env/chunk_steps"] == 1.0
        assert metrics["env/physical_steps"] == 2.0
        assert metrics["env/steps"] == 2.0
        assert metrics["env/real_env/chunk_steps"] == 1.0
        assert metrics["env/real_env/steps"] == 2.0
        assert metrics["env/trajectory_shards"] == 0.0
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
            "[env rank=0 role=real_env] send action request batch_size=1 key=0"
            in line
            for line in traces
        )
        assert any(
            "[env rank=0 role=real_env] recv action response batch_size=1 key=0"
            in line
            for line in traces
        )
        assert any(
            "[env rank=0 role=real_env] step 0 start batch_size=1 keys=0:0" in line
            for line in traces
        )
        assert any(
            "[env rank=0 role=real_env] step 0 done batch_size=1 keys=0:0" in line
            for line in traces
        )
    finally:
        worker.close()


def test_real_env_exact_budget_stops_slot_after_first_trajectory(monkeypatch) -> None:
    replay = _MemoryReplay()
    env_cfg = _short_horizon_counter_env_cfg()
    env_cfg["one_trajectory_per_rollout_epoch"] = True
    worker = RealEnvWorker(
        env_cfg=env_cfg,
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=4,
        num_action_chunks=2,
        task_id=0,
        replay=replay,
        request_final_bootstrap=False,
    )
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel([_rollout_batch(_rollout_result())]),
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

        assert metrics["env/chunk_steps"] == 1.0
        assert metrics["env/episodes_completed"] == 1.0
        assert len(replay.episodes) == 1
        assert channels["rollout"].gets == ["0"]
    finally:
        worker.close()


def test_real_env_interact_can_emit_actor_trajectory_when_configured(
    monkeypatch,
) -> None:
    env_cfg = _counter_env_cfg()
    env_cfg["emit_actor_trajectories"] = True
    worker = RealEnvWorker(
        env_cfg=env_cfg,
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
        request_final_bootstrap=False,
    )
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel([
            _rollout_batch(_rollout_result()),
            _rollout_batch(_rollout_result()),
        ]),
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
        assert channels["actor"].puts[0][0] == "real_env"
        assert isinstance(channels["actor"].puts[0][1], TrajectoryShard)
        assert metrics["env/trajectory_shards"] == 1.0
        assert metrics["env/real_env/trajectory_shards"] == 1.0
    finally:
        worker.close()


def test_prefetch_bootstrap_lets_interact_skip_inline_reset(monkeypatch) -> None:
    worker = RealEnvWorker(
        env_cfg=_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
        replay=_MemoryReplay(),
    )
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel([
            _rollout_batch(_rollout_result()),
            _rollout_batch(_rollout_result()),
        ]),
        "actor": _MemoryChannel(),
    }
    monkeypatch.setattr(
        trajectory_env_worker.Channel,
        "connect",
        staticmethod(lambda name: channels[str(name)]),
    )
    try:
        worker.init()
        original_bootstrap = worker.bootstrap_obs
        calls = {"n": 0}

        def counting_bootstrap():
            calls["n"] += 1
            return original_bootstrap()

        monkeypatch.setattr(worker, "bootstrap_obs", counting_bootstrap)

        worker.prefetch_bootstrap()
        assert calls["n"] == 1  # prefetch performed the slot reset
        worker.prefetch_bootstrap()
        assert calls["n"] == 1  # idempotent: cache already populated

        calls["n"] = 0
        metrics = worker.interact("env", "rollout", "actor")
        assert calls["n"] == 0  # interact consumed the prefetched batch
        assert metrics["env/chunk_steps"] == 1.0
    finally:
        worker.close()


def test_interact_buffers_chunks_until_complete_trajectory(monkeypatch) -> None:
    worker = WMEnvWorker(
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
        "rollout": _MemoryChannel([
            _rollout_batch(_rollout_result()),
            _rollout_batch(_rollout_result()),
        ]),
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


def test_interact_writes_manual_cotrain_progress_file(monkeypatch, tmp_path) -> None:
    worker = RealEnvWorker(
        env_cfg=_long_horizon_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=4,
        num_action_chunks=2,
        task_id=0,
        request_final_bootstrap=False,
    )
    worker.set_global_step(3)
    worker.configure_progress(str(tmp_path), min_interval_s=0.0)
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel([
            _rollout_batch(_rollout_result()),
            _rollout_batch(_rollout_result()),
        ]),
        "actor": _MemoryChannel(),
    }
    monkeypatch.setattr(
        trajectory_env_worker.Channel,
        "connect",
        staticmethod(lambda name: channels[str(name)]),
    )

    try:
        worker.init()
        worker.interact("env", "rollout", "actor")

        payload = json.loads((tmp_path / "real_env_0.json").read_text(encoding="utf-8"))
        assert payload["role"] == "real_env"
        assert payload["rank"] == 0
        assert payload["env_rank"] == 0
        assert payload["global_step"] == 3
        assert payload["done"] == 2
        assert payload["total"] == 2
        assert payload["active"] is False
        assert payload["finished"] is True
    finally:
        worker.close()


def test_real_env_action_diagnostics_do_not_force_progress_by_default(tmp_path) -> None:
    worker = RealEnvWorker(
        env_cfg=_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=1,
        task_id=0,
    )
    worker.set_global_step(3)
    worker.configure_progress(str(tmp_path), min_interval_s=999.0)

    worker._record_action_diagnostics(  # noqa: SLF001 - verifies progress throttling
        0,
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([0.4, 0.5, 0.6], dtype=np.float32),
    )

    payload = json.loads((tmp_path / "real_env_0.json").read_text(encoding="utf-8"))
    assert "last_action" not in payload


def test_real_env_action_diagnostics_can_force_progress_when_debug_enabled(
    tmp_path,
) -> None:
    cfg = _counter_env_cfg()
    cfg["debug_action_diagnostics"] = True
    worker = RealEnvWorker(
        env_cfg=cfg,
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=1,
        task_id=0,
    )
    worker.set_global_step(3)
    worker.configure_progress(str(tmp_path), min_interval_s=999.0)

    worker._record_action_diagnostics(  # noqa: SLF001 - verifies debug diagnostics
        0,
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([0.4, 0.5, 0.6], dtype=np.float32),
    )

    payload = json.loads((tmp_path / "real_env_0.json").read_text(encoding="utf-8"))
    assert payload["last_action"]["policy_absmax"] == pytest.approx(0.3)
    assert payload["last_action"]["env_absmax"] == pytest.approx(0.6)


def test_wm_interact_progress_file_includes_classifier_success_counters(
    monkeypatch,
    tmp_path,
) -> None:
    worker = WMEnvWorker(
        env_cfg=_short_horizon_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=4,
        num_action_chunks=2,
        task_id=0,
        request_final_bootstrap=False,
    )
    worker.set_global_step(3)
    worker.configure_progress(str(tmp_path), min_interval_s=0.0)
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel([
            _rollout_batch(_rollout_result()),
            _rollout_batch(_rollout_result()),
        ]),
        "actor": _MemoryChannel(),
    }
    monkeypatch.setattr(
        trajectory_env_worker.Channel,
        "connect",
        staticmethod(lambda name: channels[str(name)]),
    )

    try:
        worker.init()
        worker.interact("env", "rollout", "actor")

        payload = json.loads((tmp_path / "wm_env_0.json").read_text(encoding="utf-8"))
        assert payload["classifier_total_chunks"] > 0
        assert payload["classifier_success_chunks"] > 0
        assert payload["classifier_total_trajectories"] > 0
        assert payload["classifier_success_trajectories"] > 0
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
        "rollout": _MemoryChannel([_rollout_batch(_rollout_result())]),
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


def test_wm_env_worker_skips_transition_building_without_episode_sinks(
    monkeypatch,
) -> None:
    worker = WMEnvWorker(
        env_cfg=_counter_env_cfg(),
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
        request_final_bootstrap=False,
    )
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel([_rollout_batch(_rollout_result())]),
        "actor": _MemoryChannel(),
    }

    def fail_transition(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("WMEnvWorker should not build unused imagined episodes")

    monkeypatch.setattr(
        trajectory_env_worker.BaseTrajectoryEnvWorker,
        "_make_transition",
        staticmethod(fail_transition),
    )
    monkeypatch.setattr(
        trajectory_env_worker.Channel,
        "connect",
        staticmethod(lambda name: channels[str(name)]),
    )

    try:
        worker.init()
        metrics = worker.interact("env", "rollout", "actor")

        assert metrics["env/episodes_completed"] == 1.0
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
                _rollout_batch(
                    _rollout_result_for_slot(0),
                    _rollout_result_for_slot(1),
                ),
                _rollout_batch(
                    _rollout_result_for_slot(0),
                    _rollout_result_for_slot(1),
                ),
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
        assert metrics["env/wm_env/classifier_success_chunks"] == 2.0
        assert metrics["env/wm_env/classifier_total_chunks"] == 4.0
        assert metrics["env/wm_env/classifier_success_rate"] == 0.5
        assert metrics["env/wm_env/classifier_success_trajectories"] == 2.0
        assert metrics["env/wm_env/classifier_total_trajectories"] == 2.0
        assert metrics["env/wm_env/classifier_trajectory_success_rate"] == 1.0
        assert len(channels["actor"].puts) == 1
        assert len(channels["actor"].put_no_wait_calls) == 1
        key, shard = channels["actor"].puts[0]
        assert key == "wm_env"
        assert isinstance(shard, TrajectoryShard)
        assert shard.actions.shape == (1, 2, 2, 3)
        assert shard.rewards.shape == (1, 2, 2)
        assert shard.loss_mask is not None
        assert shard.loss_mask.shape == (1, 2)
    finally:
        worker.close()


def test_wm_env_worker_prefers_chunk_step_batch_when_available(monkeypatch) -> None:
    class _ChunkBatchWMEnv:
        num_envs = 2
        success_threshold = 0.5

        def __init__(self) -> None:
            self.step_i = [0, 0]
            self.chunk_calls: list[tuple[list[int], tuple[int, ...]]] = []

        def reset_slot(self, slot_id: int, *, task_id: int = 0, episode_id: int = 0):
            self.step_i[int(slot_id)] = 0
            return self._obs(int(slot_id), task_id, episode_id, is_first=True), {}

        def step_slot(self, slot_id: int, action):
            raise AssertionError("WMEnvWorker should use chunk_step_batch")

        def step_batch(self, actions, env_ids=None):
            raise AssertionError("WMEnvWorker should use chunk_step_batch")

        def chunk_step_batch(self, actions, env_ids=None):
            slots = [int(v) for v in env_ids]
            action_arr = np.asarray(actions, dtype=np.float32)
            self.chunk_calls.append((slots, tuple(action_arr.shape)))
            rewards = np.zeros((len(slots), action_arr.shape[1]), dtype=np.float32)
            terminations = np.zeros_like(rewards, dtype=np.bool_)
            truncations = np.zeros_like(rewards, dtype=np.bool_)
            observations = []
            infos = []
            for batch_index, slot_id in enumerate(slots):
                self.step_i[slot_id] += int(action_arr.shape[1])
                if batch_index == 0:
                    rewards[batch_index, -1] = 0.75
                    terminations[batch_index, -1] = True
                observations.append(
                    self._obs(
                        slot_id,
                        task_id=0,
                        episode_id=slot_id,
                        is_first=False,
                    )
                )
                infos.append(
                    {
                        "success": bool(rewards[batch_index].max() >= self.success_threshold),
                        "classifier_evaluations": 1,
                        "classifier_success_evaluations": int(
                            rewards[batch_index].max() >= self.success_threshold
                        ),
                        "wm_action": action_arr[batch_index, -1],
                    }
                )
            return observations, rewards, terminations, truncations, infos

        def get_metrics(self, *, reset: bool = False):
            del reset
            return {
                "model_forwards": len(self.chunk_calls),
                "wm_forward_calls": len(self.chunk_calls),
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

    env = _ChunkBatchWMEnv()
    monkeypatch.setattr(
        trajectory_env_worker,
        "_build_env_from_cfg",
        lambda _cfg: env,
    )
    worker = WMEnvWorker(
        env_cfg={"target": "unused"},
        num_slots=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=4,
        num_action_chunks=2,
        task_id=0,
        request_final_bootstrap=False,
    )
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel(
            [
                _rollout_batch(
                    _rollout_result_for_slot(0),
                    _rollout_result_for_slot(1),
                ),
                _rollout_batch(
                    _rollout_result_for_slot(0),
                    _rollout_result_for_slot(1),
                ),
            ]
        ),
        "actor": _MemoryChannel(),
    }
    monkeypatch.setattr(
        trajectory_env_worker.Channel,
        "connect",
        staticmethod(lambda name: channels[str(name)]),
    )
    monkeypatch.setattr(
        trajectory_env_worker.BaseTrajectoryEnvWorker,
        "_build_trajectory_shard",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("chunk_step_batch results should materialize at flush")
        ),
    )
    monkeypatch.setattr(
        trajectory_env_worker.BaseTrajectoryEnvWorker,
        "_build_trajectory_shard_from_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("worker slot chunks should materialize as one batch")
        ),
    )
    monkeypatch.setattr(
        trajectory_env_worker.BaseTrajectoryEnvWorker,
        "_env_action_from_policy_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no-op action postprocess should pass chunks directly")
        ),
    )
    monkeypatch.setattr(
        trajectory_env_worker.np,
        "flatnonzero",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("WoVR-style final-column chunks should not scan per slot")
        ),
    )
    monkeypatch.setattr(
        trajectory_env_worker.np,
        "stack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("worker chunk materialization should preallocate arrays")
        ),
    )
    monkeypatch.setattr(
        trajectory_env_worker.torch,
        "stack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("worker chunk materialization should stack through numpy")
        ),
    )

    try:
        worker.init()
        metrics = worker.interact("env", "rollout", "actor")

        assert env.chunk_calls == [([0, 1], (2, 2, 3)), ([0, 1], (2, 2, 3))]
        assert metrics["env/wm_env/model_forwards"] == 2.0
        assert metrics["env/wm_env/classifier_success_chunks"] == 2.0
        assert metrics["env/wm_env/classifier_total_chunks"] == 4.0
        assert metrics["env/wm_env/classifier_success_rate"] == 0.5
        assert metrics["env/wm_env/classifier_success_trajectories"] == 2.0
        assert metrics["env/wm_env/classifier_total_trajectories"] == 4.0
        assert metrics["env/wm_env/classifier_trajectory_success_rate"] == 0.5
        assert len(channels["actor"].puts) == 1
        _key, shard = channels["actor"].puts[0]
        assert isinstance(shard, TrajectoryShard)
        assert shard.actions.shape == (2, 2, 2, 3)
        assert shard.rewards.shape == (2, 2, 2)
        assert shard.loss_mask is not None
        assert shard.loss_mask.shape == (2, 2)
        assert shard.forward_inputs["action"].data_ptr() == shard.actions.data_ptr()
    finally:
        worker.close()


def test_wm_env_worker_batches_openvla_oft_action_postprocess(monkeypatch) -> None:
    class _ChunkBatchWMEnv:
        num_envs = 2

        def __init__(self) -> None:
            self.chunk_actions: list[np.ndarray] = []

        def reset_slot(self, slot_id: int, *, task_id: int = 0, episode_id: int = 0):
            return self._obs(int(slot_id), task_id, episode_id), {}

        def step_slot(self, slot_id: int, action):
            raise AssertionError("WMEnvWorker should use chunk_step_batch")

        def step_batch(self, actions, env_ids=None):
            raise AssertionError("WMEnvWorker should use chunk_step_batch")

        def chunk_step_batch(self, actions, env_ids=None):
            del env_ids
            action_arr = np.asarray(actions, dtype=np.float32)
            self.chunk_actions.append(action_arr.copy())
            batch_size, chunk_len, _action_dim = action_arr.shape
            rewards = np.zeros((batch_size, chunk_len), dtype=np.float32)
            dones = np.zeros((batch_size, chunk_len), dtype=np.bool_)
            observations = [self._obs(slot_id, 0, slot_id) for slot_id in range(batch_size)]
            infos = [{"wm_action": action_arr[index, -1]} for index in range(batch_size)]
            return observations, rewards, dones, dones.copy(), infos

        def get_metrics(self, *, reset: bool = False):
            del reset
            return {"model_forwards": len(self.chunk_actions)}

        def _obs(self, slot_id: int, task_id: int, episode_id: int):
            return {
                "image": np.zeros((1, 1, 3), dtype=np.uint8),
                "state": np.zeros((2,), dtype=np.float32),
                "obs_embedding": np.full((4,), float(slot_id), dtype=np.float32),
                "task_id": int(task_id),
                "episode_id": int(episode_id),
                "step": 0,
                "task_description": f"task {task_id}",
            }

    def result_for_slot(slot_id: int, grippers: tuple[float, float]) -> RolloutResultMsg:
        actions = np.zeros((2, 7), dtype=np.float32)
        actions[:, :6] = float(slot_id) + 0.1
        actions[:, -1] = np.asarray(grippers, dtype=np.float32)
        return RolloutResultMsg(
            env_rank=0,
            slot_id=int(slot_id),
            task_id=0,
            episode_id=int(slot_id),
            step=0,
            actions=actions,
            prev_logprobs=np.array([0.0], dtype=np.float32),
            prev_values=None,
            forward_inputs={
                "hidden": np.full((1, 4), float(slot_id), dtype=np.float32),
                "action": actions.reshape(1, 2, 7),
            },
            versions={"policy": 1},
        )

    env = _ChunkBatchWMEnv()
    monkeypatch.setattr(
        trajectory_env_worker,
        "_build_env_from_cfg",
        lambda _cfg: env,
    )
    monkeypatch.setattr(
        trajectory_env_worker.BaseTrajectoryEnvWorker,
        "_env_action_from_policy_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("openvla_oft chunks should use vectorized postprocess")
        ),
    )
    worker = WMEnvWorker(
        env_cfg={"target": "unused", "action_postprocess": "openvla_oft"},
        num_slots=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=4,
        num_action_chunks=2,
        task_id=0,
        request_final_bootstrap=False,
    )
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel(
            [
                _rollout_batch(
                    result_for_slot(0, (0.25, 0.75)),
                    result_for_slot(1, (0.49, 0.51)),
                ),
                _rollout_batch(
                    result_for_slot(0, (0.25, 0.75)),
                    result_for_slot(1, (0.49, 0.51)),
                ),
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
        worker.interact("env", "rollout", "actor")

        assert len(env.chunk_actions) == 2
        np.testing.assert_allclose(env.chunk_actions[0][0, :, -1], [1.0, -1.0])
        np.testing.assert_allclose(env.chunk_actions[0][1, :, -1], [1.0, -1.0])
        np.testing.assert_allclose(env.chunk_actions[0][0, :, :6], 0.1)
        np.testing.assert_allclose(env.chunk_actions[0][1, :, :6], 1.1)
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
        "rollout": _MemoryChannel([
            _rollout_batch(one_step),
            _rollout_batch(one_step),
        ]),
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


def test_get_rollout_result_batch_injects_hidden_from_slot_obs() -> None:
    worker = BaseTrajectoryEnvWorker(
        role="wm_env",
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
        batch = RolloutResultBatchMsg(
            env_rank=0,
            results=[],
            slot_ids=[0],
            task_ids=[0],
            episode_ids=[0],
            steps=[0],
            actions=torch.zeros(1, 2, 3),
            prev_logprobs=torch.zeros(1, 1),
            prev_values=None,
            forward_inputs={"action": torch.zeros(1, 2, 3)},
            versions={"policy": torch.ones(1, dtype=torch.long)},
        )
        channel = _MemoryChannel([batch])
        metrics = worker._new_interact_metrics()

        results = worker._get_rollout_result_batch(channel, [0], metrics)

        hidden = torch.as_tensor(results[0].forward_inputs["hidden"])
        expected = torch.as_tensor(
            np.asarray(worker._obs_by_slot[0]["obs_embedding"], dtype=np.float32)
        ).reshape(1, -1)
        assert hidden.shape == expected.shape
        assert torch.allclose(hidden, expected)
    finally:
        worker.close()


def test_get_rollout_result_batch_preserves_token_grid_for_actor() -> None:
    worker = BaseTrajectoryEnvWorker(
        role="wm_env",
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
        worker._obs_by_slot[0]["obs_embedding"] = torch.arange(6).reshape(2, 3)
        batch = RolloutResultBatchMsg(
            env_rank=0,
            results=[],
            slot_ids=[0],
            task_ids=[0],
            episode_ids=[0],
            steps=[0],
            actions=torch.zeros(1, 2, 3),
            prev_logprobs=torch.zeros(1, 1),
            prev_values=None,
            forward_inputs={"action": torch.zeros(1, 2, 3)},
            versions={"policy": torch.ones(1, dtype=torch.long)},
        )
        channel = _MemoryChannel([batch])

        results = worker._get_rollout_result_batch(
            channel,
            [0],
            worker._new_interact_metrics(),
        )

        assert results[0].forward_inputs["hidden"].shape == (1, 2, 3)
        shard = worker.apply_rollout_result(results[0])
        assert shard.forward_inputs["hidden"].shape == (1, 1, 2, 3)
    finally:
        worker.close()


def test_get_rollout_result_batch_injects_bf16_tensor_hidden() -> None:
    worker = BaseTrajectoryEnvWorker(
        role="wm_env",
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
        worker._obs_by_slot[0]["obs_embedding"] = torch.full(
            (4,), 2.0, dtype=torch.bfloat16
        )
        batch = RolloutResultBatchMsg(
            env_rank=0,
            results=[],
            slot_ids=[0],
            task_ids=[0],
            episode_ids=[0],
            steps=[0],
            actions=torch.zeros(1, 2, 3),
            prev_logprobs=torch.zeros(1, 1),
            prev_values=None,
            forward_inputs={"action": torch.zeros(1, 2, 3)},
            versions={"policy": torch.ones(1, dtype=torch.long)},
        )
        channel = _MemoryChannel([batch])
        metrics = worker._new_interact_metrics()

        results = worker._get_rollout_result_batch(channel, [0], metrics)

        hidden = torch.as_tensor(results[0].forward_inputs["hidden"])
        assert hidden.dtype == torch.bfloat16
        assert hidden.tolist() == [[2.0, 2.0, 2.0, 2.0]]
    finally:
        worker.close()


def test_worker_batched_trajectory_keeps_bf16_hidden_payload() -> None:
    worker = BaseTrajectoryEnvWorker(
        role="wm_env",
        env_cfg=_counter_env_cfg(),
        num_slots=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
    )

    def chunk(slot_id: int):
        result = _rollout_result_for_slot(slot_id)
        result.forward_inputs["hidden"] = torch.full(
            (1, 2, 3),
            float(slot_id + 1),
            dtype=torch.bfloat16,
        )
        return trajectory_env_worker._TrajectoryChunk(
            result=result,
            slot_id=slot_id,
            actions_np=np.zeros((2, 3), dtype=np.float32),
            rewards=np.zeros((2,), dtype=np.float32),
            dones=np.zeros((2,), dtype=np.bool_),
            action_dim=3,
        )

    shard = worker._build_worker_trajectory_shard_from_slot_chunks(
        [[chunk(0)], [chunk(1)]]
    )

    assert shard.forward_inputs["hidden"].shape == (1, 2, 2, 3)
    assert shard.forward_inputs["hidden"].dtype == torch.bfloat16


class _SplitCounterSlot(CounterEnv):
    """CounterEnv whose step is split into send/recv and logs its call order.

    Stands in for a spawned real-env slot: ``send_step`` dispatches the step and
    ``recv_step`` returns its result, so a driver can keep several slots' steps
    in flight at once. The shared ``call_log`` records the order to prove the
    stepping is lockstep (all sends before the matching recvs).
    """

    def __init__(
        self,
        *,
        slot_id: int,
        horizon: int,
        call_log: list[tuple[str, int]],
        embedding_dim: int = 4,
    ) -> None:
        super().__init__(horizon=horizon, embedding_dim=embedding_dim)
        self._slot_id = int(slot_id)
        self._call_log = call_log
        self._pending: np.ndarray | None = None

    def send_step(self, action: Any) -> None:
        self._call_log.append(("send", self._slot_id))
        self._pending = np.asarray(action, dtype=np.float32).copy()

    def recv_step(self) -> tuple[Any, ...]:
        self._call_log.append(("recv", self._slot_id))
        action = self._pending
        self._pending = None
        return super().step(action)


def _build_split_slot_worker(
    horizons: list[int],
    call_log: list[tuple[str, int]],
) -> BaseTrajectoryEnvWorker:
    worker = BaseTrajectoryEnvWorker(
        role="real_env",
        env_cfg=_long_horizon_counter_env_cfg(),
        num_slots=len(horizons),
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
        replay=_MemoryReplay(),
    )
    worker.init()
    for env in worker.envs:
        env.close()
    worker.envs = [
        _SplitCounterSlot(slot_id=i, horizon=h, call_log=call_log)
        for i, h in enumerate(horizons)
    ]
    worker._batched_env = False
    worker.bootstrap_obs()
    call_log.clear()
    return worker


def test_step_slots_parallel_is_lockstep_across_slots() -> None:
    call_log: list[tuple[str, int]] = []
    worker = _build_split_slot_worker([5, 5, 5, 5], call_log)
    try:
        results = [_rollout_result_for_slot(i) for i in range(4)]
        worker._step_slots_parallel(results)
        # Two physical steps; each scatters all four sends before gathering the
        # four recvs, so no subprocess idles while another steps.
        assert call_log == [
            ("send", 0),
            ("send", 1),
            ("send", 2),
            ("send", 3),
            ("recv", 0),
            ("recv", 1),
            ("recv", 2),
            ("recv", 3),
            ("send", 0),
            ("send", 1),
            ("send", 2),
            ("send", 3),
            ("recv", 0),
            ("recv", 1),
            ("recv", 2),
            ("recv", 3),
        ]
    finally:
        worker.close()


def test_step_slots_parallel_matches_serial_reference() -> None:
    # Varied horizons so some slots terminate mid-chunk and some do not.
    horizons = [5, 1, 3, 2]

    par_log: list[tuple[str, int]] = []
    parallel = _build_split_slot_worker(horizons, par_log)
    ser_log: list[tuple[str, int]] = []
    serial = _build_split_slot_worker(horizons, ser_log)
    try:
        accums = parallel._step_slots_parallel(
            [_rollout_result_for_slot(i) for i in range(4)]
        )
        parallel_shards = {i: parallel._finalize_accum(accums[i]) for i in range(4)}
        serial_shards = {
            i: serial.apply_rollout_result(_rollout_result_for_slot(i))
            for i in range(4)
        }

        for i in range(4):
            par = parallel_shards[i]
            ser = serial_shards[i]
            assert np.array_equal(par.actions.numpy(), ser.actions.numpy())
            assert np.array_equal(par.rewards.numpy(), ser.rewards.numpy())
            assert np.array_equal(par.dones.numpy(), ser.dones.numpy())
    finally:
        parallel.close()
        serial.close()


def test_step_slots_parallel_drops_terminated_slots() -> None:
    # Slot 0 terminates on the first physical step (horizon 1) and must drop out
    # of the active set for the second step, matching serial early-stop.
    call_log: list[tuple[str, int]] = []
    worker = _build_split_slot_worker([1, 5, 5, 5], call_log)
    try:
        accums = worker._step_slots_parallel(
            [_rollout_result_for_slot(i) for i in range(4)]
        )

        slot0_ops = [op for op in call_log if op[1] == 0]
        assert slot0_ops == [("send", 0), ("recv", 0)]

        sends = [op for op in call_log if op[0] == "send"]
        assert sends == [
            ("send", 0),
            ("send", 1),
            ("send", 2),
            ("send", 3),
            ("send", 1),
            ("send", 2),
            ("send", 3),
        ]
        assert accums[0].dones.tolist() == [True, True]
        assert not accums[0].active
    finally:
        worker.close()
