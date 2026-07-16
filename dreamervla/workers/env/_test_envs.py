"""Importable lightweight envs for Ray EnvWorker e2e tests."""

from __future__ import annotations

from typing import Any

import numpy as np


class CounterEnv:
    """Small deterministic env implementing DreamerVLAOnlineTrainEnv's contract."""

    def __init__(
        self,
        horizon: int = 3,
        image_shape: tuple[int, int, int] = (4, 4, 3),
        embedding_dim: int = 6,
    ) -> None:
        self.horizon = int(horizon)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.embedding_dim = int(embedding_dim)
        self.task_id = 0
        self.episode_id = 0
        self.step_i = 0

    def set_task(self, task_id: int) -> None:
        self.task_id = int(task_id)

    def reset(
        self,
        *,
        task_id: int | None = None,
        episode_id: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if task_id is not None:
            self.set_task(int(task_id))
        if episode_id is not None:
            self.episode_id = int(episode_id)
        self.step_i = 0
        return self._obs(is_first=True), {"episode_id": self.episode_id}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.step_i += 1
        done = self.step_i >= self.horizon
        reward = 1.0 if done else 0.0
        info = {
            "success": done,
            "wm_action": np.asarray(action, dtype=np.float32).reshape(-1)[:7],
        }
        return self._obs(is_first=False), reward, done, False, info

    def make_transition(
        self,
        obs: dict[str, Any],
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        done = bool(terminated or truncated)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
        return {
            "image": np.asarray(obs["image"], dtype=np.uint8),
            "state": np.asarray(obs["state"], dtype=np.float32),
            "obs_embedding": np.asarray(obs["obs_embedding"], dtype=np.float32),
            "action": action_arr,
            "wm_action": action_arr,
            "policy_action": action_arr,
            "reward": np.float32(reward),
            "done": np.float32(done),
            "discount": np.float32(0.0 if terminated else 1.0),
            "is_first": bool(obs.get("is_first", False)),
            "is_terminal": bool(terminated),
            "is_last": bool(done),
            "task_id": int(obs["task_id"]),
            "episode_id": int(obs.get("episode_id", self.episode_id)),
            "step": int(obs["step"]),
            "task_description": str(obs["task_description"]),
            "success": bool((info or {}).get("success", False)),
        }

    def full_record(self) -> dict[str, Any]:
        return self._obs(is_first=self.step_i == 0)

    def close(self) -> None:
        return None

    def _obs(self, *, is_first: bool) -> dict[str, Any]:
        value = self.step_i + self.episode_id * 10
        return {
            "image": np.full(self.image_shape, value, dtype=np.uint8),
            "state": np.full((2,), value, dtype=np.float32),
            "obs_embedding": np.full((self.embedding_dim,), value, dtype=np.float32),
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "step": self.step_i,
            "task_description": f"task {self.task_id}",
            "is_first": bool(is_first),
        }


class NoSidecarTrainEnv(CounterEnv):
    """DreamerVLAOnlineTrainEnv-like env that emits no hidden sidecars itself."""

    def __init__(
        self,
        horizon: int = 1,
        image_shape: tuple[int, int, int] = (4, 4, 3),
        state_dim: int = 2,
    ) -> None:
        super().__init__(
            horizon=horizon,
            image_shape=image_shape,
            embedding_dim=state_dim,
        )
        self.state_dim = int(state_dim)
        self.received_actions: list[np.ndarray] = []

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        self.received_actions.append(action_arr.copy())
        self.step_i += 1
        done = self.step_i >= self.horizon
        reward = 1.0 if done else 0.0
        return (
            self._obs(is_first=False),
            reward,
            done,
            False,
            {
                "success": done,
                "wm_action": action_arr.astype(np.float32, copy=False),
            },
        )

    def make_transition(
        self,
        obs: dict[str, Any],
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        done = bool(terminated or truncated)
        info = dict(info or {})
        policy_action = np.asarray(action, dtype=np.float32).reshape(-1)
        wm_action = np.asarray(
            info.get("wm_action", policy_action),
            dtype=np.float32,
        ).reshape(-1)
        return {
            "image": np.asarray(obs["image"], dtype=np.uint8),
            "state": np.asarray(obs["state"], dtype=np.float32),
            "action": wm_action,
            "wm_action": wm_action,
            "policy_action": policy_action,
            "reward": np.float32(reward),
            "done": np.float32(done),
            "discount": np.float32(0.0 if terminated else 1.0),
            "is_first": bool(obs.get("is_first", False)),
            "is_terminal": bool(terminated),
            "is_last": bool(done),
            "task_id": int(obs["task_id"]),
            "episode_id": int(obs.get("episode_id", self.episode_id)),
            "step": int(obs["step"]),
            "task_description": str(obs["task_description"]),
        }

    def _obs(self, *, is_first: bool) -> dict[str, Any]:
        value = self.step_i + self.episode_id * 10
        return {
            "image": np.full(self.image_shape, value, dtype=np.uint8),
            "state": np.full((self.state_dim,), value, dtype=np.float32),
            "agentview_rgb": np.full(self.image_shape, value, dtype=np.uint8),
            "eye_in_hand_rgb": np.full(self.image_shape, value, dtype=np.uint8),
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "step": self.step_i,
            "task_description": f"task {self.task_id}",
            "is_first": bool(is_first),
        }


class BatchedCounterEnv:
    """Counter env that manages multiple slots inside one env object."""

    def __init__(
        self,
        num_envs: int = 1,
        horizon: int = 3,
        image_shape: tuple[int, int, int] = (4, 4, 3),
        embedding_dim: int = 6,
    ) -> None:
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.embedding_dim = int(embedding_dim)
        self.task_ids = [0 for _ in range(self.num_envs)]
        self.episode_ids = [0 for _ in range(self.num_envs)]
        self.step_i = [0 for _ in range(self.num_envs)]
        self.reset_batch_calls = 0
        self.wm_loaded_version: int | None = None
        self.classifier_loaded_version: int | None = None

    def reset_slot(
        self,
        slot_id: int,
        *,
        task_id: int = 0,
        episode_id: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._validate_slot(slot_id)
        self.task_ids[slot_id] = int(task_id)
        self.episode_ids[slot_id] = int(episode_id)
        self.step_i[slot_id] = 0
        return self._obs(slot_id, is_first=True), {"episode_id": self.episode_ids[slot_id]}

    def reset_batch(
        self,
        task_ids: list[int],
        episode_ids: list[int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.reset_batch_calls += 1
        outputs = [
            self.reset_slot(
                slot_id,
                task_id=int(task_id),
                episode_id=int(episode_id),
            )
            for slot_id, (task_id, episode_id) in enumerate(zip(task_ids, episode_ids, strict=True))
        ]
        return [obs for obs, _info in outputs], [info for _obs, info in outputs]

    def step_slot(
        self,
        slot_id: int,
        action: Any,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._validate_slot(slot_id)
        self.step_i[slot_id] += 1
        done = self.step_i[slot_id] >= self.horizon
        reward = 1.0 if done else 0.0
        info = {
            "success": done,
            "slot_id": int(slot_id),
            "wm_action": np.asarray(action, dtype=np.float32).reshape(-1)[:7],
        }
        return self._obs(slot_id, is_first=False), reward, done, False, info

    def make_transition(
        self,
        obs: dict[str, Any],
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        done = bool(terminated or truncated)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
        return {
            "image": np.asarray(obs["image"], dtype=np.uint8),
            "state": np.asarray(obs["state"], dtype=np.float32),
            "obs_embedding": np.asarray(obs["obs_embedding"], dtype=np.float32),
            "action": action_arr,
            "wm_action": action_arr,
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
            "success": bool((info or {}).get("success", False)),
        }

    def close(self) -> None:
        return None

    def load_world_model_state(self, state_dict: dict[str, Any], version: int) -> None:
        del state_dict
        self.wm_loaded_version = int(version)

    def load_classifier_state(self, state_dict: dict[str, Any], version: int) -> None:
        del state_dict
        self.classifier_loaded_version = int(version)

    def _obs(self, slot_id: int, *, is_first: bool) -> dict[str, Any]:
        value = self.step_i[slot_id] + self.episode_ids[slot_id] * 10
        return {
            "image": np.full(self.image_shape, value, dtype=np.uint8),
            "state": np.full((2,), value, dtype=np.float32),
            "obs_embedding": np.full((self.embedding_dim,), value, dtype=np.float32),
            "task_id": self.task_ids[slot_id],
            "episode_id": self.episode_ids[slot_id],
            "step": self.step_i[slot_id],
            "task_description": f"task {self.task_ids[slot_id]}",
            "is_first": bool(is_first),
        }

    def _validate_slot(self, slot_id: int) -> None:
        if not 0 <= int(slot_id) < self.num_envs:
            raise ValueError(f"slot_id {slot_id} is out of range")


class DumpCounterEnv(CounterEnv):
    """CounterEnv variant whose transitions match RolloutDumpWriter schema."""

    def make_transition(
        self,
        obs: dict[str, Any],
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        done = bool(terminated or truncated)
        success = bool((info or {}).get("success", False))
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
        step = int(obs["step"])
        state_value = float(step + int(obs.get("episode_id", self.episode_id)) * 10)
        robot_states = np.full((9,), state_value, dtype=np.float64)
        states = np.full((5,), state_value, dtype=np.float64)
        ee_pos = np.full((3,), state_value, dtype=np.float64)
        ee_ori = np.full((3,), state_value + 0.1, dtype=np.float64)
        gripper = np.full((2,), state_value + 0.2, dtype=np.float64)
        joint = np.full((7,), state_value + 0.3, dtype=np.float64)
        return {
            "actions": action_arr.astype(np.float64),
            "rewards": np.float32(0.0),
            "sparse_rewards": np.uint8(1 if success else 0),
            "dones": np.uint8(1 if done else 0),
            "robot_states": robot_states,
            "states": states,
            "obs": {
                "agentview_rgb": np.asarray(obs["image"], dtype=np.uint8),
                "eye_in_hand_rgb": np.asarray(obs["image"], dtype=np.uint8),
                "ee_pos": ee_pos,
                "ee_ori": ee_ori,
                "ee_states": np.concatenate([ee_pos, ee_ori]),
                "gripper_states": gripper,
                "joint_states": joint,
            },
            "task_id": int(obs["task_id"]),
            "episode_id": int(self.episode_id),
            "task_description": str(obs["task_description"]),
            "success": success,
        }


class AlternatingSuccessDumpEnv(DumpCounterEnv):
    """DumpCounterEnv with deterministic 50% success by local episode id."""

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        if bool(terminated or truncated):
            success = (int(self.episode_id) % 2) == 0
            info = dict(info)
            info["success"] = success
            reward = 1.0 if success else 0.0
            terminated = bool(success)
            truncated = not success
        return obs, reward, terminated, truncated, info
