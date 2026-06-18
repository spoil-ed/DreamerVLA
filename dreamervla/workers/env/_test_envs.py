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
            "task_id": self.task_id,
            "step": self.step_i,
            "task_description": f"task {self.task_id}",
            "is_first": bool(is_first),
        }


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
