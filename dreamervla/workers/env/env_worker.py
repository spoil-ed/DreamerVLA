"""Ray EnvWorker for single-env online rollout collection."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import ray

from dreamervla.scheduler.worker import Worker


class EnvWorker(Worker):
    """Hold one env instance, collect episodes, and push completed episodes."""

    def __init__(
        self,
        env_cfg: dict[str, Any],
        task_id: int,
        replay: Any,
        record_builder: Any | None = None,
    ) -> None:
        super().__init__()
        self.env_cfg = dict(env_cfg)
        self.task_id = int(task_id)
        self.replay = replay
        self._record_builder = record_builder
        self.env: Any | None = None
        self.obs: dict[str, Any] | None = None
        self.episode: list[dict[str, Any]] = []
        self.episode_id = 0

    def init(self) -> None:
        self.env = self._build_env(self.env_cfg)
        if hasattr(self.env, "set_task"):
            self.env.set_task(self.task_id)
        self.obs, _ = self._reset_env()
        self.episode = []
        self.episode_id = 0

    def current_obs(self) -> dict[str, Any]:
        if self.obs is None:
            raise RuntimeError("EnvWorker.init() has not been called")
        return self.obs

    def set_task(self, task_id: int) -> dict[str, Any]:
        self.task_id = int(task_id)
        env = self._env()
        if hasattr(env, "set_task"):
            env.set_task(self.task_id)
        self.episode = []
        self.episode_id = 0
        self.obs, _ = self._reset_env()
        return self.obs

    def step(
        self,
        action: Any,
        obs_embedding: Any,
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        env = self._env()
        obs = self.current_obs()
        record_obs = obs
        if self._record_builder is not None and hasattr(env, "full_record"):
            record_obs = dict(obs)
            record_obs["_full_record"] = env.full_record()
        next_obs, reward, terminated, truncated, info = env.step(action)
        if self._record_builder is not None:
            transition = self._record_builder(
                env,
                record_obs,
                action,
                reward,
                terminated,
                truncated,
                info,
                obs_embedding,
            )
        else:
            transition = env.make_transition(obs, action, reward, terminated, truncated, info)
            transition["obs_embedding"] = np.asarray(obs_embedding, dtype=np.float32)
        self.episode.append(transition)

        done = bool(terminated or truncated)
        if done:
            ray.get(self.replay.add_episode.remote(list(self.episode)))
            self.episode = []
            self.episode_id += 1
            self.obs, reset_info = self._reset_env()
            merged_info = dict(info or {})
            merged_info["reset_info"] = reset_info
            return self.obs, True, merged_info

        self.obs = next_obs
        return self.obs, False, dict(info or {})

    def close(self) -> None:
        env = self.env
        if env is not None and hasattr(env, "close"):
            env.close()
        self.env = None

    def _reset_env(self) -> tuple[dict[str, Any], dict[str, Any]]:
        env = self._env()
        return env.reset(task_id=self.task_id, episode_id=self.episode_id)

    def _env(self) -> Any:
        if self.env is None:
            raise RuntimeError("EnvWorker.init() has not been called")
        return self.env

    @staticmethod
    def _build_env(env_cfg: dict[str, Any]) -> Any:
        target = env_cfg.get("target") or env_cfg.get("_target_") or env_cfg.get("class_path")
        if not target:
            raise ValueError("env_cfg must include target/_target_/class_path")
        kwargs = dict(env_cfg.get("kwargs", {}))
        if ":" in str(target):
            module_name, class_name = str(target).split(":", 1)
        else:
            module_name, class_name = str(target).rsplit(".", 1)
        module = importlib.import_module(module_name)
        env_cls = getattr(module, class_name)
        if hasattr(env_cls, "from_config") and env_cfg.get("use_from_config", False):
            return env_cls.from_config(kwargs)
        return env_cls(**kwargs)
