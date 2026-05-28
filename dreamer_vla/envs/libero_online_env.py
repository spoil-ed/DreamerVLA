"""Online LIBERO environment wrapper for DreamerVLA.

This wrapper is intentionally separate from the eval-only helpers in
``libero_env.py``.  Online training needs a Gymnasium-style API where success
and timeout are not collapsed into one ``done`` flag, plus observations already
formatted for the two DreamerVLA routes:

* pixel DreamerV3: ``obs["image"]`` is uint8 ``[C,H,W]`` with third-view and
  wrist-view channels concatenated.
* token/VLA route: ``obs["frame_history"]`` contains PIL image pairs and
  ``obs["state"]`` contains the VLA proprio state used by the tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from PIL import Image

from libero.libero import benchmark as libero_benchmark

from dreamer_vla.envs.libero_env import (
    TASK_MAX_STEPS,
    get_libero_dummy_action,
    get_libero_env,
    quat2axisangle,
)
from dreamer_vla.utils.episode_end import resolve_episode_end


ACTION_LOW = np.array(
    [-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0],
    dtype=np.float32,
)
ACTION_HIGH = np.array(
    [0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0],
    dtype=np.float32,
)


def normalize_libero_action(action: np.ndarray | Sequence[float]) -> np.ndarray:
    """Map raw LIBERO action-space values to approximately [-1, 1]."""
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
    denom = np.maximum(ACTION_HIGH - ACTION_LOW, 1e-8)
    return 2.0 * (action_arr - ACTION_LOW) / denom - 1.0


def unnormalize_libero_action(action: np.ndarray | Sequence[float]) -> np.ndarray:
    """Map [-1, 1] policy outputs to raw LIBERO action-space values."""
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
    return (action_arr + 1.0) * 0.5 * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW


@dataclass(frozen=True)
class LIBEROOnlineEnvConfig:
    task_suite_name: str = "libero_10"
    task_id: int = 0
    task_ids: tuple[int, ...] | None = None
    resolution: int = 256
    image_size: int = 64
    history_length: int = 1
    warmup_steps: int = 10
    seed: int = 0
    max_steps: int | None = None
    action_input: Literal["raw", "normalized"] = "raw"
    clip_actions: bool = True
    sparse_success_reward: bool = True
    task_sampling: Literal["sequential", "random"] = "sequential"
    init_state_sampling: Literal["sequential", "random"] = "sequential"
    pixel_rotate_180: bool = False
    vla_rotate_180: bool = True


class LIBEROOnlineEnv:
    """Gymnasium-style online LIBERO wrapper.

    ``reset()`` returns ``(obs, info)`` and ``step(action)`` returns
    ``(obs, reward, terminated, truncated, info)``.

    The wrapper keeps raw LIBERO simulator access in ``self.env`` for advanced
    users, but online DreamerVLA code should depend only on the public API.
    """

    def __init__(
        self,
        task_suite_name: str = "libero_10",
        task_id: int = 0,
        task_ids: Sequence[int] | None = None,
        resolution: int = 256,
        image_size: int = 64,
        history_length: int = 1,
        warmup_steps: int = 10,
        seed: int = 0,
        max_steps: int | None = None,
        action_input: Literal["raw", "normalized"] = "raw",
        clip_actions: bool = True,
        sparse_success_reward: bool = True,
        task_sampling: Literal["sequential", "random"] = "sequential",
        init_state_sampling: Literal["sequential", "random"] = "sequential",
        pixel_rotate_180: bool = False,
        vla_rotate_180: bool = True,
    ) -> None:
        if action_input not in {"raw", "normalized"}:
            raise ValueError("action_input must be one of {'raw', 'normalized'}")
        if task_sampling not in {"sequential", "random"}:
            raise ValueError("task_sampling must be one of {'sequential', 'random'}")
        if init_state_sampling not in {"sequential", "random"}:
            raise ValueError(
                "init_state_sampling must be one of {'sequential', 'random'}"
            )

        self.cfg = LIBEROOnlineEnvConfig(
            task_suite_name=str(task_suite_name),
            task_id=int(task_id),
            task_ids=None if task_ids is None else tuple(int(x) for x in task_ids),
            resolution=int(resolution),
            image_size=int(image_size),
            history_length=max(int(history_length), 1),
            warmup_steps=max(int(warmup_steps), 0),
            seed=int(seed),
            max_steps=None if max_steps is None else int(max_steps),
            action_input=action_input,
            clip_actions=bool(clip_actions),
            sparse_success_reward=bool(sparse_success_reward),
            task_sampling=task_sampling,
            init_state_sampling=init_state_sampling,
            pixel_rotate_180=bool(pixel_rotate_180),
            vla_rotate_180=bool(vla_rotate_180),
        )

        benchmark_dict = libero_benchmark.get_benchmark_dict()
        if self.cfg.task_suite_name not in benchmark_dict:
            valid = ", ".join(sorted(benchmark_dict))
            raise ValueError(
                f"Unknown LIBERO task suite {self.cfg.task_suite_name!r}; valid: {valid}"
            )
        self.task_suite = benchmark_dict[self.cfg.task_suite_name]()
        self.num_tasks = int(getattr(self.task_suite, "n_tasks", 0))
        if self.num_tasks <= 0:
            raise RuntimeError(
                f"LIBERO suite {self.cfg.task_suite_name} reports no tasks"
            )

        self.rng = np.random.default_rng(self.cfg.seed)
        self._task_cycle_idx = 0
        self._episode_counter = 0
        self._frame_history: list[tuple[Image.Image, Image.Image]] = []
        self._raw_obs: dict[str, Any] | None = None
        self._elapsed_steps = 0
        self._closed = False

        self.task_id = -1
        self.task = None
        self.initial_states: np.ndarray | Sequence[Any] = []
        self.env = None
        self.task_description = ""
        self.max_steps = 0
        self.set_task(self.cfg.task_id)

    @classmethod
    def from_config(
        cls, cfg: LIBEROOnlineEnvConfig | dict[str, Any]
    ) -> "LIBEROOnlineEnv":
        if isinstance(cfg, LIBEROOnlineEnvConfig):
            return cls(**cfg.__dict__)
        return cls(**dict(cfg))

    @property
    def elapsed_steps(self) -> int:
        return self._elapsed_steps

    @property
    def action_low(self) -> np.ndarray:
        return ACTION_LOW.copy()

    @property
    def action_high(self) -> np.ndarray:
        return ACTION_HIGH.copy()

    def set_task(self, task_id: int) -> None:
        task_id = int(task_id)
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(
                f"task_id={task_id} out of range for {self.cfg.task_suite_name} ({self.num_tasks} tasks)"
            )
        if self.env is not None and task_id == self.task_id:
            return
        self.close()
        self._closed = False
        self.task_id = task_id
        self.task = self.task_suite.get_task(self.task_id)
        self.initial_states = self.task_suite.get_task_init_states(self.task_id)
        if len(self.initial_states) <= 0:
            raise RuntimeError(
                f"LIBERO task {self.cfg.task_suite_name}/{self.task_id} has no initial states"
            )
        self.env, self.task_description = get_libero_env(
            self.task, resolution=self.cfg.resolution
        )
        self.env.seed(self.cfg.seed + self.task_id)
        self.max_steps = int(
            self.cfg.max_steps
            if self.cfg.max_steps is not None
            else TASK_MAX_STEPS.get(self.cfg.task_suite_name, 300)
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        task_id: int | None = None,
        episode_id: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            if self.env is not None:
                self.env.seed(int(seed) + max(self.task_id, 0))
        selected_task_id = self._select_task_id() if task_id is None else int(task_id)
        self.set_task(selected_task_id)

        init_idx = self._select_init_state_index(episode_id)
        self.env.reset()
        raw_obs = self.env.set_init_state(self.initial_states[init_idx])
        for _ in range(self.cfg.warmup_steps):
            raw_obs, _reward, raw_done, _info = self.env.step(get_libero_dummy_action())
            if raw_done:
                break

        self._elapsed_steps = 0
        self._frame_history = []
        self._raw_obs = raw_obs
        obs = self._format_obs(raw_obs, is_first=True, is_last=False, is_terminal=False)
        info = self._make_info(
            raw_info={},
            reward=0.0,
            terminated=False,
            truncated=False,
            raw_done=False,
            init_state_index=init_idx,
        )
        return obs, info

    def step(
        self, action: np.ndarray | Sequence[float]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("LIBEROOnlineEnv is closed or was not initialised")
        action_arr = self._prepare_action(action)
        raw_obs, raw_reward, raw_done, raw_info = self.env.step(action_arr.tolist())
        self._elapsed_steps += 1
        success = self._is_success(
            raw_done=raw_done, reward=float(raw_reward), info=raw_info
        )
        episode_end = resolve_episode_end(
            success=success,
            elapsed_steps=self._elapsed_steps,
            max_steps=self.max_steps,
        )
        terminated = episode_end.terminated
        truncated = episode_end.truncated
        reward = (
            float(1.0 if success else 0.0)
            if self.cfg.sparse_success_reward
            else float(raw_reward)
        )

        self._raw_obs = raw_obs
        is_last = episode_end.done
        obs = self._format_obs(
            raw_obs, is_first=False, is_last=is_last, is_terminal=terminated
        )
        info = self._make_info(
            raw_info=raw_info,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            raw_done=bool(raw_done),
            action=action_arr,
        )
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        self.env = None
        self._closed = True

    def render_frame(self, view: Literal["third", "wrist"] = "third") -> np.ndarray:
        if self._raw_obs is None:
            raise RuntimeError("render_frame called before reset")
        if view == "third":
            return self._camera_image(
                self._raw_obs, "agentview_image", rotate_180=self.cfg.vla_rotate_180
            )
        if view == "wrist":
            return self._camera_image(
                self._raw_obs,
                "robot0_eye_in_hand_image",
                rotate_180=self.cfg.vla_rotate_180,
            )
        raise ValueError("view must be one of {'third', 'wrist'}")

    def make_transition(
        self,
        obs: dict[str, Any],
        action: np.ndarray | Sequence[float],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        """Return a single-step record ready for an online replay buffer."""
        action_arr = self._prepare_action(action)
        done = bool(terminated or truncated)
        return {
            "image": np.asarray(obs["image"], dtype=np.uint8),
            "state": np.asarray(obs["state"], dtype=np.float32),
            "action": action_arr,
            "reward": np.float32(reward),
            "done": np.float32(done),
            "is_first": bool(obs.get("is_first", False)),
            "is_terminal": bool(terminated),
            "is_last": bool(done),
            "task_id": int(obs["task_id"]),
            "step": int(obs["step"]),
            "task_description": str(obs["task_description"]),
        }

    def _select_task_id(self) -> int:
        task_ids = self.cfg.task_ids
        if not task_ids:
            return self.task_id if self.task_id >= 0 else self.cfg.task_id
        if self.cfg.task_sampling == "random":
            return int(task_ids[int(self.rng.integers(0, len(task_ids)))])
        task_id = int(task_ids[self._task_cycle_idx % len(task_ids)])
        self._task_cycle_idx += 1
        return task_id

    def _select_init_state_index(self, episode_id: int | None) -> int:
        if episode_id is not None:
            return int(episode_id) % len(self.initial_states)
        if self.cfg.init_state_sampling == "random":
            return int(self.rng.integers(0, len(self.initial_states)))
        idx = self._episode_counter % len(self.initial_states)
        self._episode_counter += 1
        return int(idx)

    def _prepare_action(self, action: np.ndarray | Sequence[float]) -> np.ndarray:
        if self.cfg.action_input == "normalized":
            action_arr = unnormalize_libero_action(action)
        else:
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
        if action_arr.shape[0] != 7:
            raise ValueError(
                f"LIBERO action must have 7 values, got shape {tuple(action_arr.shape)}"
            )
        if self.cfg.clip_actions:
            action_arr = np.clip(action_arr, ACTION_LOW, ACTION_HIGH)
        return action_arr.astype(np.float32, copy=False)

    @staticmethod
    def _camera_image(
        raw_obs: dict[str, Any], key: str, rotate_180: bool
    ) -> np.ndarray:
        if key not in raw_obs:
            raise KeyError(f"LIBERO observation missing camera key {key!r}")
        img = np.asarray(raw_obs[key], dtype=np.uint8)
        if rotate_180:
            img = img[::-1, ::-1]
        return np.ascontiguousarray(img)

    @staticmethod
    def _resize_hwc_uint8(image: np.ndarray, size: int) -> np.ndarray:
        if image.shape[0] == size and image.shape[1] == size:
            return np.ascontiguousarray(image)
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        return np.asarray(
            Image.fromarray(image).resize((size, size), resample=resample),
            dtype=np.uint8,
        )

    def _format_obs(
        self,
        raw_obs: dict[str, Any],
        *,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
    ) -> dict[str, Any]:
        pixel_third = self._camera_image(
            raw_obs, "agentview_image", rotate_180=self.cfg.pixel_rotate_180
        )
        pixel_wrist = self._camera_image(
            raw_obs, "robot0_eye_in_hand_image", rotate_180=self.cfg.pixel_rotate_180
        )
        pixel_third_small = self._resize_hwc_uint8(pixel_third, self.cfg.image_size)
        pixel_wrist_small = self._resize_hwc_uint8(pixel_wrist, self.cfg.image_size)
        dreamer_image = np.concatenate(
            [
                pixel_third_small.transpose(2, 0, 1),
                pixel_wrist_small.transpose(2, 0, 1),
            ],
            axis=0,
        ).astype(np.uint8, copy=False)

        vla_third = self._camera_image(
            raw_obs, "agentview_image", rotate_180=self.cfg.vla_rotate_180
        )
        vla_wrist = self._camera_image(
            raw_obs, "robot0_eye_in_hand_image", rotate_180=self.cfg.vla_rotate_180
        )
        third_pil = Image.fromarray(vla_third)
        wrist_pil = Image.fromarray(vla_wrist)
        self._frame_history.append((third_pil, wrist_pil))
        if len(self._frame_history) > self.cfg.history_length:
            self._frame_history = self._frame_history[-self.cfg.history_length :]
        history_pad = self.cfg.history_length - len(self._frame_history)
        frame_history = [self._frame_history[0]] * history_pad + list(
            self._frame_history
        )

        state = np.concatenate(
            [
                np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
                quat2axisangle(
                    np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32)
                ).astype(np.float32),
                np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        return {
            "image": np.ascontiguousarray(dreamer_image),
            "agentview_rgb": np.ascontiguousarray(pixel_third),
            "eye_in_hand_rgb": np.ascontiguousarray(pixel_wrist),
            "third_image": np.ascontiguousarray(vla_third),
            "wrist_image": np.ascontiguousarray(vla_wrist),
            "state": state,
            "proprio": state,
            "task_description": self.task_description,
            "language": self.task_description,
            "frame_history": frame_history,
            "step": int(self._elapsed_steps),
            "task_suite_name": self.cfg.task_suite_name,
            "task_id": int(self.task_id),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
        }

    @staticmethod
    def _is_success(raw_done: bool, reward: float, info: Any) -> bool:
        if isinstance(info, dict):
            for key in ("success", "is_success", "task_success"):
                value = info.get(key)
                if value is not None:
                    return bool(value)
        return bool(raw_done or reward > 0.0)

    def _make_info(
        self,
        *,
        raw_info: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        raw_done: bool,
        init_state_index: int | None = None,
        action: np.ndarray | None = None,
    ) -> dict[str, Any]:
        info = dict(raw_info) if isinstance(raw_info, dict) else {}
        info.update(
            {
                "success": bool(terminated),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "timeout": bool(truncated),
                "raw_done": bool(raw_done),
                "reward": float(reward),
                "step": int(self._elapsed_steps),
                "max_steps": int(self.max_steps),
                "task_suite_name": self.cfg.task_suite_name,
                "task_id": int(self.task_id),
                "task_description": self.task_description,
            }
        )
        if init_state_index is not None:
            info["init_state_index"] = int(init_state_index)
        if action is not None:
            info["action"] = np.asarray(action, dtype=np.float32)
        return info

    def __enter__(self) -> "LIBEROOnlineEnv":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            self.close()


def build_libero_online_envs(
    *,
    task_suite_name: str = "libero_10",
    task_ids: Iterable[int] | None = None,
    num_envs: int | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> list[LIBEROOnlineEnv]:
    """Build one env per task id for simple synchronous online collection."""
    ids = list(int(x) for x in (task_ids if task_ids is not None else [0]))
    if num_envs is not None:
        if len(ids) == 1:
            ids = [ids[0] for _ in range(int(num_envs))]
        else:
            ids = ids[: int(num_envs)]
    envs = []
    for idx, task_id in enumerate(ids):
        envs.append(
            LIBEROOnlineEnv(
                task_suite_name=task_suite_name,
                task_id=task_id,
                seed=int(seed) + idx,
                **kwargs,
            )
        )
    return envs


__all__ = [
    "ACTION_HIGH",
    "ACTION_LOW",
    "LIBEROOnlineEnv",
    "LIBEROOnlineEnvConfig",
    "build_libero_online_envs",
    "normalize_libero_action",
    "unnormalize_libero_action",
]
