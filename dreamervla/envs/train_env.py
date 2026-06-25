"""Aligned online training environment for DreamerVLA on LIBERO.

This module is the canonical online env path for the RynnVLA action-hidden route.
It keeps the environment API Dreamer-style while making the VLA/WM observation
contract explicit:

    vla_policy prompt + history=2 + proprio state + rotate180 + action_query

The env does not run the VLA encoder itself.  Instead, each observation carries
the PIL history and a ready-to-tokenize VLA record so online trainers can build
the same action-hidden input as the offline sidecar.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from libero.libero import benchmark as libero_benchmark
from PIL import Image

from dreamervla.constants import DEFAULT_ACTION_TOKEN_ID
from dreamervla.envs.image_utils import resize_hwc_uint8
from dreamervla.envs.libero_env import (
    TASK_MAX_STEPS,
    get_libero_dummy_action,
    get_libero_env,
    quat2axisangle,
)
from dreamervla.utils.episode_end import resolve_episode_end

ACTION_LOW = np.array(
    [-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0],
    dtype=np.float32,
)
ACTION_HIGH = np.array(
    [0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0],
    dtype=np.float32,
)


def normalize_libero_action(action: np.ndarray | Sequence[float]) -> np.ndarray:
    """Map raw LIBERO env-scale actions to the policy range [-1, 1]."""
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
    denom = np.maximum(ACTION_HIGH - ACTION_LOW, 1e-8)
    return (2.0 * (action_arr - ACTION_LOW) / denom - 1.0).astype(
        np.float32, copy=False
    )


def unnormalize_libero_action(action: np.ndarray | Sequence[float]) -> np.ndarray:
    """Map policy actions in [-1, 1] to raw LIBERO env-scale actions."""
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
    return ((action_arr + 1.0) * 0.5 * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW).astype(
        np.float32,
        copy=False,
    )


@dataclass(frozen=True)
class DreamerVLAOnlineTrainEnvConfig:
    task_suite_name: str = "libero_goal"
    task_id: int = 0
    task_ids: tuple[int, ...] | None = None
    resolution: int = 256
    image_size: int = 64
    history_length: int = 2
    warmup_steps: int = 10
    seed: int = 0
    max_steps: int | None = None
    action_input: Literal["raw", "normalized"] = "normalized"
    clip_actions: bool = True
    reward_mode: Literal["sparse_success", "raw"] = "sparse_success"
    task_sampling: Literal["sequential", "random"] = "sequential"
    init_state_sampling: Literal["sequential", "random"] = "sequential"
    pixel_rotate_180: bool = False
    vla_rotate_180: bool = True
    prompt_style: Literal["vla_policy"] = "vla_policy"
    include_state: bool = True
    obs_hidden_source: Literal["action_query", "input_token_embedding"] = "action_query"
    action_head_type: Literal["legacy", "oft_discrete_token", "oft_l1_regression"] = "legacy"
    target_token_id: int = DEFAULT_ACTION_TOKEN_ID
    full_record: bool = False
    validate_canonical: bool = True


def _coerce_task_ids(task_ids: Sequence[int] | str | None) -> tuple[int, ...] | None:
    if task_ids is None:
        return None
    if isinstance(task_ids, str):
        values = [item.strip() for item in task_ids.split(",") if item.strip()]
        return tuple(int(item) for item in values)
    return tuple(int(item) for item in task_ids)


class DreamerVLAOnlineTrainEnv:
    """Gymnasium-style LIBERO env for online DreamerVLA training.

    ``reset()`` returns ``(obs, info)``. ``step(action)`` returns
    ``(obs, reward, terminated, truncated, info)``.

    ``action_input='normalized'`` means the caller passes RynnVLA/VLA policy-scale
    actions in [-1, 1].  The env executes raw LIBERO actions and records those
    raw actions as ``info['wm_action']`` because the RSSM is trained on HDF5
    executed-action scale.
    """

    def __init__(
        self,
        config: DreamerVLAOnlineTrainEnvConfig | dict[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            cfg = DreamerVLAOnlineTrainEnvConfig()
        elif isinstance(config, DreamerVLAOnlineTrainEnvConfig):
            cfg = config
        else:
            cfg = DreamerVLAOnlineTrainEnvConfig(**dict(config))
        if "task_ids" in overrides:
            overrides["task_ids"] = _coerce_task_ids(overrides["task_ids"])
        if overrides:
            cfg = replace(cfg, **overrides)
        if cfg.task_ids is not None and not isinstance(cfg.task_ids, tuple):
            cfg = replace(cfg, task_ids=_coerce_task_ids(cfg.task_ids))
        self.cfg = cfg
        if bool(self.cfg.validate_canonical):
            self._validate_canonical_config()

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
        self._init_state: np.ndarray | None = None
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
        cls,
        config: DreamerVLAOnlineTrainEnvConfig | dict[str, Any],
    ) -> DreamerVLAOnlineTrainEnv:
        return cls(config)

    @property
    def elapsed_steps(self) -> int:
        return int(self._elapsed_steps)

    @property
    def action_low(self) -> np.ndarray:
        return ACTION_LOW.copy()

    @property
    def action_high(self) -> np.ndarray:
        return ACTION_HIGH.copy()

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

        init_state_index = self._select_init_state_index(episode_id)
        self._init_state = np.asarray(self.initial_states[init_state_index], dtype=np.float64)
        self.env.reset()
        raw_obs = self.env.set_init_state(self.initial_states[init_state_index])
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
            init_state_index=init_state_index,
            policy_action=None,
            env_action=None,
        )
        return obs, info

    def step(
        self, action: np.ndarray | Sequence[float]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError(
                "DreamerVLAOnlineTrainEnv is closed or was not initialised"
            )
        policy_action = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
        env_action = self.policy_action_to_env_action(policy_action)
        raw_obs, raw_reward, raw_done, raw_info = self.env.step(env_action.tolist())
        self._elapsed_steps += 1

        success = self._is_success(
            raw_done=bool(raw_done), reward=float(raw_reward), info=raw_info
        )
        episode_end = resolve_episode_end(
            success=success,
            elapsed_steps=self._elapsed_steps,
            max_steps=self.max_steps,
        )
        terminated = episode_end.terminated
        truncated = episode_end.truncated
        reward = self._reward_from_env(raw_reward=float(raw_reward), success=success)

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
            policy_action=policy_action,
            env_action=env_action,
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

    def render_frame(
        self, view: Literal["third", "wrist"] = "third", *, vla_aligned: bool = True
    ) -> np.ndarray:
        if self._raw_obs is None:
            raise RuntimeError("render_frame called before reset")
        key = "agentview_image" if view == "third" else "robot0_eye_in_hand_image"
        if view not in {"third", "wrist"}:
            raise ValueError("view must be one of {'third', 'wrist'}")
        rotate_180 = (
            self.cfg.vla_rotate_180 if vla_aligned else self.cfg.pixel_rotate_180
        )
        return self._camera_image(self._raw_obs, key, rotate_180=rotate_180)

    def full_record(self) -> dict[str, Any]:
        """Return a dict of LIBERO HDF5 schema fields for the current step.

        Requires cfg.full_record=True (flag checked by the collector).
        Must be called after reset() (or step()).
        """
        if not self.cfg.full_record:
            raise RuntimeError("full_record() requires cfg.full_record=True")
        if self._raw_obs is None:
            raise RuntimeError("full_record called before reset")
        raw = self._raw_obs
        # np.array (not asarray) so the returned leaves never alias self._raw_obs,
        # which robosuite may reuse across step() calls (the collector batches records).
        ee_pos = np.array(raw["robot0_eef_pos"], dtype=np.float64)
        eef_quat = np.array(raw["robot0_eef_quat"], dtype=np.float64)
        ee_ori = quat2axisangle(eef_quat.copy())
        gripper_states = np.array(raw["robot0_gripper_qpos"], dtype=np.float64)
        joint_states = np.array(raw["robot0_joint_pos"], dtype=np.float64)
        ee_states = np.concatenate([ee_pos, ee_ori])
        robot_states = np.concatenate([gripper_states, ee_pos, eef_quat])
        agentview_rgb = self._camera_image(
            raw, "agentview_image", rotate_180=self.cfg.pixel_rotate_180
        )
        eye_in_hand_rgb = self._camera_image(
            raw, "robot0_eye_in_hand_image", rotate_180=self.cfg.pixel_rotate_180
        )
        return {
            "agentview_rgb": agentview_rgb,
            "eye_in_hand_rgb": eye_in_hand_rgb,
            "ee_pos": ee_pos,
            "ee_ori": ee_ori,
            "ee_states": ee_states,
            "gripper_states": gripper_states,
            "joint_states": joint_states,
            "robot_states": robot_states,
            "states": np.asarray(self.env.sim.get_state().flatten(), dtype=np.float64),
            "init_state": self._init_state.copy(),
        }

    def policy_action_to_env_action(
        self, action: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        if self.cfg.action_input == "normalized":
            action_arr = unnormalize_libero_action(action)
        elif self.cfg.action_input == "raw":
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
        else:
            raise ValueError("action_input must be one of {'raw', 'normalized'}")
        if action_arr.shape[0] != 7:
            raise ValueError(
                f"LIBERO action must have 7 values, got shape {tuple(action_arr.shape)}"
            )
        if self.cfg.clip_actions:
            action_arr = np.clip(action_arr, ACTION_LOW, ACTION_HIGH)
        return action_arr.astype(np.float32, copy=False)

    def make_transition(
        self,
        obs: dict[str, Any],
        action: np.ndarray | Sequence[float],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a single-step replay record using WM-compatible action scale."""
        done = bool(terminated or truncated)
        if info is not None and "wm_action" in info:
            wm_action = np.asarray(info["wm_action"], dtype=np.float32).reshape(-1)[:7]
        else:
            wm_action = self.policy_action_to_env_action(action)
        policy_action = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
        return {
            "image": np.asarray(obs["image"], dtype=np.uint8),
            "state": np.asarray(obs["state"], dtype=np.float32),
            "action": wm_action.astype(np.float32, copy=False),
            "wm_action": wm_action.astype(np.float32, copy=False),
            "policy_action": policy_action.astype(np.float32, copy=False),
            "reward": np.float32(reward),
            "done": np.float32(done),
            "discount": np.float32(0.0 if terminated else 1.0),
            "is_first": bool(obs.get("is_first", False)),
            "is_terminal": bool(terminated),
            "is_last": bool(done),
            "task_id": int(obs["task_id"]),
            "step": int(obs["step"]),
            "task_description": str(obs["task_description"]),
        }

    def observation_to_vla_record(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self.build_vla_record(
            frame_history=obs["frame_history"],
            state=np.asarray(obs["state"], dtype=np.float32),
            task_description=str(obs["task_description"]),
        )

    @classmethod
    def build_vla_record(
        cls,
        *,
        frame_history: Sequence[tuple[Image.Image, Image.Image]],
        state: np.ndarray | Sequence[float],
        task_description: str,
    ) -> dict[str, Any]:
        images: list[Image.Image] = []
        for third_pil, wrist_pil in frame_history:
            images.extend([third_pil, wrist_pil])
        human_value = cls.build_vla_prompt(
            task_description=task_description, num_images=len(images)
        )
        return {
            "conversations": [{"from": "human", "value": human_value}],
            "image": images,
            "state": [np.asarray(state, dtype=np.float32)],
            "action": [],
        }

    @staticmethod
    def build_vla_prompt(*, task_description: str, num_images: int) -> str:
        return (
            f"Finish the task: {task_description}."
            + "<|state|>"
            + "<|image|>" * int(num_images)
        )

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

    def _validate_canonical_config(self) -> None:
        errors: list[str] = []
        if int(self.cfg.history_length) < 1:
            errors.append(f"history_length={self.cfg.history_length}, expected >= 1")
        if str(self.cfg.prompt_style) != "vla_policy":
            errors.append(
                f"prompt_style={self.cfg.prompt_style!r}, expected 'vla_policy'"
            )
        if not bool(self.cfg.vla_rotate_180):
            errors.append("vla_rotate_180=False, expected True")
        if str(self.cfg.obs_hidden_source) not in ("action_query", "input_token_embedding"):
            errors.append(
                f"obs_hidden_source={self.cfg.obs_hidden_source!r}, expected "
                "'action_query' or 'input_token_embedding'"
            )
        if str(self.cfg.action_head_type) not in (
            "legacy",
            "oft_discrete_token",
            "oft_l1_regression",
        ):
            errors.append(
                f"action_head_type={self.cfg.action_head_type!r}, expected "
                "legacy, oft_discrete_token, or oft_l1_regression"
            )
        if self.cfg.action_input not in {"raw", "normalized"}:
            errors.append(
                f"action_input={self.cfg.action_input!r}, expected raw or normalized"
            )
        if self.cfg.reward_mode not in {"sparse_success", "raw"}:
            errors.append(
                f"reward_mode={self.cfg.reward_mode!r}, expected sparse_success or raw"
            )
        if self.cfg.task_sampling not in {"sequential", "random"}:
            errors.append(
                f"task_sampling={self.cfg.task_sampling!r}, expected sequential or random"
            )
        if self.cfg.init_state_sampling not in {"sequential", "random"}:
            errors.append(
                f"init_state_sampling={self.cfg.init_state_sampling!r}, expected sequential or random"
            )
        if errors:
            joined = "\n  - ".join(errors)
            raise ValueError(
                f"Non-canonical DreamerVLA online env config:\n  - {joined}"
            )

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

    def _reward_from_env(self, *, raw_reward: float, success: bool) -> float:
        if self.cfg.reward_mode == "raw":
            return float(raw_reward)
        return float(1.0 if success else 0.0)

    @staticmethod
    def _camera_image(
        raw_obs: dict[str, Any], key: str, rotate_180: bool
    ) -> np.ndarray:
        if key not in raw_obs:
            raise KeyError(f"LIBERO observation missing camera key {key!r}")
        image = np.asarray(raw_obs[key], dtype=np.uint8)
        if rotate_180:
            image = image[::-1, ::-1]
        return np.ascontiguousarray(image)

    @staticmethod
    def _resize_hwc_uint8(image: np.ndarray, size: int) -> np.ndarray:
        return resize_hwc_uint8(image, size)

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

        vla_record = self.build_vla_record(
            frame_history=frame_history,
            state=state,
            task_description=self.task_description,
        )
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
            "prompt": self.task_description,
            "vla_prompt": str(vla_record["conversations"][0]["value"]),
            "frame_history": frame_history,
            "vla_record": vla_record,
            "history_length": int(self.cfg.history_length),
            "step": int(self._elapsed_steps),
            "task_suite_name": self.cfg.task_suite_name,
            "task_id": int(self.task_id),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            "discount": np.float32(0.0 if is_terminal else 1.0),
            "alignment": {
                "prompt_style": self.cfg.prompt_style,
                "history": int(self.cfg.history_length),
                "include_state": bool(self.cfg.include_state),
                "rotate_images_180": bool(self.cfg.vla_rotate_180),
                "obs_hidden_source": self.cfg.obs_hidden_source,
                "action_head_type": self.cfg.action_head_type,
                "target_token_id": int(self.cfg.target_token_id),
            },
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
        policy_action: np.ndarray | None = None,
        env_action: np.ndarray | None = None,
    ) -> dict[str, Any]:
        info = dict(raw_info) if isinstance(raw_info, dict) else {}
        done = bool(terminated or truncated)
        info.update(
            {
                "success": bool(terminated),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "timeout": bool(truncated),
                "raw_done": bool(raw_done),
                "reward": float(reward),
                "discount": float(0.0 if terminated else 1.0),
                "step": int(self._elapsed_steps),
                "max_steps": int(self.max_steps),
                "task_suite_name": self.cfg.task_suite_name,
                "task_id": int(self.task_id),
                "task_description": self.task_description,
                "is_last": done,
                "is_terminal": bool(terminated),
            }
        )
        if init_state_index is not None:
            info["init_state_index"] = int(init_state_index)
        if policy_action is not None:
            info["policy_action"] = np.asarray(policy_action, dtype=np.float32)
        if env_action is not None:
            env_action_arr = np.asarray(env_action, dtype=np.float32)
            info["env_action"] = env_action_arr
            info["wm_action"] = env_action_arr
            if self.cfg.action_input == "raw":
                info["normalized_action"] = normalize_libero_action(env_action_arr)
        return info

    def __enter__(self) -> DreamerVLAOnlineTrainEnv:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            self.close()


def build_dreamervla_online_train_envs(
    *,
    task_suite_name: str = "libero_goal",
    task_ids: Iterable[int] | str | None = None,
    num_envs: int | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> list[DreamerVLAOnlineTrainEnv]:
    ids = list(_coerce_task_ids(task_ids) or (0,))
    if num_envs is not None:
        if len(ids) == 1:
            ids = [ids[0] for _ in range(int(num_envs))]
        else:
            ids = ids[: int(num_envs)]
    return [
        DreamerVLAOnlineTrainEnv(
            task_suite_name=task_suite_name,
            task_id=int(task_id),
            seed=int(seed) + idx,
            **kwargs,
        )
        for idx, task_id in enumerate(ids)
    ]


TrainEnv = DreamerVLAOnlineTrainEnv


__all__ = [
    "ACTION_HIGH",
    "ACTION_LOW",
    "DreamerVLAOnlineTrainEnv",
    "DreamerVLAOnlineTrainEnvConfig",
    "TrainEnv",
    "build_dreamervla_online_train_envs",
    "normalize_libero_action",
    "unnormalize_libero_action",
]
