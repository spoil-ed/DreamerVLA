"""Batched LIBERO env with RLinf-style subprocess parallelism.

Port of RLinf ``rlinf/envs/libero/libero_env.py`` (Apache-2.0). Deviations
from the source, all deliberate:

- standard LIBERO only (pro/plus module routing stripped);
- numpy obs/reward/termination arrays instead of torch tensors (single-process
  consumer, no channel transfer);
- ``episode_info`` additionally records ``task_id`` and ``reset_state_id`` so
  the eval driver can compute per-task success rates and dedup wrapped
  enumeration blocks;
- warmup is configured (``reset_wait_steps``/``reset_gripper_open``) instead of
  the hardcoded ``range(15)``;
- ``max_trials_per_task`` caps the reset-state enumeration to the first K init
  states of every task (= ``num_episodes_per_task`` eval semantics);
- ``video_cfg`` support dropped.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from libero.libero import benchmark as libero_benchmark
from PIL import Image

from dreamervla.constants import DEFAULT_ACTION_TOKEN_ID
from dreamervla.envs.libero.utils import (
    TASK_MAX_STEPS,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    resize_hwc_uint8,
)
from dreamervla.envs.libero.venv import ReconfigureSubprocEnv
from dreamervla.utils.episode_end import resolve_episode_end


def _make_offscreen_env_fn(bddl_file_name, camera_heights, camera_widths, seed):
    """Picklable factory for one LIBERO OffScreenRenderEnv subprocess."""

    def _env_fn():
        from libero.libero.envs import OffScreenRenderEnv

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_name,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
        )
        env.seed(seed)
        return env

    return _env_fn


class LiberoEnv:
    def __init__(self, cfg, num_envs, seed_offset=0, total_num_processes=1):
        self.cfg = cfg
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.seed = int(cfg.seed) + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = int(cfg.group_size)
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = bool(cfg.use_fixed_reset_state_ids)
        self.specific_reset_id = cfg.get("specific_reset_id", None)
        self.task_id_filter = cfg.get("task_id_filter", None)
        if self.task_id_filter is not None:
            self.task_id_filter = list(self.task_id_filter)

        self.ignore_terminations = bool(cfg.ignore_terminations)
        self.auto_reset = bool(cfg.auto_reset)

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_suite = self._load_task_suite()
        self.task_descriptions = [""] * self.num_envs

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = bool(cfg.use_rel_reward)
        self.use_step_penalty = bool(cfg.get("use_step_penalty", False))

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.current_raw_obs = None

    # -- construction hooks (overridden by CPU tests) --

    def _load_task_suite(self):
        from libero.libero import benchmark as libero_benchmark

        suite = libero_benchmark.get_benchmark_dict()[
            str(self.cfg.task_suite_name)
        ]()
        return suite

    def _init_env(self):
        self.env = ReconfigureSubprocEnv(self._make_env_fns())

    def _make_env_fns(self, env_idx=None):
        from libero.libero import get_libero_path

        bddl_root = get_libero_path("bddl_files")
        heights = int(self.cfg.init_params.camera_heights)
        widths = int(self.cfg.init_params.camera_widths)
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        env_fns = []
        for env_id in env_idx:
            task = self.task_suite.get_task(int(self.task_ids[env_id]))
            bddl_path = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
            env_fns.append(_make_offscreen_env_fn(bddl_path, heights, widths, self.seed))
            self.task_descriptions[env_id] = task.language
        return env_fns

    # -- reset-state enumeration (RLinf port) --

    def _compute_total_num_group_envs(self):
        max_trials = self.cfg.get("max_trials_per_task", None)
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        for task_id in range(self._get_num_tasks()):
            task_num_trials = len(self.task_suite.get_task_init_states(task_id))
            if max_trials is not None:
                task_num_trials = min(task_num_trials, int(max_trials))
            self.trial_id_bins.append(task_num_trials)
            self.total_num_group_envs += task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

        if self.task_id_filter is not None:
            num_tasks = len(self.trial_id_bins)
            validated_tids = sorted(
                {self._validated_task_id(tid, num_tasks) for tid in self.task_id_filter}
            )
            valid_ids: list[int] = []
            for tid in validated_tids:
                start = self.cumsum_trial_id_bins[tid - 1] if tid > 0 else 0
                end = self.cumsum_trial_id_bins[tid]
                valid_ids.extend(range(start, end))
            self._valid_reset_state_ids = np.array(valid_ids)
        else:
            self._valid_reset_state_ids = None

    def _get_num_tasks(self):
        get_num_tasks = getattr(self.task_suite, "get_num_tasks", None)
        if callable(get_num_tasks):
            return int(get_num_tasks())
        return int(self.task_suite.n_tasks)

    @staticmethod
    def _validated_task_id(tid, num_tasks):
        if not isinstance(tid, (int, np.integer)):
            raise ValueError(f"task_id_filter must contain ints, got {tid!r}")
        tid_int = int(tid)
        if tid_int < 0 or tid_int >= num_tasks:
            raise ValueError(
                f"task_id {tid_int} in task_id_filter is out of range [0, {num_tasks - 1}]"
            )
        return tid_int

    def update_reset_state_ids(self):
        if bool(self.cfg.is_eval) or bool(
            self.cfg.get("use_ordered_reset_state_ids", False)
        ):
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            return int(self.specific_reset_id) * np.ones((num_reset_states,), dtype=int)
        if self._valid_reset_state_ids is not None:
            indices = self._generator.integers(
                low=0, high=len(self._valid_reset_state_ids), size=(num_reset_states,)
            )
            return self._valid_reset_state_ids[indices]
        return self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )

    def get_reset_state_ids_all(self):
        if self._valid_reset_state_ids is not None:
            reset_state_ids = self._valid_reset_state_ids.copy()
        else:
            reset_state_ids = np.arange(self.total_num_group_envs)

        if not bool(self.cfg.is_eval):
            self._generator_ordered.shuffle(reset_state_ids)

        if len(reset_state_ids) < self.total_num_processes:
            repeats = (self.total_num_processes // len(reset_state_ids)) + 1
            reset_state_ids = np.tile(reset_state_ids, repeats)

        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        reset_state_ids = reset_state_ids[:valid_size]
        return reset_state_ids.reshape(self.total_num_processes, -1)

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            return int(self.specific_reset_id) * np.ones((self.num_group,), dtype=int)
        pool = self.reset_state_ids_all[self.seed_offset]
        if len(pool) == 0:
            raise ValueError("LiberoEnv has no reset states to enumerate")
        if int(num_reset_states) > len(pool):
            repeats = (int(num_reset_states) + len(pool) - 1) // len(pool)
            reset_state_ids = np.tile(pool, repeats)[: int(num_reset_states)]
            self.start_idx = int(num_reset_states) % len(pool)
            return reset_state_ids
        if self.start_idx + num_reset_states > len(pool):
            self.reset_state_ids_all = self.get_reset_state_ids_all()
            self.start_idx = 0
            pool = self.reset_state_ids_all[self.seed_offset]
        reset_state_ids = pool[self.start_idx : self.start_idx + num_reset_states]
        self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if start_pivot <= reset_state_id < end_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot
        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        return [
            self.task_suite.get_task_init_states(int(self.task_ids[env_id]))[
                int(self.trial_ids[env_id])
            ]
            for env_id in env_idx
        ]

    # -- properties / metrics (RLinf port) --

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.success_episode_len = np.zeros(self.num_envs, dtype=np.int32)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self.success_episode_len[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.success_episode_len[:] = 0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward * (~self.success_once)
        new_success_mask = terminations & ~self.success_once
        if new_success_mask.any():
            self.success_episode_len[new_success_mask] = self.elapsed_steps[
                new_success_mask
            ]
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_len_for_reward = np.where(
            self.success_once, self.success_episode_len, self.elapsed_steps
        )
        episode_info["reward"] = episode_info["return"] / np.maximum(
            episode_len_for_reward, 1
        )
        # DreamerVLA extension: per-task SR + enumeration dedup need these.
        episode_info["task_id"] = self.task_ids.copy()
        episode_info["reset_state_id"] = self.reset_state_ids.copy()
        infos["episode"] = episode_info
        return infos

    # -- obs wrapping (numpy port of RLinf _wrap_obs) --

    def _extract_image_and_state(self, obs):
        resolution = int(self.cfg.init_params.camera_heights)
        return {
            "full_image": get_libero_image(obs, resolution),
            "wrist_image": get_libero_image(obs, resolution, "robot0_eye_in_hand_image"),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        }

    def _wrap_obs(self, obs_list):
        extracted = [self._extract_image_and_state(obs) for obs in obs_list]
        return {
            "main_images": np.stack([e["full_image"] for e in extracted]),
            "wrist_images": np.stack([e["wrist_image"] for e in extracted]),
            "states": np.stack([e["state"] for e in extracted]),
            "task_descriptions": list(self.task_descriptions),
        }

    # -- reconfigure / reset / step (RLinf port) --

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            task_changed = self.task_ids[env_id] != task_ids[j]
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
            if task_changed or not bool(self.cfg.is_eval):
                reconfig_env_idx.append(env_id)
        if reconfig_env_idx:
            env_fns = self._make_env_fns(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fns, reconfig_env_idx)
        self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        init_state = self._get_reset_states(env_idx=env_idx)
        self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: int | list | np.ndarray | None = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            reset_state_ids = self._get_random_reset_state_ids(len(env_idx))

        self._reconfigure(reset_state_ids, env_idx)
        raw_obs = None
        for _ in range(int(self.cfg.reset_wait_steps)):
            zero_actions = np.zeros((len(env_idx), 7))
            if bool(self.cfg.reset_gripper_open):
                zero_actions[:, -1] = -1
            raw_obs, _reward, _terms, _info_lists = self.env.step(zero_actions, env_idx)
        if raw_obs is None:
            raw_obs = self.env.reset(id=env_idx)
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        return obs, {}

    def step(self, actions=None, auto_reset=True):
        actions = np.asarray(actions)
        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = list(raw_obs)
        terminations = np.asarray(terminations, dtype=bool)
        truncations = self.elapsed_steps >= int(self.cfg.max_episode_steps)
        obs = self._wrap_obs(self.current_raw_obs)

        step_reward = self._calc_step_reward(terminations)
        infos = self._record_metrics(step_reward, terminations, {})
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations.copy()
            terminations = np.zeros_like(terminations)

        dones = terminations | truncations
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_steps, action_dim]
        chunk_actions = np.asarray(chunk_actions)
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)
            chunk_rewards.append(np.asarray(step_reward, dtype=np.float64))
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = np.stack(chunk_rewards, axis=1)
        raw_chunk_terminations = np.stack(raw_chunk_terminations, axis=1)
        raw_chunk_truncations = np.stack(raw_chunk_truncations, axis=1)

        past_terminations = raw_chunk_terminations.any(axis=1)
        past_truncations = raw_chunk_truncations.any(axis=1)
        past_dones = np.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = np.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = np.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.copy()
            chunk_truncations = raw_chunk_truncations.copy()
        return obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        if bool(self.cfg.is_eval):
            self.update_reset_state_ids()
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=self.reset_state_ids[env_idx]
            if self.use_fixed_reset_state_ids
            else None,
        )
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        step_penalty = -1 if self.use_step_penalty else 0
        reward = step_penalty + float(self.cfg.reward_coef) * terminations
        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        return reward

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.close()
        except Exception:
            pass
        return False

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
    history_length: int = 1
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
    include_state: bool = False
    obs_hidden_source: Literal["hidden_token"] = "hidden_token"
    action_head_type: Literal["oft_discrete_token"] = "oft_discrete_token"
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

    ``action_input='normalized'`` means the caller passes VLA/VLA policy-scale
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
        self._init_state_index: int | None = None
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
        self._init_state_index = int(init_state_index)
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
            "init_state_index": int(self._init_state_index)
            if self._init_state_index is not None
            else None,
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
        if int(self.cfg.history_length) != 1:
            errors.append(f"history_length={self.cfg.history_length}, expected 1")
        if str(self.cfg.prompt_style) != "vla_policy":
            errors.append(
                f"prompt_style={self.cfg.prompt_style!r}, expected 'vla_policy'"
            )
        if not bool(self.cfg.vla_rotate_180):
            errors.append("vla_rotate_180=False, expected True")
        if str(self.cfg.obs_hidden_source) != "hidden_token":
            errors.append(
                f"obs_hidden_source={self.cfg.obs_hidden_source!r}, expected "
                "'hidden_token'"
            )
        if str(self.cfg.action_head_type) != "oft_discrete_token":
            errors.append(
                f"action_head_type={self.cfg.action_head_type!r}, expected "
                "oft_discrete_token"
            )
        if bool(self.cfg.include_state):
            errors.append("include_state=True, expected False")
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
        if init_state_index is None:
            init_state_index = getattr(self, "_init_state_index", None)
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


__all__ = [
    "ACTION_HIGH",
    "ACTION_LOW",
    "DreamerVLAOnlineTrainEnv",
    "DreamerVLAOnlineTrainEnvConfig",
    "LiberoEnv",
    "build_dreamervla_online_train_envs",
    "normalize_libero_action",
    "unnormalize_libero_action",
]
