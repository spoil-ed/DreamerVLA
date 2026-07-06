"""Batched LIBERO eval env with chunk stepping (RLinf LiberoEnv port).

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

import numpy as np

from dreamervla.envs.libero_env import get_libero_image, quat2axisangle
from dreamervla.envs.rlinf_reconfigure_venv import ReconfigureSubprocEnv


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


class LiberoChunkEnv:
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
            raise ValueError("LiberoChunkEnv has no reset states to enumerate")
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
