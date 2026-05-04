"""Utilities for LIBERO environment rollout and evaluation.

Adapted from RynnVLA-002/rynnvla-002/libero_util/libero_utils.py
"""
from __future__ import annotations

import math
import os
import time

import imageio
import numpy as np
from PIL import Image

from libero.libero import get_libero_path
from libero.libero import benchmark as libero_benchmark
from libero.libero.envs import OffScreenRenderEnv

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

TASK_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def get_libero_env(task, resolution: int = 256):
    """Create an off-screen LIBERO environment for *task*."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_dummy_action():
    """No-op action used during the initial stabilisation steps."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs, resize_size, image_view: str = "agentview_image"):
    """Extract an image from the observation dict and rotate 180 degrees."""
    img = obs[image_view]
    img = img[::-1, ::-1]  # rotate 180° to match training preprocessing
    return img


def quat2axisangle(quat):
    """Quaternion (x, y, z, w) -> axis-angle (ax, ay, az)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def save_rollout_video(rollout_dir: str, rollout_images: list, idx: int, success: bool, task_description: str) -> str:
    """Save an MP4 replay of a single episode."""
    os.makedirs(rollout_dir, exist_ok=True)
    sanitised = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = os.path.join(rollout_dir, f"{DATE_TIME}--episode={idx}--success={success}--task={sanitised}.mp4")
    writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        writer.append_data(img)
    writer.close()
    return mp4_path


class LIBERODreamerEnv:
    """Small Dreamer-style wrapper around LIBERO OffScreenRenderEnv.

    The wrapper exposes ``reset() -> obs`` and ``step(action) -> (obs, reward,
    done, info)`` while keeping the fields used by the RynnVLA/DreamerVLA
    tokenizers: third-view image, wrist image, proprioceptive state, task text,
    and a rolling image history.
    """

    def __init__(
        self,
        task_suite_name: str = "libero_10",
        task_id: int = 0,
        episode_id: int = 0,
        resolution: int = 256,
        history_length: int = 1,
        warmup_steps: int = 10,
        seed: int = 0,
    ) -> None:
        self.task_suite_name = str(task_suite_name)
        self.task_id = int(task_id)
        self.episode_id = int(episode_id)
        self.resolution = int(resolution)
        self.history_length = max(int(history_length), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.seed = int(seed)

        benchmark_dict = libero_benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.task_suite_name]()
        self.task = self.task_suite.get_task(self.task_id)
        self.initial_states = self.task_suite.get_task_init_states(self.task_id)
        if not len(self.initial_states):
            raise RuntimeError(f"LIBERO task {self.task_suite_name}/{self.task_id} has no initial states")
        self.env, self.task_description = get_libero_env(self.task, resolution=self.resolution)
        self.env.seed(self.seed)
        self.max_steps = TASK_MAX_STEPS.get(self.task_suite_name, 300)
        self.frame_history: list[tuple[Image.Image, Image.Image]] = []
        self.steps = 0
        self._last_obs = None

    def close(self) -> None:
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            close_fn()

    def reset(self, episode_id: int | None = None) -> dict:
        if episode_id is not None:
            self.episode_id = int(episode_id)
        init_idx = self.episode_id % len(self.initial_states)
        self.env.reset()
        obs = self.env.set_init_state(self.initial_states[init_idx])
        for _ in range(self.warmup_steps):
            obs, _, done, _ = self.env.step(get_libero_dummy_action())
            if done:
                break
        self.steps = 0
        self.frame_history = []
        self._last_obs = obs
        return self._format_obs(obs)

    def step(self, action) -> tuple[dict, float, bool, dict]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
        obs, reward, done, info = self.env.step(action_arr.tolist())
        self.steps += 1
        self._last_obs = obs
        timeout = self.steps >= self.max_steps
        done = bool(done or timeout)
        if info is None:
            info = {}
        info = dict(info)
        info.setdefault("success", bool(done and not timeout))
        info.setdefault("timeout", bool(timeout))
        info.setdefault("task_description", self.task_description)
        return self._format_obs(obs), float(reward), done, info

    def _format_obs(self, obs) -> dict:
        third = get_libero_image(obs, self.resolution)
        wrist = get_libero_image(obs, self.resolution, "robot0_eye_in_hand_image")
        state = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ).astype(np.float32)
        third_pil = Image.fromarray(third)
        wrist_pil = Image.fromarray(wrist)
        self.frame_history.append((third_pil, wrist_pil))
        if len(self.frame_history) > self.history_length:
            self.frame_history = self.frame_history[-self.history_length:]
        padded_history = [self.frame_history[0]] * (self.history_length - len(self.frame_history)) + self.frame_history
        return {
            "third_image": third,
            "wrist_image": wrist,
            "state": state,
            "task_description": self.task_description,
            "frame_history": list(padded_history),
            "step": self.steps,
            "task_id": self.task_id,
        }
