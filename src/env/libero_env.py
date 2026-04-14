"""Utilities for LIBERO environment rollout and evaluation.

Adapted from RynnVLA-002/rynnvla-002/libero_util/libero_utils.py
"""
from __future__ import annotations

import math
import os
import time

import imageio
import numpy as np

from libero.libero import get_libero_path
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
