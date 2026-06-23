"""Unit tests for the vectorized (multi-env egl) online cotrain rollout (Option 1).

Pure-CPU: no LIBERO env, no GPU. Helpers and the continuous loop are exercised with
synthetic records and fake VecRolloutEnv/extractors.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rec(h: int = 120, w: int = 160) -> dict:
    rng = np.random.default_rng(0)
    return {
        "agentview_rgb": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
        "eye_in_hand_rgb": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
    }


def _full_record() -> dict:
    rng = np.random.default_rng(1)
    return {
        "agentview_rgb": rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
        "eye_in_hand_rgb": rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
        "ee_pos": rng.standard_normal(3).astype(np.float64),
        "ee_ori": rng.standard_normal(3).astype(np.float64),
        "gripper_states": rng.standard_normal(2).astype(np.float64),
    }


# --------------------------------------------------------------- Task 1
def test_dreamer_image_from_record_matches_format_obs_formula():
    from dreamervla.envs.train_env import DreamerVLAOnlineTrainEnv as Env
    from dreamervla.runners.vectorized_collect import dreamer_image_from_record

    rec, size = _rec(), 64
    out = dreamer_image_from_record(rec, size)
    assert out.shape == (6, size, size) and out.dtype == np.uint8
    third = Env._resize_hwc_uint8(rec["agentview_rgb"], size).transpose(2, 0, 1)
    wrist = Env._resize_hwc_uint8(rec["eye_in_hand_rgb"], size).transpose(2, 0, 1)
    expected = np.concatenate([third, wrist], axis=0).astype(np.uint8)
    np.testing.assert_array_equal(out, expected)
