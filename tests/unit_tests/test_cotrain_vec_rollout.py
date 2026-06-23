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


# --------------------------------------------------------------- Task 2
def test_build_transition_has_replay_keys_and_dtypes():
    from dreamervla.runners.online_cotrain_runner import build_cotrain_replay_transition
    from dreamervla.runners.vectorized_collect import (
        dreamer_image_from_record,
        proprio_from_record,
    )

    rec = _full_record()
    emb = np.arange(229376, dtype=np.float32)
    wm = np.arange(7, dtype=np.float32)
    tr = build_cotrain_replay_transition(
        rec, emb, wm, reward=1.5, terminated=True, truncated=False,
        task_id=3, task_description="pick up the bowl", step=4, is_first=False, image_size=64,
    )
    # keys OnlineReplay.sample / add_episode require
    for k in (
        "image", "obs_embedding", "reward", "done", "is_terminal", "is_last",
        "wm_action", "task_id",
    ):
        assert k in tr
    np.testing.assert_array_equal(tr["image"], dreamer_image_from_record(rec, 64))
    np.testing.assert_array_equal(tr["state"], proprio_from_record(rec))
    np.testing.assert_array_equal(tr["wm_action"], wm)
    assert tr["reward"].dtype == np.float32 and float(tr["reward"]) == 1.5
    assert float(tr["done"]) == 1.0 and bool(tr["is_terminal"]) is True
    assert float(tr["discount"]) == 0.0
    assert tr["task_id"] == 3 and tr["task_description"] == "pick up the bowl"
    assert tr["obs_embedding"].dtype == np.float32


def test_build_transition_truncated_keeps_discount_one():
    from dreamervla.runners.online_cotrain_runner import build_cotrain_replay_transition

    tr = build_cotrain_replay_transition(
        _full_record(), np.zeros(8, np.float32), np.zeros(7, np.float32),
        reward=0.0, terminated=False, truncated=True,
        task_id=0, task_description="t", step=10, is_first=False, image_size=64,
    )
    assert float(tr["done"]) == 1.0
    assert bool(tr["is_terminal"]) is False
    assert float(tr["discount"]) == 1.0
