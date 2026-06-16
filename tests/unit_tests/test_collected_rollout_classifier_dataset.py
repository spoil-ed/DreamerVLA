"""Round-trip test: RolloutDumpWriter → CollectedRolloutClassifierDataset.

Writes two fake demos (one success, one failure) using RolloutDumpWriter,
then loads them through CollectedRolloutClassifierDataset and verifies that
the dataset yields obs_embedding windows plus a binary success label derived
from the terminal (sparse_rewards==1) frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ── test parameters ─────────────────────────────────────────────────────────
T = 10        # episode length; must satisfy T >= SEQ_LEN + 1 so there are negatives
SEQ_LEN = 4   # positive window at start=(T-SEQ_LEN), negative windows at 0..(T-SEQ_LEN-1)
IMAGE_H = 64
IMAGE_W = 64
ACTION_DIM = 7
HIDDEN_DIM = 229376   # 229376 = 56 * 4096; must match what the base class expects
STATE_DIM = 79

PREPROCESS_CONFIG = {
    "action_dim": 7,
    "action_head_type": "oft_l1_regression",
    "center_crop": True,
    "chunk_size": 4,
    "hidden_key": "obs_embedding",
    "history": 2,
    "image_keys": ["agentview_rgb", "eye_in_hand_rgb"],
    "include_state": True,
    "model_path": "/fake/model/path",
    "num_images_in_input": 4,
    "obs_hidden_source": "action_query",
    "output_dtype": "float16",
    "prompt_style": "vla_policy",
    "resolution": 256,
    "rotate_images_180": True,
    "time_horizon": 8,
    "token_dim": 4096,
}


def _make_step(t: int, is_terminal: bool, episode_seed: int = 0) -> dict:
    """Build one fake timestep dict matching RolloutDumpWriter.write_demo input."""
    rng = np.random.default_rng(t + episode_seed)
    return {
        "actions": rng.standard_normal(ACTION_DIM),
        "rewards": np.float32(0.0),
        "sparse_rewards": np.uint8(1 if is_terminal else 0),
        "dones": np.uint8(1 if is_terminal else 0),
        "robot_states": rng.standard_normal(9),
        "states": rng.standard_normal(STATE_DIM),
        "obs": {
            "agentview_rgb": rng.integers(0, 255, (IMAGE_H, IMAGE_W, 3), dtype=np.uint8),
            "eye_in_hand_rgb": rng.integers(0, 255, (IMAGE_H, IMAGE_W, 3), dtype=np.uint8),
            "ee_pos": rng.standard_normal(3),
            "ee_ori": rng.standard_normal(3),
            "ee_states": rng.standard_normal(6),
            "gripper_states": rng.standard_normal(2),
            "joint_states": rng.standard_normal(7),
        },
        "obs_embedding": rng.standard_normal(HIDDEN_DIM).astype(np.float16),
    }


def _make_episode(success: bool) -> list[dict]:
    """Build T steps; last step is terminal only for success episodes."""
    episode_seed = 1000 if success else 2000
    return [
        _make_step(t, is_terminal=success and (t == T - 1), episode_seed=episode_seed)
        for t in range(T)
    ]


@pytest.fixture()
def dump(tmp_path: Path):
    """Write a two-demo dump (one success, one failure) and return dirs."""
    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    shard_name = "shard_000.hdf5"

    writer = RolloutDumpWriter(
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
        shard_name=shard_name,
    )
    writer.write_demo(index=0, steps=_make_episode(success=True), preprocess_config=PREPROCESS_CONFIG)
    writer.write_demo(index=1, steps=_make_episode(success=False), preprocess_config=PREPROCESS_CONFIG)
    writer.close()

    return {"reward_dir": reward_dir, "hidden_dir": hidden_dir}


@pytest.fixture()
def dataset(dump):
    """Instantiate CollectedRolloutClassifierDataset over the two-demo dump."""
    from dreamervla.dataset.collected_rollout_classifier_dataset import (
        CollectedRolloutClassifierDataset,
    )

    return CollectedRolloutClassifierDataset(
        hdf5_dir=str(dump["reward_dir"]),
        hidden_dir=str(dump["hidden_dir"]),
        sequence_length=SEQ_LEN,
        image_size=64,
        expected_model_path="/fake/model/path",
        expected_time_horizon=8,
        expected_action_head_type="oft_l1_regression",
        expected_obs_hidden_source="action_query",
        expected_prompt_style="vla_policy",
        expected_history=2,
        expected_include_state=True,
        expected_rotate_images_180=True,
    )


# ── tests ────────────────────────────────────────────────────────────────────

def test_import():
    """Class is importable from its module."""
    from dreamervla.dataset.collected_rollout_classifier_dataset import (
        CollectedRolloutClassifierDataset,
    )
    from dreamervla.dataset.pixel_hidden_sequence_dataset import PixelHiddenSequenceDataset

    assert issubclass(CollectedRolloutClassifierDataset, PixelHiddenSequenceDataset)


def test_dataset_nonempty(dataset):
    assert len(dataset) > 0


def test_item_has_obs_embedding(dataset):
    item = dataset[0]
    assert "obs_embedding" in item
    assert item["obs_embedding"].shape == (SEQ_LEN, HIDDEN_DIM)


def test_item_has_success_float(dataset):
    item = dataset[0]
    assert "success" in item, "item is missing 'success' key"
    assert isinstance(item["success"], float), (
        f"expected float, got {type(item['success'])}"
    )


def test_success_window_label(dataset):
    """Window that ends at the terminal frame of the success demo → success==1.0.

    demo_0 (success): episode_length=T=10, positive window start=T-SEQ_LEN=6.
    We need to find that entry in the dataset.
    """
    # Iterate over all entries to find the positive one (end == episode_length for demo_0)
    positive_idx = None
    for i, entry in enumerate(dataset._entries):
        end = entry.start + SEQ_LEN
        if "demo_0" in entry.demo_key and end == entry.episode_length:
            positive_idx = i
            break
    assert positive_idx is not None, "No positive window found for demo_0"
    item = dataset[positive_idx]
    assert item["success"] == 1.0, f"expected 1.0, got {item['success']}"


def test_failure_terminal_window_label(dataset):
    """The TERMINAL window of the FAILURE demo must be success==0.0.

    Discriminating case: a window ending at episode_length but with no terminal
    success frame (sparse_rewards all 0) is a FAILURE → 0.0, even though it is
    structurally a terminal-ending window. Guards against labelling success from
    window position (is_positive_window) instead of sparse_rewards.
    """
    idx = None
    for i, entry in enumerate(dataset._entries):
        end = entry.start + SEQ_LEN
        if "demo_1" in entry.demo_key and end == entry.episode_length:
            idx = i
            break
    assert idx is not None, "No terminal window found for failure demo_1"
    item = dataset[idx]
    assert item["success"] == 0.0, f"failure terminal window must be 0.0, got {item['success']}"


def test_success_nonterminal_window_label(dataset):
    """A non-terminal window of the SUCCESS demo (not reaching the success frame) → 0.0."""
    idx = None
    for i, entry in enumerate(dataset._entries):
        end = entry.start + SEQ_LEN
        if "demo_0" in entry.demo_key and end < entry.episode_length:
            idx = i
            break
    assert idx is not None, "No non-terminal window found for success demo_0"
    item = dataset[idx]
    assert item["success"] == 0.0, f"non-terminal success window must be 0.0, got {item['success']}"


def test_standard_fields_present(dataset):
    """Inherited fields from the base class are still present."""
    item = dataset[0]
    for key in ("images", "actions", "current_actions", "rewards", "dones", "is_first"):
        assert key in item, f"missing field: {key}"
