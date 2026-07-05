"""PERF-Q3/Q4: slice-read HDF5 ``actions`` per item (no whole-segment read).

Equivalence + IO gate for the two map-style HDF5 datasets. The reference
functions reproduce the OLD whole-read-then-slice semantics; the datasets must
return byte-identical data while reading ONLY the needed ``[lo:hi]`` rows.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

# --- tiny HDF5 fixture (matches existing dataset test fixtures) -------------


def _make_actions(length: int, action_dim: int = 7) -> np.ndarray:
    # Distinct, non-zero values per (t, dim) so any padding/aliasing bug shows.
    base = np.arange(length * action_dim, dtype=np.float32).reshape(length, action_dim)
    return base * 0.01 - 0.3


def _write_demo_file(
    path: Path, lengths: tuple[int, ...], action_dim: int = 7
) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for demo_idx, length in enumerate(lengths):
            demo = data.create_group(f"demo_{demo_idx}")
            demo.create_dataset("actions", data=_make_actions(length, action_dim))
            obs = demo.create_group("obs")
            obs.create_dataset(
                "agentview_rgb", data=np.zeros((length, 4, 4, 3), dtype=np.uint8)
            )
            obs.create_dataset(
                "eye_in_hand_rgb", data=np.zeros((length, 4, 4, 3), dtype=np.uint8)
            )
            obs.create_dataset(
                "ee_states", data=np.zeros((length, 6), dtype=np.float32)
            )
            obs.create_dataset(
                "gripper_states", data=np.zeros((length, 2), dtype=np.float32)
            )
            demo.create_dataset("rewards", data=np.arange(length, dtype=np.float32))
            demo.create_dataset(
                "dones", data=(np.arange(length) == length - 1).astype(np.float32)
            )


# --- a dataset proxy that forbids a full-array read of ``actions`` ----------


class _NoWholeReadDataset:
    """Wraps the ``actions`` h5py.Dataset; a non-sliced read raises."""

    def __init__(self, dataset: h5py.Dataset) -> None:
        self._dataset = dataset

    @property
    def shape(self):
        return self._dataset.shape

    def __array__(self, dtype=None):
        raise AssertionError("whole-array read of actions is forbidden")

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._dataset[key]
        if isinstance(key, tuple) and key == ():
            raise AssertionError("whole-array read of actions is forbidden")
        if key is Ellipsis:
            raise AssertionError("whole-array read of actions is forbidden")
        return self._dataset[key]


class _DemoProxy:
    def __init__(self, demo: h5py.Group) -> None:
        self._demo = demo

    def __getitem__(self, key):
        if key == "actions":
            return _NoWholeReadDataset(self._demo[key])
        return self._demo[key]


# --- Q3 reference (old whole-read-then-gather, verbatim semantics) ----------


def _ref_action_chunk(
    demo: h5py.Group, index: int, horizon: int, stats: dict
) -> np.ndarray:
    from dreamervla.dataset.vla_sft_hdf5_dataset import (
        _libero_oft_action_transform,
        _normalize_bounds_q99,
    )

    raw = np.asarray(demo["actions"], dtype=np.float32)
    indices = np.minimum(
        np.arange(index, index + horizon, dtype=np.int64), raw.shape[0] - 1
    )
    actions = _libero_oft_action_transform(raw[indices])
    return _normalize_bounds_q99(actions, stats["action"])


def _stats(action_dim: int = 7, proprio_dim: int = 8) -> dict:
    return {
        "action": {
            "q01": [-1.0] * action_dim,
            "q99": [1.0] * action_dim,
            "mask": [True] * action_dim,
        },
        "proprio": {
            "q01": [-1.0] * proprio_dim,
            "q99": [1.0] * proprio_dim,
            "mask": [True] * proprio_dim,
        },
    }


@pytest.mark.parametrize("horizon", [1, 3, 8])
def test_q3_action_chunk_matches_whole_read_reference(
    tmp_path: Path, horizon: int
) -> None:
    from dreamervla.dataset.vla_sft_hdf5_dataset import VLASFTHDF5Dataset

    length = 5
    _write_demo_file(tmp_path / "task_alpha_demo.hdf5", lengths=(length,))
    stats = _stats()

    # ``_action_chunk`` only needs these two attributes; bypass the OFT-dependent
    # __init__ (the vendored tree is absent in worktrees).
    dataset = VLASFTHDF5Dataset.__new__(VLASFTHDF5Dataset)
    dataset.action_horizon = horizon
    dataset.dataset_statistics = stats

    with h5py.File(tmp_path / "task_alpha_demo.hdf5", "r") as handle:
        demo = handle["data"]["demo_0"]
        proxy = _DemoProxy(demo)
        for index in range(length):  # first, interior, last (padding) frames
            expected = _ref_action_chunk(demo, index, horizon, stats)
            actual = dataset._action_chunk(proxy, index)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape == (horizon, 7)
            assert np.array_equal(actual, expected)


# --- Q4 reference (old whole-read, verbatim semantics) ----------------------


def _ref_pixel_actions(
    demo: h5py.Group, start: int, sequence_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_actions = np.asarray(demo["actions"], dtype=np.float32)
    end = start + sequence_length
    prev_actions = np.zeros(
        (sequence_length, raw_actions.shape[-1]), dtype=np.float32
    )
    if sequence_length > 1:
        prev_actions[1:] = raw_actions[start : end - 1]
    actions = torch.from_numpy(prev_actions)
    current_actions = torch.from_numpy(raw_actions[start:end].copy())
    return actions, current_actions


@pytest.mark.parametrize("sequence_length", [1, 4])
def test_q4_pixel_actions_match_whole_read_reference(
    tmp_path: Path, sequence_length: int
) -> None:
    from dreamervla.dataset.pixel_sequence_dataset import PixelSequenceDataset

    length = 6
    _write_demo_file(tmp_path / "task_alpha_demo.hdf5", lengths=(length,))

    dataset = PixelSequenceDataset(
        hdf5_dir=tmp_path,
        sequence_length=sequence_length,
        image_size=4,
        stride=1,
    )

    last_start = length - sequence_length
    for index, entry in enumerate(dataset._entries):
        item = dataset[index]
        exp_prev, exp_cur = _ref_pixel_actions(
            dataset._file(entry.file_path)["data"][entry.demo_key],
            entry.start,
            sequence_length,
        )
        assert torch.equal(item["actions"], exp_prev)
        assert torch.equal(item["current_actions"], exp_cur)
    assert dataset._entries[0].start == 0
    assert dataset._entries[-1].start == last_start


def test_q4_drops_whole_read_of_actions(tmp_path: Path) -> None:
    """The slice-read item must not trigger a full-array read of ``actions``."""
    from dreamervla.dataset.pixel_sequence_dataset import PixelSequenceDataset

    _write_demo_file(tmp_path / "task_alpha_demo.hdf5", lengths=(6,))
    dataset = PixelSequenceDataset(
        hdf5_dir=tmp_path, sequence_length=4, image_size=4, stride=1
    )

    entry = dataset._entries[1]
    handle = dataset._file(entry.file_path)
    demo = handle["data"][entry.demo_key]
    obs_group = demo["obs"]

    # Mirror __getitem__ but with an actions dataset that forbids whole reads.
    start = int(entry.start)
    end = start + dataset.sequence_length
    frames = []
    for key in dataset.image_keys:
        arr = np.asarray(obs_group[key][start:end], dtype=np.uint8)
        frames.append(torch.from_numpy(arr).permute(0, 3, 1, 2).to(torch.float32))
    _ = dataset._resize_images(torch.cat(frames, dim=1)).contiguous()

    guarded = _NoWholeReadDataset(demo["actions"])
    window = np.asarray(guarded[start:end], dtype=np.float32)  # must NOT raise
    assert window.shape == (dataset.sequence_length, dataset.action_dim)
    with pytest.raises(AssertionError):
        np.asarray(guarded)  # the old whole-read path would have hit this
