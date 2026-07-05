from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dreamervla.preprocess.libero_utils.noop_marking import (
    compute_noop_mask,
    filter_marked_hdf5_file,
)


def test_compute_noop_mask_matches_filter_equivalent_previous_action() -> None:
    actions = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    mask = compute_noop_mask(actions)

    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, True, False, True, False]


def test_filter_marked_hdf5_file_uses_noop_mask_and_preserves_source_indices(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.hdf5"
    output_path = tmp_path / "filtered.hdf5"

    with h5py.File(source_path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        demo.create_dataset("actions", data=np.arange(4 * 7).reshape(4, 7))
        demo.create_dataset("states", data=np.arange(4 * 3).reshape(4, 3))
        demo.create_dataset("robot_states", data=np.arange(4 * 2).reshape(4, 2))
        demo.create_dataset("rewards", data=np.asarray([0, 0, 0, 1]))
        demo.create_dataset("dones", data=np.asarray([0, 0, 0, 1]))
        demo.create_dataset(
            "noop_mask", data=np.asarray([True, False, True, False], dtype=np.bool_)
        )
        demo.create_dataset("source_indices", data=np.arange(4, dtype=np.int64))
        obs.create_dataset(
            "agentview_rgb", data=np.zeros((4, 2, 2, 3), dtype=np.uint8)
        )
        obs.create_dataset(
            "eye_in_hand_rgb", data=np.ones((4, 2, 2, 3), dtype=np.uint8)
        )
        obs.create_dataset("ee_pos", data=np.arange(4 * 3).reshape(4, 3))

    summary = filter_marked_hdf5_file(source_path, output_path, filter_noops=True)

    assert summary["frames_in"] == 4
    assert summary["frames_out"] == 2
    assert summary["noop_frames"] == 2
    with h5py.File(output_path, "r") as handle:
        demo = handle["data"]["demo_0"]
        assert demo["actions"].shape == (2, 7)
        assert demo["states"][:].tolist() == [[3, 4, 5], [9, 10, 11]]
        assert demo["source_indices"][:].tolist() == [1, 3]
        assert demo["noop_mask"][:].tolist() == [False, False]
        assert bool(demo.attrs["noop_filtered"]) is True
        assert handle.attrs["noop_filter_source_hdf5"] == str(source_path)
