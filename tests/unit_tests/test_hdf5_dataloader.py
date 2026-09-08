"""Raw HDF5 reading remains independent of model transforms."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dreamervla.dataset.base.base_dataloader import BaseDataset
from dreamervla.dataset.base.hdf5_dataloader import HDF5ActionChunkDataset


def test_base_relative_paths_still_resolve_from_repository_root() -> None:
    root = Path(__file__).resolve().parents[2]
    assert BaseDataset.resolve_project_path("configs/train.yaml") == root / "configs/train.yaml"


def test_raw_action_chunks_keep_values_and_repeat_only_their_own_tail(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "task_demo.hdf5", "w") as handle:
        for eid in range(2):
            demo = handle.create_group(f"data/demo_{eid}")
            demo["actions"] = np.full((2, 7), 5 + eid * 10, dtype=np.float32)
            demo.create_group("obs")["agentview_rgb"] = np.full((2, 4, 4, 3), eid, dtype=np.uint8)
    dataset = HDF5ActionChunkDataset(tmp_path, action_horizon=4)
    np.testing.assert_array_equal(dataset[1]["actions"], np.full((4, 7), 5, dtype=np.float32))
    np.testing.assert_array_equal(dataset[2]["actions"], np.full((4, 7), 15, dtype=np.float32))
    assert dataset[1]["demo_key"] == "demo_0"
    assert dataset[2]["demo_key"] == "demo_1"
    assert dataset[1]["images"]["agentview_rgb"].dtype == np.uint8
    assert dataset.get_normalizer() is None
