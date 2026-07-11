from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from dreamervla.diagnostics import wm_single_trajectory_raw_overfit as diag


def _write_raw_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "raw.hdf5"
    states = np.arange(8 * 8, dtype=np.float32).reshape(8, 8) / 10.0
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:, 0] = 0.1
    with h5py.File(path, "w") as hdf5:
        demo = hdf5.create_group("data/demo_0")
        demo.create_dataset("actions", data=actions)
        obs = demo.create_group("obs")
        obs.create_dataset("ee_pos", data=states[:, :3])
        obs.create_dataset("ee_ori", data=states[:, 3:7])
        obs.create_dataset("gripper_states", data=states[:, 7:])
    return path


def test_load_raw_episode_without_sidecars(tmp_path: Path) -> None:
    path = _write_raw_fixture(tmp_path)

    episode = diag.load_raw_episode(path, "demo_0")

    assert episode.states.shape == (8, 8)
    assert episode.actions.shape == (8, 7)


def test_raw_overfit_converges_on_one_trajectory(tmp_path: Path) -> None:
    states = np.arange(10 * 2, dtype=np.float32).reshape(10, 2) / 10.0
    actions = np.zeros((10, 1), dtype=np.float32)
    episode = diag.RawEpisode(states=states, actions=actions)

    summary = diag.run_overfit(
        episode=episode,
        out_dir=tmp_path,
        device=torch.device("cpu"),
        history=1,
        max_epochs=100,
        batch_size=4,
        lr=1.0e-2,
        mse_threshold=1.0e-4,
        cosine_threshold=0.999,
        required_passes=2,
        seed=3,
    )

    assert summary["status"] == "converged"
    assert (tmp_path / "raw_wm.ckpt").is_file()
