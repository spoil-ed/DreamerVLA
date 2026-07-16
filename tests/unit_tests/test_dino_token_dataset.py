from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dreamervla.dataset.dino_token_dataset import DinoTokenTrajectoryDataset


def _write_demo_pair(
    raw_dir: Path,
    hidden_dir: Path,
    *,
    demo_lengths: list[int],
    constant_controls: bool = False,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "task_demo.hdf5"
    hidden_path = hidden_dir / raw_path.name
    with h5py.File(raw_path, "w") as raw, h5py.File(hidden_path, "w") as hidden:
        raw_data = raw.create_group("data")
        hidden_data = hidden.create_group("data")
        for demo_index, length in enumerate(demo_lengths):
            time = np.arange(length, dtype=np.float32)
            control_time = np.zeros_like(time) if constant_controls else time
            raw_demo = raw_data.create_group(f"demo_{demo_index}")
            raw_demo.create_dataset(
                "actions",
                data=np.stack([control_time, control_time + 10.0], axis=-1),
            )
            obs = raw_demo.create_group("obs")
            obs.create_dataset(
                "ee_pos",
                data=np.stack(
                    [control_time, control_time + 1.0, control_time + 2.0],
                    axis=-1,
                ),
            )
            obs.create_dataset(
                "ee_ori",
                data=np.stack(
                    [
                        control_time + 3.0,
                        control_time + 4.0,
                        control_time + 5.0,
                        control_time + 6.0,
                    ],
                    axis=-1,
                ),
            )
            obs.create_dataset(
                "gripper_states",
                data=control_time[:, None] + 7.0,
            )

            hidden_demo = hidden_data.create_group(f"demo_{demo_index}")
            hidden_demo.create_dataset(
                "obs_embedding",
                data=np.broadcast_to(
                    time[:, None, None],
                    (length, 2, 4),
                ).copy(),
            )


def test_dino_dataset_matches_upstream_frameskip_and_action_concat(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    hidden_dir = tmp_path / "hidden"
    _write_demo_pair(raw_dir, hidden_dir, demo_lengths=[10])
    dataset = DinoTokenTrajectoryDataset(
        raw_dir=raw_dir,
        hidden_dir=hidden_dir,
        split="train",
        num_hist=1,
        num_pred=1,
        frameskip=3,
        train_fraction=1.0,
        normalize_action=False,
        normalize_proprio=False,
        slice_seed=0,
    )

    sample_index = next(
        index for index, (_pair_index, start, _end) in enumerate(dataset.slices) if start == 1
    )
    sample = dataset[sample_index]

    assert sample["obs_embedding"].shape == (2, 2, 4)
    assert torch.equal(sample["obs_embedding"][:, 0, 0], torch.tensor([1.0, 4.0]))
    assert sample["current_actions"].shape == (2, 6)
    assert torch.equal(
        sample["current_actions"],
        torch.tensor(
            [
                [1.0, 11.0, 2.0, 12.0, 3.0, 13.0],
                [4.0, 14.0, 5.0, 15.0, 6.0, 16.0],
            ]
        ),
    )
    assert torch.equal(sample["actions"], sample["current_actions"])
    assert torch.equal(sample["proprio"][:, 0], torch.tensor([1.0, 4.0]))


def test_dino_dataset_uses_upstream_trajectory_split_and_fixed_slice_permutation(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    hidden_dir = tmp_path / "hidden"
    _write_demo_pair(raw_dir, hidden_dir, demo_lengths=[8] * 10)

    train = DinoTokenTrajectoryDataset(
        raw_dir=raw_dir,
        hidden_dir=hidden_dir,
        split="train",
        num_hist=1,
        num_pred=1,
        frameskip=2,
        train_fraction=0.9,
        split_seed=42,
        slice_seed=7,
        normalize_action=False,
        normalize_proprio=False,
    )
    valid = DinoTokenTrajectoryDataset(
        raw_dir=raw_dir,
        hidden_dir=hidden_dir,
        split="valid",
        num_hist=1,
        num_pred=1,
        frameskip=2,
        train_fraction=0.9,
        split_seed=42,
        slice_seed=7,
        normalize_action=False,
        normalize_proprio=False,
    )

    order = torch.randperm(10, generator=torch.Generator().manual_seed(42)).tolist()
    assert train.trajectory_indices == order[:9]
    assert valid.trajectory_indices == order[9:]
    assert set(train.trajectory_indices).isdisjoint(valid.trajectory_indices)
    assert (
        train.slices
        == DinoTokenTrajectoryDataset(
            raw_dir=raw_dir,
            hidden_dir=hidden_dir,
            split="train",
            num_hist=1,
            num_pred=1,
            frameskip=2,
            train_fraction=0.9,
            split_seed=42,
            slice_seed=7,
            normalize_action=False,
            normalize_proprio=False,
        ).slices
    )


def test_dino_dataset_normalizes_action_and_proprio_before_slicing(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    hidden_dir = tmp_path / "hidden"
    _write_demo_pair(raw_dir, hidden_dir, demo_lengths=[8, 8])
    dataset = DinoTokenTrajectoryDataset(
        raw_dir=raw_dir,
        hidden_dir=hidden_dir,
        split="train",
        num_hist=1,
        num_pred=1,
        frameskip=2,
        train_fraction=1.0,
        normalize_action=True,
        normalize_proprio=True,
        slice_seed=0,
    )

    all_actions = torch.cat(
        [
            torch.stack(
                [torch.arange(8, dtype=torch.float32), torch.arange(8) + 10.0],
                dim=-1,
            )
            for _ in range(2)
        ]
    )
    assert torch.allclose(dataset.action_mean, all_actions.mean(dim=0))
    assert torch.allclose(dataset.action_std, all_actions.std(dim=0))

    sample = dataset[0]
    assert torch.isfinite(sample["current_actions"]).all()
    assert torch.isfinite(sample["proprio"]).all()


def test_dino_dataset_selects_fixed_evaluation_windows_per_trajectory(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    hidden_dir = tmp_path / "hidden"
    _write_demo_pair(raw_dir, hidden_dir, demo_lengths=[10, 10])
    dataset = DinoTokenTrajectoryDataset(
        raw_dir=raw_dir,
        hidden_dir=hidden_dir,
        split="valid",
        num_hist=1,
        num_pred=1,
        frameskip=2,
        train_fraction=0.0,
        normalize_action=False,
        normalize_proprio=False,
        slice_seed=9,
    )

    indices = dataset.evaluation_indices(
        max_trajectories=1,
        windows_per_trajectory=3,
    )
    selected = [dataset.slices[index] for index in indices]

    assert len(selected) == 3
    assert {pair_index for pair_index, _start, _end in selected} == {dataset.trajectory_indices[0]}
    assert [start for _pair_index, start, _end in selected] == [0, 3, 6]


def test_dino_dataset_rejects_zero_variance_normalization_corpus(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    hidden_dir = tmp_path / "hidden"
    _write_demo_pair(
        raw_dir,
        hidden_dir,
        demo_lengths=[8],
        constant_controls=True,
    )

    with pytest.raises(ValueError, match="action normalization.*nonzero finite std"):
        DinoTokenTrajectoryDataset(
            raw_dir=raw_dir,
            hidden_dir=hidden_dir,
            split="train",
            num_hist=1,
            num_pred=1,
            frameskip=2,
            train_fraction=1.0,
            normalize_action=True,
            normalize_proprio=True,
        )
