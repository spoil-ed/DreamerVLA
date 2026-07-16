"""DINO-WM trajectory slicing over persisted DreamerVLA token sidecars.

This module mirrors the data protocol in the upstream MIT-licensed
``dino_wm/datasets/traj_dset.py`` while replacing decoded images with the
existing OpenVLA-OFT token grid. In particular, the train/valid split happens
at trajectory granularity, slice order is permuted once, observations are
sampled every ``frameskip`` environment steps, and the intervening actions are
concatenated into one transition action.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dreamervla.dataset.wm_replay_classifier_dataset import _find_demo_pairs


def _demo_proprio(demo: h5py.Group) -> np.ndarray:
    """Return the canonical LIBERO proprio sidecar as ``[T,8]`` float32."""

    obs = demo["obs"]
    return np.concatenate(
        [
            np.asarray(obs["ee_pos"][...], dtype=np.float32),
            np.asarray(obs["ee_ori"][...], dtype=np.float32),
            np.asarray(obs["gripper_states"][...], dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


class DinoTokenTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Map-style DINO-WM dataset backed by paired LIBERO HDF5 trajectories.

    ``num_hist + num_pred`` is the number of model frames. A model frame is
    separated from the next by ``frameskip`` environment transitions, and its
    action is the flattened concatenation of those transitions. Action and
    proprio statistics are computed over the complete pre-split trajectory
    corpus, matching the upstream PointMaze dataset.
    """

    def __init__(
        self,
        *,
        raw_dir: str | Path,
        hidden_dir: str | Path,
        split: Literal["train", "valid"],
        num_hist: int = 3,
        num_pred: int = 1,
        frameskip: int = 5,
        train_fraction: float = 0.9,
        split_seed: int = 42,
        slice_seed: int = 0,
        normalize_action: bool = True,
        normalize_proprio: bool = True,
        max_episodes: int | None = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "valid"}:
            raise ValueError(f"split must be 'train' or 'valid', got {split!r}")
        if int(num_hist) < 1 or int(num_pred) != 1:
            raise ValueError("DINO token training requires num_hist>=1 and num_pred=1")
        if int(frameskip) < 1:
            raise ValueError("frameskip must be positive")
        if not 0.0 <= float(train_fraction) <= 1.0:
            raise ValueError("train_fraction must be in [0,1]")
        if max_episodes is not None and int(max_episodes) < 1:
            raise ValueError("max_episodes must be positive when provided")

        self.raw_dir = Path(raw_dir).expanduser().resolve()
        self.hidden_dir = Path(hidden_dir).expanduser().resolve()
        self.split = str(split)
        self.num_hist = int(num_hist)
        self.num_pred = int(num_pred)
        self.num_frames = self.num_hist + self.num_pred
        self.frameskip = int(frameskip)
        self.train_fraction = float(train_fraction)
        self.split_seed = int(split_seed)
        self.slice_seed = int(slice_seed)
        self.normalize_action = bool(normalize_action)
        self.normalize_proprio = bool(normalize_proprio)

        pairs = _find_demo_pairs(self.raw_dir, self.hidden_dir)
        if max_episodes is not None:
            pairs = pairs[: int(max_episodes)]
        if not pairs:
            raise RuntimeError(
                f"no paired DINO token trajectories under raw={self.raw_dir} "
                f"hidden={self.hidden_dir}"
            )
        self._pairs = pairs
        self._lengths = self._trajectory_lengths()

        order = torch.randperm(
            len(self._pairs),
            generator=torch.Generator().manual_seed(self.split_seed),
        ).tolist()
        train_count = int(self.train_fraction * len(order))
        train_indices = [int(index) for index in order[:train_count]]
        valid_indices = [int(index) for index in order[train_count:]]
        self.trajectory_indices = train_indices if self.split == "train" else valid_indices

        rng = np.random.RandomState(self.slice_seed)
        train_slices = self._build_slices(train_indices)
        valid_slices = self._build_slices(valid_indices)
        train_slices = self._permuted_slices(rng, train_slices)
        valid_slices = self._permuted_slices(rng, valid_slices)
        self.slices = train_slices if self.split == "train" else valid_slices

        self.action_mean, self.action_std, self.proprio_mean, self.proprio_std = (
            self._normalization_statistics()
        )
        self.base_action_dim = int(self.action_mean.numel())
        self.action_dim = self.base_action_dim * self.frameskip
        self.proprio_dim = int(self.proprio_mean.numel())

    def _trajectory_lengths(self) -> list[int]:
        lengths: list[int] = []
        for raw_path, _hidden_path, demo_key in self._pairs:
            with h5py.File(raw_path, "r") as handle:
                lengths.append(int(handle[f"{demo_key}/actions"].shape[0]))
        return lengths

    def _build_slices(self, indices: list[int]) -> list[tuple[int, int, int]]:
        span = self.num_frames * self.frameskip
        slices: list[tuple[int, int, int]] = []
        for pair_index in indices:
            length = int(self._lengths[pair_index])
            slices.extend((pair_index, start, start + span) for start in range(length - span + 1))
        return slices

    @staticmethod
    def _permuted_slices(
        rng: np.random.RandomState,
        slices: list[tuple[int, int, int]],
    ) -> list[tuple[int, int, int]]:
        if not slices:
            return []
        order = rng.permutation(len(slices)).tolist()
        return [slices[int(index)] for index in order]

    def _normalization_statistics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions: list[torch.Tensor] = []
        proprios: list[torch.Tensor] = []
        for raw_path, _hidden_path, demo_key in self._pairs:
            with h5py.File(raw_path, "r") as handle:
                demo = handle[demo_key]
                actions.append(torch.from_numpy(np.asarray(demo["actions"][...], dtype=np.float32)))
                proprios.append(torch.from_numpy(_demo_proprio(demo)))
        all_actions = torch.cat(actions, dim=0)
        all_proprios = torch.cat(proprios, dim=0)
        action_mean = (
            all_actions.mean(dim=0)
            if self.normalize_action
            else torch.zeros(all_actions.shape[-1], dtype=torch.float32)
        )
        action_std = (
            all_actions.std(dim=0)
            if self.normalize_action
            else torch.ones(all_actions.shape[-1], dtype=torch.float32)
        )
        proprio_mean = (
            all_proprios.mean(dim=0)
            if self.normalize_proprio
            else torch.zeros(all_proprios.shape[-1], dtype=torch.float32)
        )
        proprio_std = (
            all_proprios.std(dim=0)
            if self.normalize_proprio
            else torch.ones(all_proprios.shape[-1], dtype=torch.float32)
        )
        self._validate_normalization_std(
            name="action",
            std=action_std,
            enabled=self.normalize_action,
        )
        self._validate_normalization_std(
            name="proprio",
            std=proprio_std,
            enabled=self.normalize_proprio,
        )
        return action_mean, action_std, proprio_mean, proprio_std

    @staticmethod
    def _validate_normalization_std(
        *,
        name: str,
        std: torch.Tensor,
        enabled: bool,
    ) -> None:
        """Reject undersized smoke corpora that would silently create NaNs."""

        if not enabled:
            return
        invalid = (~torch.isfinite(std)) | (std <= 0)
        if bool(invalid.any()):
            channels = invalid.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"DINO {name} normalization requires nonzero finite std in "
                f"every channel; invalid channels={channels}. Use the full "
                "official trajectory corpus or disable that normalization."
            )

    def __len__(self) -> int:
        return len(self.slices)

    def evaluation_indices(
        self,
        *,
        max_trajectories: int,
        windows_per_trajectory: int,
    ) -> list[int]:
        """Return fixed, evenly spaced slice indices for bounded evaluation."""

        trajectory_limit = int(max_trajectories)
        window_limit = int(windows_per_trajectory)
        selected_trajectories = (
            self.trajectory_indices
            if trajectory_limit <= 0
            else self.trajectory_indices[:trajectory_limit]
        )
        by_trajectory: dict[int, list[tuple[int, int]]] = {
            pair_index: [] for pair_index in selected_trajectories
        }
        for dataset_index, (pair_index, start, _end) in enumerate(self.slices):
            if pair_index in by_trajectory:
                by_trajectory[pair_index].append((start, dataset_index))

        selected: list[int] = []
        for pair_index in selected_trajectories:
            candidates = sorted(by_trajectory[pair_index])
            if not candidates:
                continue
            if window_limit <= 0 or window_limit >= len(candidates):
                positions = range(len(candidates))
            else:
                positions = np.linspace(
                    0,
                    len(candidates) - 1,
                    num=window_limit,
                    dtype=np.int64,
                ).tolist()
            selected.extend(candidates[int(position)][1] for position in positions)
        return selected

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair_index, start, end = self.slices[int(index)]
        raw_path, hidden_path, demo_key = self._pairs[pair_index]
        with h5py.File(raw_path, "r") as raw, h5py.File(hidden_path, "r") as hidden:
            raw_demo = raw[demo_key]
            hidden_demo = hidden[demo_key]
            tokens = np.asarray(hidden_demo["obs_embedding"][start : end : self.frameskip])
            actions = torch.from_numpy(np.asarray(raw_demo["actions"][start:end], dtype=np.float32))
            proprio = torch.from_numpy(_demo_proprio(raw_demo)[start : end : self.frameskip])

        if int(tokens.shape[0]) != self.num_frames:
            raise RuntimeError(
                f"DINO token slice produced {tokens.shape[0]} frames; expected {self.num_frames}"
            )
        actions = (actions - self.action_mean) / self.action_std
        actions = actions.reshape(self.num_frames, self.action_dim)
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        current_actions = actions.contiguous()
        return {
            "obs_embedding": torch.from_numpy(np.ascontiguousarray(tokens)),
            "proprio": proprio.contiguous(),
            "actions": current_actions,
            "current_actions": current_actions,
            "trajectory_index": torch.tensor(pair_index, dtype=torch.long),
            "start_index": torch.tensor(start, dtype=torch.long),
        }


__all__ = ["DinoTokenTrajectoryDataset"]
