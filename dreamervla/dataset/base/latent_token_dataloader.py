"""Aligned hidden-token windows and strided latent trajectories.

The strided trajectory protocol follows the MIT-licensed DINO-WM dataset."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dreamervla.dataset.base.hdf5_dataloader import (
    PixelSequenceDataset,
)
from dreamervla.dataset.libero import read_libero_proprio
from dreamervla.preprocess.sidecar_schema import (
    DEFAULT_HIDDEN_KEY,
    validate_hidden_token_preprocess_config,
    validate_hidden_token_sidecar_dir,
)


def find_demo_pairs(raw_dir: str | Path, hidden_dir: str | Path) -> list[tuple[Path, Path, str]]:
    """Return exact raw/hidden pairs; mismatches are hard data errors.

    Returns a list of (raw_path, hidden_path, demo_key) where ``demo_key`` is
    an HDF5 path like ``data/demo_0``. Each raw file may contain many demo
    groups; each raw/hidden intersection with ``obs_embedding`` gets one tuple.
    """
    raw_dir = Path(raw_dir)
    hidden_dir = Path(hidden_dir)
    raw_files = sorted(raw_dir.glob("*.hdf5"))
    hidden_files = sorted(hidden_dir.glob("*.hdf5"))
    raw_names = {path.name for path in raw_files}
    hidden_names = {path.name for path in hidden_files}
    if raw_names != hidden_names:
        raise ValueError(
            "reward/hidden file set mismatch: "
            f"missing_hidden={sorted(raw_names - hidden_names)!r}, "
            f"extra_hidden={sorted(hidden_names - raw_names)!r}"
        )
    pairs: list[tuple[Path, Path, str]] = []
    for raw_p in raw_files:
        hid_p = hidden_dir / raw_p.name
        with h5py.File(str(raw_p), "r") as rr, h5py.File(str(hid_p), "r") as hh:
            if "data" not in rr or "data" not in hh:
                raise ValueError(f"{raw_p.name} must contain data groups in both files")
            raw_keys = set(rr["data"].keys())
            hidden_keys = set(hh["data"].keys())
            if raw_keys != hidden_keys:
                raise ValueError(
                    f"{raw_p.name} reward/hidden demo set mismatch: "
                    f"missing_hidden={sorted(raw_keys - hidden_keys)!r}, "
                    f"extra_hidden={sorted(hidden_keys - raw_keys)!r}"
                )
            for key in sorted(raw_keys):
                raw_demo = rr[f"data/{key}"]
                hidden_demo = hh[f"data/{key}"]
                if "actions" not in raw_demo or "obs_embedding" not in hidden_demo:
                    raise ValueError(f"{raw_p.name}:data/{key} requires actions and obs_embedding")
                raw_length = int(raw_demo["actions"].shape[0])
                hidden_length = int(hidden_demo["obs_embedding"].shape[0])
                if raw_length <= 0 or hidden_length <= 0 or raw_length != hidden_length:
                    raise ValueError(
                        f"{raw_p.name}:data/{key} reward/hidden length mismatch: "
                        f"reward={raw_length}, hidden={hidden_length}"
                    )
        for k in sorted(raw_keys):
            pairs.append((raw_p, hid_p, f"data/{k}"))
    return pairs


class PixelHiddenSequenceDataset(PixelSequenceDataset):
    """LIBERO pixel windows plus canonical OpenVLA hidden-token observations.

    The original pixel HDF5 files remain the image source. This dataset reads a
    shape-aligned sidecar directory whose only supported payload is:

      images:        [T, C, H, W], uint8-range float tensor from the source HDF5
      obs_embedding: [T, 256, 4096] projected vision hidden tokens
    """

    def __init__(
        self,
        hdf5_dir: str | Path,
        hidden_dir: str | Path,
        sequence_length: int = 32,
        image_size: int = 256,
        image_keys: Sequence[str] = ("agentview_rgb",),
        proprio_keys: Sequence[str] | None = None,
        hidden_key: str = DEFAULT_HIDDEN_KEY,
        lang_emb_dir: str | Path | None = None,
        lang_emb_key: str = "lang_emb",
        max_files: int | None = None,
        max_demos_per_file: int | None = None,
        max_windows: int | None = None,
        stride: int = 1,
        expected_model_path: str | None = None,
        expected_encoder_state_ckpt: str | None = None,
        expected_time_horizon: int | None = None,
        expected_action_head_type: str | None = None,
        expected_obs_hidden_source: str | None = None,
        expected_prompt_style: str | None = None,
        expected_history: int | None = None,
        expected_include_state: bool | None = None,
        expected_rotate_images_180: bool | None = None,
        require_preprocess_config: bool = True,
    ) -> None:
        super().__init__(
            hdf5_dir=hdf5_dir,
            sequence_length=sequence_length,
            image_size=image_size,
            image_keys=image_keys,
            proprio_keys=proprio_keys,
            max_files=max_files,
            max_demos_per_file=max_demos_per_file,
            max_windows=max_windows,
            stride=stride,
        )
        self.hidden_dir = self.resolve_project_path(hidden_dir)
        if not self.hidden_dir.exists():
            raise FileNotFoundError(f"Hidden sidecar directory does not exist: {self.hidden_dir}")
        if str(hidden_key) != DEFAULT_HIDDEN_KEY:
            raise ValueError(f"hidden_key is fixed to {DEFAULT_HIDDEN_KEY!r}, got {hidden_key!r}")
        if not bool(require_preprocess_config):
            raise ValueError("canonical hidden-token sidecars always require metadata")
        self.hidden_key = DEFAULT_HIDDEN_KEY
        self.lang_emb_dir = (
            self.resolve_project_path(lang_emb_dir) if lang_emb_dir is not None else None
        )
        if self.lang_emb_dir is not None and not self.lang_emb_dir.exists():
            raise FileNotFoundError(
                f"Language sidecar directory does not exist: {self.lang_emb_dir}"
            )
        self.lang_emb_key = str(lang_emb_key)
        self._hidden_file_cache: dict[str, h5py.File] = {}
        self._lang_emb_file_cache: dict[str, h5py.File] = {}
        sidecar_config = self._validate_hidden_sidecar(
            expected_model_path=expected_model_path,
            expected_encoder_state_ckpt=expected_encoder_state_ckpt,
            expected_time_horizon=expected_time_horizon,
            expected_action_head_type=expected_action_head_type,
            expected_obs_hidden_source=expected_obs_hidden_source,
            expected_prompt_style=expected_prompt_style,
            expected_history=expected_history,
            expected_include_state=expected_include_state,
            expected_rotate_images_180=expected_rotate_images_180,
            require_preprocess_config=bool(require_preprocess_config),
        )
        if sidecar_config.get("hidden_key") != DEFAULT_HIDDEN_KEY:
            raise AssertionError("validated hidden-token sidecar changed hidden_key")

    @staticmethod
    def _canonical_path(value: str) -> str:
        return str(Path(value).expanduser().resolve())

    @staticmethod
    def _same_path(left: str | None, right: str | None) -> bool:
        if not left or not right:
            return left == right
        return PixelHiddenSequenceDataset._canonical_path(
            left
        ) == PixelHiddenSequenceDataset._canonical_path(right)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _validate_hidden_sidecar(
        self,
        *,
        expected_model_path: str | None,
        expected_encoder_state_ckpt: str | None,
        expected_time_horizon: int | None,
        expected_action_head_type: str | None,
        expected_obs_hidden_source: str | None = None,
        expected_prompt_style: str | None = None,
        expected_history: int | None = None,
        expected_include_state: bool | None = None,
        expected_rotate_images_180: bool | None = None,
        require_preprocess_config: bool = True,
    ) -> dict[str, Any]:
        config_path = self.hidden_dir / "preprocess_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"hidden-token sidecar is missing preprocess_config.json: {config_path}"
            )
        if not bool(require_preprocess_config):
            raise ValueError("canonical hidden-token sidecars always require metadata")
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        validate_hidden_token_preprocess_config(config, context=str(config_path))
        validate_hidden_token_sidecar_dir(
            self.hidden_dir,
            reference_dir=getattr(self, "hdf5_dir", None),
            require_reference_complete=True,
        )
        errors: list[str] = []
        if expected_model_path and not self._same_path(
            config.get("model_path"), expected_model_path
        ):
            errors.append(
                f"model_path mismatch: sidecar={config.get('model_path')!r}, expected={expected_model_path!r}"
            )
        if expected_encoder_state_ckpt and not self._same_path(
            config.get("encoder_state_ckpt"),
            expected_encoder_state_ckpt,
        ):
            errors.append(
                "encoder_state_ckpt mismatch: "
                f"sidecar={config.get('encoder_state_ckpt')!r}, expected={expected_encoder_state_ckpt!r}"
            )
        if expected_time_horizon is not None:
            got = config.get("time_horizon")
            if got is None or int(got) != int(expected_time_horizon):
                errors.append(
                    f"time_horizon mismatch: sidecar={got!r}, expected={int(expected_time_horizon)}"
                )
        if expected_action_head_type:
            got = str(config["action_head_type"])
            if got != str(expected_action_head_type):
                errors.append(
                    f"action_head_type mismatch: sidecar={got!r}, expected={expected_action_head_type!r}"
                )
        if expected_obs_hidden_source:
            got = str(config["obs_hidden_source"])
            if got != str(expected_obs_hidden_source):
                errors.append(
                    f"obs_hidden_source mismatch: sidecar={got!r}, expected={expected_obs_hidden_source!r}"
                )
        if expected_prompt_style:
            got = str(config.get("prompt_style", ""))
            if got != str(expected_prompt_style):
                errors.append(
                    f"prompt_style mismatch: sidecar={got!r}, expected={expected_prompt_style!r}"
                )
        if expected_history is not None:
            got = config.get("history")
            if got is None or int(got) != int(expected_history):
                errors.append(
                    f"history mismatch: sidecar={got!r}, expected={int(expected_history)}"
                )
        if expected_include_state is not None:
            got = self._as_bool(config.get("include_state", False))
            expected = self._as_bool(expected_include_state)
            if got != expected:
                errors.append(f"include_state mismatch: sidecar={got!r}, expected={expected!r}")
        if expected_rotate_images_180 is not None:
            got = self._as_bool(config.get("rotate_images_180", False))
            expected = self._as_bool(expected_rotate_images_180)
            if got != expected:
                errors.append(f"rotate_images_180 mismatch: sidecar={got!r}, expected={expected!r}")
        if errors:
            joined = "\n  - ".join(errors)
            raise ValueError(
                f"Hidden sidecar metadata does not match this run: {self.hidden_dir}\n  - {joined}"
            )
        return config

    @staticmethod
    def _flat_hidden_dim_from_shape(shape: tuple[int, ...] | None) -> int | None:
        if shape is None:
            return None
        return int(np.prod(shape, dtype=np.int64))

    def _first_sidecar_hidden_shape(self, hidden_key: str) -> tuple[int, ...] | None:
        for path in sorted(self.hidden_dir.glob("*.hdf5")):
            with h5py.File(path, "r") as handle:
                data = handle.get("data")
                if data is None:
                    continue
                for demo_key in data:
                    demo = data[demo_key]
                    if hidden_key in demo:
                        return tuple(int(dim) for dim in demo[hidden_key].shape[1:])
        return None

    def _first_sidecar_hidden_dim(self, hidden_key: str) -> int | None:
        return self._flat_hidden_dim_from_shape(self._first_sidecar_hidden_shape(hidden_key))

    def _hidden_path_for_source(self, source_path: str | Path) -> Path:
        return self.hidden_dir / Path(source_path).name

    def _hidden_file(self, source_path: str | Path) -> h5py.File:
        hidden_path = self._hidden_path_for_source(source_path)
        key = str(hidden_path)
        handle = self._hidden_file_cache.get(key)
        if handle is None:
            if not hidden_path.is_file():
                raise FileNotFoundError(f"Missing hidden sidecar for {source_path}: {hidden_path}")
            handle = h5py.File(hidden_path, mode="r", swmr=True, libver="latest")
            self._hidden_file_cache[key] = handle
        return handle

    def _lang_emb_path_for_source(self, source_path: str | Path) -> Path:
        if self.lang_emb_dir is None:
            raise RuntimeError("lang_emb_dir is not configured")
        return self.lang_emb_dir / Path(source_path).name

    def _lang_emb_file(self, source_path: str | Path) -> h5py.File:
        lang_path = self._lang_emb_path_for_source(source_path)
        key = str(lang_path)
        handle = self._lang_emb_file_cache.get(key)
        if handle is None:
            if not lang_path.is_file():
                raise FileNotFoundError(f"Missing language sidecar for {source_path}: {lang_path}")
            handle = h5py.File(lang_path, mode="r", swmr=True, libver="latest")
            self._lang_emb_file_cache[key] = handle
        return handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self._entries[int(index)]
        item = super().__getitem__(index)
        start = int(entry.start)
        end = start + self.sequence_length
        handle = self._hidden_file(entry.file_path)
        try:
            dset = handle["data"][entry.demo_key][self.hidden_key]
        except KeyError as exc:
            raise KeyError(
                f"{self._hidden_path_for_source(entry.file_path)}:{entry.demo_key} "
                f"missing {self.hidden_key}"
            ) from exc
        if int(dset.shape[0]) < end:
            raise ValueError(
                f"Hidden sidecar length mismatch for {entry.demo_key}: "
                f"need end={end}, have {dset.shape[0]}"
            )
        hidden = np.asarray(dset[start:end])
        item["obs_embedding"] = torch.from_numpy(hidden)
        if self.lang_emb_dir is not None:
            lang_handle = self._lang_emb_file(entry.file_path)
            try:
                lang_dset = lang_handle["data"][entry.demo_key][self.lang_emb_key]
            except KeyError as exc:
                raise KeyError(
                    f"{self._lang_emb_path_for_source(entry.file_path)}:{entry.demo_key} "
                    f"missing {self.lang_emb_key}"
                ) from exc
            lang_emb = np.asarray(lang_dset[...], dtype=np.float32)
            if lang_emb.ndim != 1:
                raise ValueError(
                    f"{self.lang_emb_key} must be a per-demo vector, got {lang_emb.shape}"
                )
            item["lang_emb"] = torch.from_numpy(lang_emb)
        item["hidden_path"] = str(self._hidden_path_for_source(entry.file_path))
        return item


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

        pairs = find_demo_pairs(self.raw_dir, self.hidden_dir)
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
                proprios.append(torch.from_numpy(read_libero_proprio(demo)))
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
            proprio = torch.from_numpy(read_libero_proprio(raw_demo)[start : end : self.frameskip])

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


__all__ = ["DinoTokenTrajectoryDataset", "PixelHiddenSequenceDataset", "find_demo_pairs"]
