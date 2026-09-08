"""HDF5 file access and pixel sequence windows."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dreamervla.dataset.base.base_dataloader import BaseDataset


class HDF5Dataset(BaseDataset):
    """Common HDF5 demo indexing and handle access."""

    @staticmethod
    def extract_demo_index(demo_key: str) -> int:
        match = re.fullmatch(r"demo_(\d+)", demo_key)
        if match is None:
            return -1
        return int(match.group(1))

    @classmethod
    def list_demo_keys(cls, data_group: h5py.Group) -> list[str]:
        demo_keys = [key for key in data_group.keys() if cls.extract_demo_index(key) >= 0]
        return sorted(demo_keys, key=cls.extract_demo_index)

    @staticmethod
    def cached_hdf5_file(
        cache: dict[str, h5py.File],
        path: str,
        open_kwargs: dict[str, Any],
    ) -> h5py.File:
        handle = cache.get(path)
        if handle is None:
            import h5py

            handle = h5py.File(path, **open_kwargs)
            cache[path] = handle
        return handle


@dataclass(frozen=True)
class PixelSequenceSpec:
    hdf5_dir: str
    num_files: int
    num_windows: int
    sequence_length: int
    action_dim: int
    proprio_dim: int
    image_size: int
    image_channels: int
    image_keys: tuple[str, ...]
    proprio_keys: tuple[str, ...]


@dataclass(frozen=True)
class _WindowEntry:
    file_path: str
    demo_key: str
    start: int
    episode_length: int


class PixelSequenceDataset(HDF5Dataset):
    """Pixel-level LIBERO sequence windows for DreamerV3-style WM training.

    Reads Robomimic/LIBERO HDF5 demonstrations directly and returns:

      images:    [T, C, H, W] float32 in the uint8 range [0, 255]
      actions:   [T, A] previous-action convention, actions[0] is zero
      current_actions: [T, A] action executed from this observation
      rewards:   [T]
      dones:     [T]
      is_first:  [T], always true at the first item of each sampled window

    The default image key is the mainline agent-view camera. Explicit callers
    may still select another pixel-only dataset layout; OpenVLA hidden-token
    routes impose their stricter one-camera contract in the derived dataset.
    """

    def __init__(
        self,
        hdf5_dir: str | Path,
        sequence_length: int = 32,
        image_size: int = 64,
        image_keys: Sequence[str] = ("agentview_rgb",),
        proprio_keys: Sequence[str] | None = None,
        max_files: int | None = None,
        max_demos_per_file: int | None = None,
        max_windows: int | None = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.hdf5_dir = self.resolve_project_path(hdf5_dir)
        if not self.hdf5_dir.exists():
            raise FileNotFoundError(f"HDF5 directory does not exist: {self.hdf5_dir}")
        self.sequence_length = int(sequence_length)
        self.image_size = int(image_size)
        self.image_keys = tuple(str(k) for k in image_keys)
        self.proprio_keys = tuple(str(k) for k in (proprio_keys or ()))
        self.stride = max(int(stride), 1)
        self._hdf5_open_kwargs = {"mode": "r", "swmr": True, "libver": "latest"}
        self._file_cache: dict[str, h5py.File] = {}

        files = sorted(self.hdf5_dir.glob("*.hdf5"))
        if max_files is not None:
            files = files[: int(max_files)]
        if not files:
            raise RuntimeError(f"No HDF5 files found in {self.hdf5_dir}")

        self._entries: list[_WindowEntry] = []
        self.action_dim = 0
        self.proprio_dim = 0
        stop = False
        for file_path in files:
            with h5py.File(file_path, **self._hdf5_open_kwargs) as handle:
                data = handle["data"]
                demo_keys = self.list_demo_keys(data)
                if max_demos_per_file is not None:
                    demo_keys = demo_keys[: int(max_demos_per_file)]
                for demo_key in demo_keys:
                    demo = data[demo_key]
                    episode_length = int(demo["actions"].shape[0])
                    if episode_length < self.sequence_length:
                        continue
                    if self.action_dim == 0:
                        self.action_dim = int(demo["actions"].shape[-1])
                    obs_group = demo["obs"]
                    for key in self.image_keys:
                        if key not in obs_group:
                            raise KeyError(f"{file_path}:{demo_key} missing obs/{key}")
                    for key in self.proprio_keys:
                        if key not in obs_group:
                            raise KeyError(f"{file_path}:{demo_key} missing obs/{key}")
                    if self.proprio_keys and self.proprio_dim == 0:
                        self.proprio_dim = sum(
                            int(np.prod(obs_group[key].shape[1:], dtype=np.int64))
                            for key in self.proprio_keys
                        )
                    last_start = episode_length - self.sequence_length
                    for start in range(0, last_start + 1, self.stride):
                        self._entries.append(
                            _WindowEntry(str(file_path), demo_key, start, episode_length)
                        )
                        if max_windows is not None and len(self._entries) >= int(max_windows):
                            stop = True
                            break
                    if stop:
                        break
            if stop:
                break

        if not self._entries:
            raise RuntimeError(
                f"No sequence windows of length {self.sequence_length} under {self.hdf5_dir}"
            )
        if self.action_dim <= 0:
            raise RuntimeError("Could not infer action dimension")
        self._spec = PixelSequenceSpec(
            hdf5_dir=str(self.hdf5_dir),
            num_files=len(files),
            num_windows=len(self._entries),
            sequence_length=self.sequence_length,
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            image_size=self.image_size,
            image_channels=3 * len(self.image_keys),
            image_keys=self.image_keys,
            proprio_keys=self.proprio_keys,
        )

    @property
    def data_spec(self) -> PixelSequenceSpec:
        return self._spec

    def get_normalizer(self) -> dict[str, Any]:
        return {}

    def __len__(self) -> int:
        return len(self._entries)

    def _file(self, path: str) -> h5py.File:
        return self.cached_hdf5_file(self._file_cache, path, self._hdf5_open_kwargs)

    def _resize_images(self, images: torch.Tensor) -> torch.Tensor:
        # images: [T, C, H, W] in [0, 255]
        if images.shape[-2:] == (self.image_size, self.image_size):
            return images
        return F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self._entries[int(index)]
        demo = self._file(entry.file_path)["data"][entry.demo_key]
        start = int(entry.start)
        end = start + self.sequence_length

        frames: list[torch.Tensor] = []
        obs_group = demo["obs"]
        for key in self.image_keys:
            arr = np.asarray(obs_group[key][start:end], dtype=np.uint8)  # [T,H,W,3]
            tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(torch.float32)
            frames.append(tensor)
        images = self._resize_images(torch.cat(frames, dim=1)).contiguous()

        window = np.asarray(demo["actions"][start:end], dtype=np.float32)
        prev_actions = np.zeros((self.sequence_length, window.shape[-1]), dtype=np.float32)
        if self.sequence_length > 1:
            prev_actions[1:] = window[:-1]
        actions = torch.from_numpy(prev_actions)
        # ``window`` is a fresh owned array (HDF5 slice read), so no .copy() needed.
        current_actions = torch.from_numpy(window)

        rewards = torch.from_numpy(np.asarray(demo["rewards"][start:end], dtype=np.float32))
        dones = torch.from_numpy(np.asarray(demo["dones"][start:end], dtype=np.float32))
        is_first = torch.zeros(self.sequence_length, dtype=torch.bool)
        is_first[0] = True
        proprio = None
        if self.proprio_keys:
            proprio_arrays = [
                np.asarray(obs_group[key][start:end], dtype=np.float32).reshape(
                    self.sequence_length, -1
                )
                for key in self.proprio_keys
            ]
            proprio = torch.from_numpy(np.concatenate(proprio_arrays, axis=-1))

        item = {
            "images": images,
            "actions": actions,
            "current_actions": current_actions,
            "rewards": rewards,
            "dones": dones,
            "is_first": is_first,
            "file_path": entry.file_path,
            "demo_key": entry.demo_key,
            "start": start,
        }
        if proprio is not None:
            item["proprio"] = proprio
        return item


@dataclass(frozen=True)
class HDF5ActionChunkSpec:
    hdf5_dir: str
    num_files: int
    num_samples: int
    action_horizon: int
    image_keys: tuple[str, ...]
    one_trajectory_sft: bool = False
    demos_per_task: int | None = None
    demo_selection_seed: int | None = None


@dataclass(frozen=True)
class _HDF5Sample:
    file_path: str
    demo_key: str
    index: int


def _select_demo_keys(
    demo_keys: Sequence[str],
    *,
    file_path: Path,
    demos_per_task: int | None,
    demo_selection_seed: int,
    max_demos_per_file: int | None,
) -> list[str]:
    ordered = list(demo_keys)
    if demos_per_task is not None:
        count = int(demos_per_task)
        if count < 1:
            raise ValueError("demos_per_task must be >= 1 when set.")
        rng = random.Random(f"{int(demo_selection_seed)}:{file_path.name}")
        return sorted(rng.sample(ordered, k=min(count, len(ordered))), key=ordered.index)
    if max_demos_per_file is not None:
        return ordered[: int(max_demos_per_file)]
    return ordered


class HDF5ActionChunkDataset(HDF5Dataset):
    """Raw HDF5 frames and action chunks with episode-local tail padding."""

    def __init__(
        self,
        hdf5_dir: str | Path,
        action_horizon: int = 8,
        image_keys: Sequence[str] = ("agentview_rgb",),
        max_files: int | None = None,
        max_demos_per_file: int | None = None,
        demos_per_task: int | None = None,
        demo_selection_seed: int = 0,
        max_samples: int | None = None,
    ) -> None:
        self.hdf5_dir = Path(hdf5_dir).expanduser().resolve()
        self.action_horizon = int(action_horizon)
        self.image_keys = tuple(str(key) for key in image_keys)
        self.demos_per_task = None if demos_per_task is None else int(demos_per_task)
        self.demo_selection_seed = int(demo_selection_seed)
        self._hdf5_open_kwargs = {"mode": "r", "swmr": True, "libver": "latest"}
        self._file_cache: dict[str, h5py.File] = {}

        files = sorted(self.hdf5_dir.glob("*.hdf5"))
        if max_files is not None:
            files = files[: int(max_files)]
        if not files:
            raise RuntimeError(f"No HDF5 files found under {self.hdf5_dir}")

        self.samples: list[_HDF5Sample] = []
        stop = False
        for file_path in files:
            with h5py.File(file_path, **self._hdf5_open_kwargs) as handle:
                data = handle["data"]
                demo_keys = _select_demo_keys(
                    self.list_demo_keys(data),
                    file_path=file_path,
                    demos_per_task=self.demos_per_task,
                    demo_selection_seed=self.demo_selection_seed,
                    max_demos_per_file=max_demos_per_file,
                )
                for demo_key in demo_keys:
                    demo = data[demo_key]
                    length = int(demo["actions"].shape[0])
                    obs_group = demo["obs"]
                    for key in self.image_keys:
                        if key not in obs_group:
                            raise KeyError(f"{file_path}:{demo_key} missing obs/{key}")
                    for index in range(length):
                        self.samples.append(_HDF5Sample(str(file_path), demo_key, index))
                        if max_samples is not None and len(self.samples) >= int(max_samples):
                            stop = True
                            break
                    if stop:
                        break
            if stop:
                break

        self._spec = HDF5ActionChunkSpec(
            hdf5_dir=str(self.hdf5_dir),
            num_files=len(files),
            num_samples=len(self.samples),
            action_horizon=self.action_horizon,
            image_keys=self.image_keys,
            one_trajectory_sft=self.demos_per_task == 1,
            demos_per_task=self.demos_per_task,
            demo_selection_seed=self.demo_selection_seed
            if self.demos_per_task is not None
            else None,
        )

    @property
    def data_spec(self) -> HDF5ActionChunkSpec:
        return self._spec

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self, path: str) -> h5py.File:
        return self.cached_hdf5_file(self._file_cache, path, self._hdf5_open_kwargs)

    def _action_chunk(self, demo: h5py.Group, index: int) -> np.ndarray:
        actions_ds = demo["actions"]
        length = int(actions_ds.shape[0])
        chunk = np.asarray(actions_ds[index : index + self.action_horizon], dtype=np.float32)
        if index + self.action_horizon > length:
            # Repeat the last frame for the tail past the episode end, matching
            # the previous `np.minimum(arange(...), length - 1)` clamping.
            pad = self.action_horizon - chunk.shape[0]
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)], axis=0)
        return chunk

    def get_normalizer(self) -> None:
        """Return no normalizer; storage values are unmodified."""
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[int(index)]
        demo = self._file(sample.file_path)["data"][sample.demo_key]
        return {
            "images": {
                key: np.asarray(demo["obs"][key][sample.index], dtype=np.uint8)
                for key in self.image_keys
            },
            "actions": self._action_chunk(demo, sample.index),
            "file_path": sample.file_path,
            "demo_key": sample.demo_key,
            "frame_index": sample.index,
        }


__all__ = [
    "HDF5ActionChunkDataset",
    "HDF5ActionChunkSpec",
    "HDF5Dataset",
    "PixelSequenceDataset",
    "PixelSequenceSpec",
]
