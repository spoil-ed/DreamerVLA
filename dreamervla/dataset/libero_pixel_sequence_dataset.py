from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dreamervla.dataset.base_dataset import BaseDataset


@dataclass(frozen=True)
class LIBEROPixelSequenceSpec:
    hdf5_dir: str
    num_files: int
    num_windows: int
    sequence_length: int
    action_dim: int
    image_size: int
    image_channels: int
    image_keys: tuple[str, ...]


@dataclass(frozen=True)
class _WindowEntry:
    file_path: str
    demo_key: str
    start: int
    episode_length: int


class LIBEROPixelSequenceDataset(BaseDataset):
    """Pixel-level LIBERO sequence windows for DreamerV3-style WM training.

    Reads Robomimic/LIBERO HDF5 demonstrations directly and returns:

      images:    [T, C, H, W] float32 in the uint8 range [0, 255]
      actions:   [T, A] previous-action convention, actions[0] is zero
      current_actions: [T, A] action executed from this observation
      rewards:   [T]
      dones:     [T]
      is_first:  [T], always true at the first item of each sampled window

    The two default image keys are LIBERO's third-person view and wrist view:
    ``agentview_rgb`` and ``eye_in_hand_rgb``. They are concatenated along the
    channel dimension, matching DreamerV3's "multiple image keys concatenate
    channels before the CNN" behavior.
    """

    def __init__(
        self,
        hdf5_dir: str | Path,
        sequence_length: int = 32,
        image_size: int = 64,
        image_keys: Sequence[str] = ("agentview_rgb", "eye_in_hand_rgb"),
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
                    last_start = episode_length - self.sequence_length
                    for start in range(0, last_start + 1, self.stride):
                        self._entries.append(
                            _WindowEntry(
                                str(file_path), demo_key, start, episode_length
                            )
                        )
                        if max_windows is not None and len(self._entries) >= int(
                            max_windows
                        ):
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
        self._spec = LIBEROPixelSequenceSpec(
            hdf5_dir=str(self.hdf5_dir),
            num_files=len(files),
            num_windows=len(self._entries),
            sequence_length=self.sequence_length,
            action_dim=self.action_dim,
            image_size=self.image_size,
            image_channels=3 * len(self.image_keys),
            image_keys=self.image_keys,
        )

    @property
    def data_spec(self) -> LIBEROPixelSequenceSpec:
        return self._spec

    def get_normalizer(self) -> dict[str, Any]:
        return {}

    def __len__(self) -> int:
        return len(self._entries)

    def _file(self, path: str) -> h5py.File:
        handle = self._file_cache.get(path)
        if handle is None:
            handle = h5py.File(path, **self._hdf5_open_kwargs)
            self._file_cache[path] = handle
        return handle

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

        raw_actions = np.asarray(demo["actions"], dtype=np.float32)
        prev_actions = np.zeros(
            (self.sequence_length, raw_actions.shape[-1]), dtype=np.float32
        )
        if self.sequence_length > 1:
            prev_actions[1:] = raw_actions[start : end - 1]
        actions = torch.from_numpy(prev_actions)
        current_actions = torch.from_numpy(raw_actions[start:end].copy())

        rewards = torch.from_numpy(
            np.asarray(demo["rewards"][start:end], dtype=np.float32)
        )
        dones = torch.from_numpy(np.asarray(demo["dones"][start:end], dtype=np.float32))
        is_first = torch.zeros(self.sequence_length, dtype=torch.bool)
        is_first[0] = True

        return {
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


__all__ = ["LIBEROPixelSequenceDataset", "LIBEROPixelSequenceSpec"]
