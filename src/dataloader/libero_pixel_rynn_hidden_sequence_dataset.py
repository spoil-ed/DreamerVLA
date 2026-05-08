from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch

from src.dataloader.libero_pixel_sequence_dataset import (
    LIBEROPixelSequenceDataset,
)


class LIBEROPixelRynnHiddenSequenceDataset(LIBEROPixelSequenceDataset):
    """LIBERO pixel windows plus precomputed RynnVLA hidden observations.

    The original pixel HDF5 files remain the image/reconstruction source.  This
    dataset reads a sidecar HDF5 directory with matching filenames and per-demo
    ``data/<demo_key>/obs_embedding`` arrays, then returns both:

      images:        [T, C, H, W], uint8-range float tensor from the source HDF5
      obs_embedding: [T, D], precomputed frozen RynnVLA hidden vector
    """

    def __init__(
        self,
        hdf5_dir: str | Path,
        hidden_dir: str | Path,
        sequence_length: int = 32,
        image_size: int = 256,
        image_keys: Sequence[str] = ("agentview_rgb", "eye_in_hand_rgb"),
        hidden_key: str = "obs_embedding",
        max_files: int | None = None,
        max_demos_per_file: int | None = None,
        max_windows: int | None = None,
        stride: int = 1,
    ) -> None:
        super().__init__(
            hdf5_dir=hdf5_dir,
            sequence_length=sequence_length,
            image_size=image_size,
            image_keys=image_keys,
            max_files=max_files,
            max_demos_per_file=max_demos_per_file,
            max_windows=max_windows,
            stride=stride,
        )
        self.hidden_dir = self.resolve_project_path(hidden_dir)
        if not self.hidden_dir.exists():
            raise FileNotFoundError(f"Rynn hidden sidecar directory does not exist: {self.hidden_dir}")
        self.hidden_key = str(hidden_key)
        self._hidden_file_cache: dict[str, h5py.File] = {}

    def _hidden_path_for_source(self, source_path: str | Path) -> Path:
        return self.hidden_dir / Path(source_path).name

    def _hidden_file(self, source_path: str | Path) -> h5py.File:
        hidden_path = self._hidden_path_for_source(source_path)
        key = str(hidden_path)
        handle = self._hidden_file_cache.get(key)
        if handle is None:
            if not hidden_path.is_file():
                raise FileNotFoundError(
                    f"Missing Rynn hidden sidecar for {source_path}: {hidden_path}"
                )
            handle = h5py.File(hidden_path, mode="r", swmr=True, libver="latest")
            self._hidden_file_cache[key] = handle
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
        item["hidden_path"] = str(self._hidden_path_for_source(entry.file_path))
        return item


__all__ = ["LIBEROPixelRynnHiddenSequenceDataset"]
