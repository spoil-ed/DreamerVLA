"""CollectedRolloutClassifierDataset — classifier reader for cold-start rollout dumps.

Reads the same reward-dir-compatible HDF5 dump (reward HDF5 + obs_embedding
sidecar) produced by the cold-start collector and adds a binary success label
derived from the window reaching a terminal success frame (sparse_rewards==1).

Success criterion (per-window):
  success=1.0  if the window contains a terminal success frame (sparse_rewards==1)
  success=0.0  otherwise — in particular EVERY window of a failed episode
               (sparse_rewards all 0) is 0.0, including its terminal-ending window.

This is the actual-success signal (the same ``sparse_rewards`` that
BalancedTerminalDataset turns into its reward / success_to_go), NOT the structural
``is_positive_window`` flag (``end == episode_length``), which marks any
terminal-ending window regardless of whether the episode succeeded or failed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dreamervla.dataset.pixel_hidden_sequence_dataset import PixelHiddenSequenceDataset


class CollectedRolloutClassifierDataset(PixelHiddenSequenceDataset):
    """Pixel-hidden sequence dataset with a binary success label.

    Subclasses PixelHiddenSequenceDataset unchanged; the only addition is
    ``item["success"]`` in __getitem__, derived from whether the window reaches
    a terminal success frame (sparse_rewards==1).
    """

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        entry = self._entries[int(index)]
        start = int(entry.start)
        end = start + self.sequence_length
        demo = self._file(entry.file_path)["data"][entry.demo_key]
        key = "sparse_rewards" if "sparse_rewards" in demo else "rewards"
        window = np.asarray(demo[key][start:end], dtype=np.float32)
        item["success"] = float(window.max() >= 1.0)
        return item


__all__ = ["CollectedRolloutClassifierDataset"]
