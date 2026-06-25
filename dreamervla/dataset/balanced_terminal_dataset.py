"""Balanced positive/negative window dataset for reward-head fine-tuning.

For each demo of length T, a window starting at s spans [s, s+W). We call:
  * positive  if s + W == T  → the last frame of the window IS the terminal step
  * negative  if s + W <  T  → window does not contain the terminal step

The dataset also rewrites the ``rewards`` field with ``sparse_rewards`` (typically
0 everywhere except a single 1 at the terminal step) so the WM reward head can
be retrained as a "success-terminal classifier" in the LUMOS style.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dreamervla.dataset.pixel_hidden_sequence_dataset import (
    PixelHiddenSequenceDataset,
)


class BalancedTerminalDataset(PixelHiddenSequenceDataset):
    """Balanced terminal-aware sequence dataset.

    reward_mode controls how ``rewards`` is rewritten:
      * ``"sparse"`` (default): use HDF5 ``sparse_rewards`` slice (≈ only 1 at terminal).
      * ``"per_window_dense"``: LUMOS-style. If window ends at episode terminal,
        ALL ``W`` steps get reward=1; otherwise all 0. Gives 8× denser positive
        signal so the reward_head sees "I am within W steps of success" as
        positive, not just "I am the terminal frame".
      * ``"from_hdf5"``: pass-through. Use the HDF5 ``rewards`` field as-is
        (sliced to the window). Intended for datasets that already shape the
        per-step reward (e.g. ``progress_delta`` telescoping deltas, or
        ``remaining_steps`` success-to-go). Does not modify the value.
    """

    def __init__(
        self,
        *args: Any,
        reward_mode: str = "sparse",
        success_to_go_discount: float = 0.97,
        balanced_length: int = 0,
        balanced_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reward_mode = str(reward_mode).lower()
        self.success_to_go_discount = float(success_to_go_discount)
        if self.reward_mode not in {"sparse", "per_window_dense", "from_hdf5"}:
            raise ValueError(
                f"reward_mode must be one of 'sparse'|'per_window_dense'|'from_hdf5', got {reward_mode!r}"
            )
        if self.success_to_go_discount < 0.0:
            raise ValueError(
                f"success_to_go_discount must be non-negative, got {self.success_to_go_discount}"
            )
        self.balanced_length = int(balanced_length)
        self._balanced_seed = int(balanced_seed)
        self.positive_indices: list[int] = []
        self.negative_indices: list[int] = []
        for i, entry in enumerate(self._entries):
            end = entry.start + self.sequence_length
            if end == entry.episode_length:
                self.positive_indices.append(i)
            elif end < entry.episode_length:
                self.negative_indices.append(i)
        print(
            f"[balanced-dataset] reward_mode={self.reward_mode}  "
            f"balanced_length={self.balanced_length}  "
            f"{len(self.positive_indices)} positive / "
            f"{len(self.negative_indices)} negative / total entries={len(self._entries)}",
            flush=True,
        )
        if not self.positive_indices:
            raise RuntimeError("dataset has no positive (terminal-ending) windows")
        if not self.negative_indices:
            raise RuntimeError("dataset has no negative (non-terminal) windows")

    def __len__(self) -> int:
        if self.balanced_length > 0:
            return self.balanced_length
        return super().__len__()

    def __getitem__(self, index: int) -> dict[str, Any]:
        # If balanced mode: deterministically map virtual index -> real entry index,
        # alternating positive/negative pools. Each virtual index always maps to
        # the SAME real entry, so DistributedSampler/torch shuffle works correctly.
        if self.balanced_length > 0:
            rng = np.random.RandomState(self._balanced_seed + int(index))
            if (int(index) & 1) == 0:
                real_index = int(rng.choice(self.positive_indices))
            else:
                real_index = int(rng.choice(self.negative_indices))
        else:
            real_index = int(index)
        item = super().__getitem__(real_index)
        entry = self._entries[real_index]
        demo = self._file(entry.file_path)["data"][entry.demo_key]
        start = int(entry.start)
        end = start + self.sequence_length
        is_positive = bool(end == entry.episode_length)
        if "sparse_rewards" in demo:
            sparse_rewards = np.asarray(
                demo["sparse_rewards"][start:end], dtype=np.float32
            )
        else:
            sparse_rewards = np.asarray(demo["rewards"][start:end], dtype=np.float32)
        if self.reward_mode == "per_window_dense":
            # LUMOS-style: every step in a positive window labeled 1; negative window all 0
            rewards = np.full(
                (self.sequence_length,), float(is_positive), dtype=np.float32
            )
        elif self.reward_mode == "from_hdf5":
            # Pass-through HDF5 rewards (shaped per-step, e.g. progress-delta).
            # Falls back to sparse_rewards when no separate rewards field exists.
            if "rewards" in demo:
                rewards = np.asarray(demo["rewards"][start:end], dtype=np.float32)
            else:
                rewards = sparse_rewards
        else:
            # sparse: use HDF5 sparse_rewards (typically [0,...,0,1] only at terminal)
            rewards = sparse_rewards
        success_to_go = np.zeros_like(sparse_rewards, dtype=np.float32)
        running = 0.0
        gamma = float(self.success_to_go_discount)
        for offset in range(self.sequence_length - 1, -1, -1):
            running = float(sparse_rewards[offset]) + gamma * running
            success_to_go[offset] = min(running, 1.0)
        item["rewards"] = torch.from_numpy(rewards)
        item["success_to_go"] = torch.from_numpy(success_to_go)
        item["is_positive_window"] = is_positive
        return item


class BalancedTerminalSampler(torch.utils.data.Sampler):
    """Yields a stream of dataset indices alternating between positive and
    negative pools with target ratio (default 1:1).
    """

    def __init__(
        self,
        dataset: BalancedTerminalDataset,
        num_samples: int = 100000,
        positive_ratio: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.pos = list(dataset.positive_indices)
        self.neg = list(dataset.negative_indices)
        self.num_samples = int(num_samples)
        self.positive_ratio = float(positive_ratio)
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        for _ in range(self.num_samples):
            if self.rng.random() < self.positive_ratio:
                yield int(self.rng.choice(self.pos))
            else:
                yield int(self.rng.choice(self.neg))

    def __len__(self) -> int:
        return self.num_samples


__all__ = ["BalancedTerminalDataset", "BalancedTerminalSampler"]
