#!/usr/bin/env python3
"""Random-init world-model overfit check on one LIBERO trajectory."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dreamervla.utils.paths import data_path

DEFAULT_HDF5_FILENAME = "open_the_middle_drawer_of_the_cabinet_demo.hdf5"


@dataclass(frozen=True)
class EpisodeArrays:
    """Aligned arrays loaded from one hidden/raw LIBERO demo."""

    hidden: np.ndarray
    lang: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    proprio: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            int(self.hidden.shape[0]),
            int(self.actions.shape[0]),
            int(self.rewards.shape[0]),
            int(self.proprio.shape[0]),
        }
        if len(lengths) != 1:
            raise ValueError("episode arrays must have the same leading length")

    @property
    def episode_len(self) -> int:
        """Return the aligned episode length."""

        return int(self.hidden.shape[0])


@dataclass(frozen=True)
class RunSettings:
    """Optimization and convergence settings for one overfit run."""

    max_epochs: int = 200
    batch_size: int = 8
    lr: float = 1.0e-4
    grad_clip: float = 1.0
    eval_every: int = 5
    mse_threshold: float = 0.03
    cosine_threshold: float = 0.95
    required_passes: int = 3
    seed: int = 23

    def __post_init__(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr < 0.0:
            raise ValueError("lr must be non-negative")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.mse_threshold < 0.0:
            raise ValueError("mse_threshold must be non-negative")
        if not -1.0 <= self.cosine_threshold <= 1.0:
            raise ValueError("cosine_threshold must be in [-1, 1]")
        if self.required_passes <= 0:
            raise ValueError("required_passes must be positive")


@dataclass
class ConvergenceTracker:
    """Track consecutive full-evaluation threshold passes."""

    mse_threshold: float
    cosine_threshold: float
    required_passes: int
    streak: int = 0

    def __post_init__(self) -> None:
        if self.mse_threshold < 0.0:
            raise ValueError("mse_threshold must be non-negative")
        if not -1.0 <= self.cosine_threshold <= 1.0:
            raise ValueError("cosine_threshold must be in [-1, 1]")
        if self.required_passes <= 0:
            raise ValueError("required_passes must be positive")

    def observe(self, *, mse: float, cosine_similarity: float) -> bool:
        """Record one evaluation and return whether convergence is confirmed."""

        passed = (
            mse <= self.mse_threshold
            and cosine_similarity >= self.cosine_threshold
        )
        self.streak = self.streak + 1 if passed else 0
        return self.streak >= self.required_passes


def sliding_window_starts(*, episode_len: int, sequence_len: int) -> np.ndarray:
    """Return every valid sliding-window start for one episode."""

    count = episode_len - sequence_len + 1
    if count <= 0:
        raise ValueError(
            f"episode length {episode_len} is shorter than sequence length "
            f"{sequence_len}"
        )
    return np.arange(count, dtype=np.int64)


def iter_epoch_batches(
    starts: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    """Yield a shuffled epoch in batches, visiting each start exactly once."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    shuffled = rng.permutation(starts)
    for offset in range(0, len(shuffled), batch_size):
        yield shuffled[offset : offset + batch_size]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the one-command overfit diagnostic arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--task", default="openvla_onetraj_libero")
    parser.add_argument("--hdf5-filename", default=DEFAULT_HDF5_FILENAME)
    parser.add_argument("--hidden-hdf5", type=Path, default=None)
    parser.add_argument("--raw-hdf5", type=Path, default=None)
    parser.add_argument("--demo-key", default="demo_0")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=data_path("outputs/world_model_probe/single_trajectory_overfit"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--mse-threshold", type=float, default=0.03)
    parser.add_argument("--cosine-threshold", type=float, default=0.95)
    parser.add_argument("--required-passes", type=int, default=3)
    return parser.parse_args(argv)
