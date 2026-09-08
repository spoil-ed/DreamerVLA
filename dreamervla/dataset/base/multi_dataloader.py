"""Composition and reproducible weighted sampling of configured datasets."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Sequence
from itertools import accumulate
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from dreamervla.dataset.base.base_dataloader import BaseDataset


class MultiDataset(BaseDataset):
    """Concatenate map-style sources without changing their sample contracts.

    Hydra can instantiate the nested ``datasets`` list. Uniform index sampling
    follows source sizes; use DistributedMixtureSampler for source-level weights.
    Callers select compatible source schemas and a suitable collate function.
    """

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = list(datasets)
        self.lengths = tuple(len(dataset) for dataset in self.datasets)
        if not self.lengths or any(length <= 0 for length in self.lengths):
            raise ValueError("MultiDataset requires at least one source, all nonempty")
        self.cumulative_sizes = tuple(accumulate(self.lengths))

    @property
    def data_spec(self) -> dict[str, Any]:
        """Describe each source independently, without inferring shared shapes."""
        return {
            "lengths": self.lengths,
            "sources": [getattr(dataset, "data_spec", None) for dataset in self.datasets],
        }

    def get_normalizer(self) -> dict[str, Any]:
        """Keep source normalization metadata separate."""
        return {
            "sources": [
                dataset.get_normalizer()
                if callable(getattr(dataset, "get_normalizer", None))
                else None
                for dataset in self.datasets
            ]
        }

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source = bisect_right(self.cumulative_sizes, index)
        start = 0 if source == 0 else self.cumulative_sizes[source - 1]
        return self.datasets[source][index - start]


class DistributedMixtureSampler(Sampler[int]):
    """Draw sources by weight, then samples uniformly within the chosen source.

    Sampling is with replacement. Every rank deterministically constructs the
    same global epoch and takes its own strided partition. ``num_samples`` is
    the global epoch length and must be divisible by ``num_replicas``. The seed
    and epoch fully determine the draws, so restoring an epoch reproduces it.
    """

    def __init__(
        self,
        dataset: MultiDataset,
        *,
        weights: Sequence[float],
        num_samples: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.float64).clone()
        self.num_samples = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        if self.weights.ndim != 1 or len(self.weights) != len(dataset.datasets):
            raise ValueError("weights must contain one value per dataset")
        if (
            not bool(torch.isfinite(self.weights).all())
            or bool((self.weights < 0).any())
            or not bool((self.weights > 0).any())
        ):
            raise ValueError("weights must be finite, nonnegative and not all zero")
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("num_replicas must be positive and rank must identify a replica")
        if self.num_samples <= 0 or self.num_samples % self.num_replicas:
            raise ValueError("num_samples must be positive and divisible by num_replicas")

    def set_epoch(self, epoch: int) -> None:
        """Select the reproducible epoch used by distributed training/resume."""
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        sources = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=generator
        )
        indices = torch.empty(self.num_samples, dtype=torch.int64)
        start = 0
        for source, length in enumerate(self.dataset.lengths):
            positions = sources == source
            indices[positions] = start + torch.randint(
                length, (int(positions.sum()),), generator=generator
            )
            start += length
        yield from indices[self.rank :: self.num_replicas].tolist()


__all__ = ["DistributedMixtureSampler", "MultiDataset"]
