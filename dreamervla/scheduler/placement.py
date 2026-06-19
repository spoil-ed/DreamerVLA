"""Single-node placement: map worker roles to GPU/CPU bundles.

Pure logic, intentionally ray-free so it can be unit-tested without a cluster.
Multi-node placement (node ranks, node groups, flexible strategies) is a later
sub-project; ``get_placement`` keeps the same signature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamervla.scheduler.cluster import Cluster


@dataclass(frozen=True)
class Placement:
    """Allocation for a single worker rank on one node."""

    rank: int
    local_rank: int
    local_world_size: int
    visible_accelerators: list[str]  # written to CUDA_VISIBLE_DEVICES; [] for CPU workers
    device: str  # "cuda:{gpu}" or "cpu"


class PlacementStrategy(ABC):
    @abstractmethod
    def get_placement(self, cluster: Cluster) -> list[Placement]:
        """Produce one Placement per worker rank, given the cluster's resources."""


class PackedPlacementStrategy(PlacementStrategy):
    """Pack contiguous GPUs in ``[start_gpu, end_gpu]`` into workers."""

    def __init__(self, start_gpu: int, end_gpu: int, num_gpus_per_worker: int = 1) -> None:
        if num_gpus_per_worker < 1:
            raise ValueError(
                f"num_gpus_per_worker must be >= 1, got {num_gpus_per_worker}"
            )
        if start_gpu < 0 or end_gpu < start_gpu:
            raise ValueError(f"invalid GPU range [{start_gpu}, {end_gpu}]")
        span = end_gpu - start_gpu + 1
        if span % num_gpus_per_worker != 0:
            raise ValueError(
                f"GPU span {span} not divisible by num_gpus_per_worker {num_gpus_per_worker}"
            )
        self.start_gpu = start_gpu
        self.end_gpu = end_gpu
        self.num_gpus_per_worker = num_gpus_per_worker

    def get_placement(self, cluster: Cluster) -> list[Placement]:
        if self.end_gpu >= cluster.num_gpus:
            raise ValueError(
                f"cluster has {cluster.num_gpus} GPU(s) but placement needs GPU index "
                f"up to {self.end_gpu}"
            )
        num_workers = (self.end_gpu - self.start_gpu + 1) // self.num_gpus_per_worker
        placements: list[Placement] = []
        for w in range(num_workers):
            base = self.start_gpu + w * self.num_gpus_per_worker
            gpus = [base + j for j in range(self.num_gpus_per_worker)]
            placements.append(
                Placement(
                    rank=w,
                    local_rank=w,
                    local_world_size=num_workers,
                    visible_accelerators=[str(g) for g in gpus],
                    device=f"cuda:{gpus[0]}",
                )
            )
        return placements


def parse_accelerator_range(value: str) -> list[int]:
    """Parse a compact accelerator range such as ``"0-2,4"``.

    This is a manual placement helper: it expands user-provided resource ranks
    for validation and worker placement; it does not infer counts or tune them.
    """
    ranks: list[int] = []
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start < 0 or end < start:
                raise ValueError(f"invalid accelerator range {part!r}")
            ranks.extend(range(start, end + 1))
        else:
            rank = int(part)
            if rank < 0:
                raise ValueError(f"accelerator rank must be >= 0, got {rank}")
            ranks.append(rank)
    if not ranks:
        raise ValueError("accelerator range must not be empty")
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"accelerator range contains duplicate ranks: {value!r}")
    return ranks


class FlexiblePlacementStrategy(PlacementStrategy):
    """Place workers on explicit, non-contiguous GPU groups on one node."""

    def __init__(self, accelerator_groups: Sequence[str | Sequence[int]]) -> None:
        if not accelerator_groups:
            raise ValueError("accelerator_groups must not be empty")
        groups = [_normalize_accelerator_group(group) for group in accelerator_groups]
        seen: set[int] = set()
        for group in groups:
            overlap = seen.intersection(group)
            if overlap:
                raise ValueError(f"accelerator groups contain duplicate ranks: {sorted(overlap)}")
            seen.update(group)
        self.accelerator_groups = sorted(groups, key=lambda group: group[0])

    def get_placement(self, cluster: Cluster) -> list[Placement]:
        for group in self.accelerator_groups:
            for gpu in group:
                if gpu >= cluster.num_gpus:
                    raise ValueError(
                        f"cluster has {cluster.num_gpus} GPU(s) but placement needs GPU index {gpu}"
                    )
        local_world_size = len(self.accelerator_groups)
        return [
            Placement(
                rank=rank,
                local_rank=rank,
                local_world_size=local_world_size,
                visible_accelerators=[str(gpu) for gpu in group],
                device=f"cuda:{group[0]}",
            )
            for rank, group in enumerate(self.accelerator_groups)
        ]


def _normalize_accelerator_group(group: str | Sequence[int]) -> list[int]:
    if isinstance(group, str):
        ranks = parse_accelerator_range(group)
    else:
        ranks = [int(rank) for rank in group]
    if not ranks:
        raise ValueError("accelerator group must not be empty")
    if any(rank < 0 for rank in ranks):
        raise ValueError(f"accelerator ranks must be >= 0, got {ranks}")
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"accelerator group contains duplicate ranks: {ranks}")
    return sorted(ranks)


class NodePlacementStrategy(PlacementStrategy):
    """Place ``count`` CPU-only workers (no GPU affinity)."""

    def __init__(self, count: int) -> None:
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        self.count = count

    def get_placement(self, cluster: Cluster) -> list[Placement]:
        return [
            Placement(
                rank=i,
                local_rank=i,
                local_world_size=self.count,
                visible_accelerators=[],
                device="cpu",
            )
            for i in range(self.count)
        ]
