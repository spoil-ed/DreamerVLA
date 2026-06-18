"""Single-node placement: map worker roles to GPU/CPU bundles.

Pure logic, intentionally ray-free so it can be unit-tested without a cluster.
Multi-node placement (node ranks, node groups, flexible strategies) is a later
sub-project; ``get_placement`` keeps the same signature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
