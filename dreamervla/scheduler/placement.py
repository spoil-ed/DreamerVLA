"""Single-node placement: map worker roles to GPU/CPU bundles.

Pure logic, intentionally ray-free so it can be unit-tested without a cluster.
Multi-node placement (node ranks, node groups, flexible strategies) is a later
sub-project; ``get_placement`` keeps the same signature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
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


class ResourceMapPlacementStrategy(PlacementStrategy):
    """RLinf-style single-node resource/process rank-map placement.

    Supported forms match the useful single-node subset of RLinf's
    ``resource_ranks:process_ranks`` grammar:

    - ``"0-3"``: one worker per GPU.
    - ``"0:0-3"``: workers 0..3 share GPU 0.
    - ``"0-1:0-3"``: two workers share each GPU.
    - ``"0-3:0-1"``: each worker gets two GPUs.
    - ``"all"``: one worker per visible cluster GPU.
    """

    def __init__(self, rank_map: str) -> None:
        self.rank_map = str(rank_map)

    def get_placement(self, cluster: Cluster) -> list[Placement]:
        rank_map = _parse_rank_map(self.rank_map, cluster.num_gpus)
        process_resources = _rank_map_to_process_resources(rank_map)
        if not process_resources:
            raise ValueError("component placement produced no workers")
        num_workers = max(process_resources) + 1
        placements: list[Placement] = []
        for rank in range(num_workers):
            if rank not in process_resources:
                raise ValueError(
                    f"component placement process ranks must be continuous from 0; missing {rank}"
                )
            gpus = process_resources[rank]
            for gpu in gpus:
                if gpu >= cluster.num_gpus:
                    raise ValueError(
                        f"cluster has {cluster.num_gpus} GPU(s) but placement needs GPU index {gpu}"
                    )
            placements.append(
                Placement(
                    rank=rank,
                    local_rank=rank,
                    local_world_size=num_workers,
                    visible_accelerators=[str(gpu) for gpu in gpus],
                    device=f"cuda:{gpus[0]}",
                )
            )
        return placements


class ComponentPlacement:
    """Parse ``cluster.component_placement`` into per-component strategies.

    This is the DreamerVLA single-node counterpart of RLinf's
    ``HybridComponentPlacement``. It keeps the grammar at the component layer
    instead of inventing per-feature GPU pools.
    """

    def __init__(self, cfg: object) -> None:
        placement_cfg = _select(cfg, "cluster.component_placement", default=None)
        if placement_cfg is None:
            raise ValueError("cluster.component_placement is required")
        plain = _to_plain_mapping(placement_cfg)
        self._strategies: dict[str, ResourceMapPlacementStrategy] = {}
        for raw_names, raw_spec in plain.items():
            placement_spec = _component_placement_string(raw_spec)
            names = [name.strip() for name in str(raw_names).split(",") if name.strip()]
            if not names:
                raise ValueError("component placement name must not be empty")
            strategy = ResourceMapPlacementStrategy(placement_spec)
            for name in names:
                if name in self._strategies:
                    raise ValueError(f"duplicate component placement for {name!r}")
                self._strategies[name] = strategy

    def get_strategy(self, component_name: str) -> ResourceMapPlacementStrategy:
        if component_name not in self._strategies:
            raise ValueError(f"component {component_name!r} is not in component_placement")
        return self._strategies[component_name]

    def has_component(self, component_name: str) -> bool:
        return component_name in self._strategies


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


def _parse_rank_map(rank_map_str: str, num_gpus: int) -> dict[tuple[int, ...], list[int]]:
    rank_map: dict[tuple[int, ...], list[int]] = {}
    parsed_resources: list[int] = []
    parsed_processes: list[int] = []
    for raw_part in str(rank_map_str).split(","):
        part = raw_part.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) not in (1, 2):
            raise ValueError(f"invalid component placement segment {part!r}")
        resources = _parse_rank_config(pieces[0].strip(), num_gpus, "accelerator")
        if not resources:
            raise ValueError(f"empty resource ranks in component placement {part!r}")
        if set(resources).intersection(parsed_resources):
            raise ValueError(f"duplicate resource ranks in component placement {rank_map_str!r}")
        if parsed_resources and resources[0] <= parsed_resources[-1]:
            raise ValueError(
                f"resource ranks must be ascending in component placement {rank_map_str!r}"
            )

        if len(pieces) == 2:
            processes = _parse_rank_config(pieces[1].strip(), None, "process")
        else:
            start = parsed_processes[-1] + 1 if parsed_processes else 0
            processes = list(range(start, start + len(resources)))
        if not processes:
            raise ValueError(f"empty process ranks in component placement {part!r}")
        if set(processes).intersection(parsed_processes):
            raise ValueError(f"duplicate process ranks in component placement {rank_map_str!r}")
        if parsed_processes and processes[0] != parsed_processes[-1] + 1:
            raise ValueError(
                f"process ranks must be continuous in component placement {rank_map_str!r}"
            )
        if processes != list(range(processes[0], processes[0] + len(processes))):
            raise ValueError(
                f"process ranks must be continuous in component placement {rank_map_str!r}"
            )
        if len(processes) % len(resources) and len(resources) % len(processes):
            raise ValueError(
                "resource and process counts must divide each other in component "
                f"placement segment {part!r}"
            )
        parsed_resources.extend(resources)
        parsed_processes.extend(processes)
        rank_map[tuple(resources)] = processes
    return rank_map


def _parse_rank_config(value: str, num_gpus: int | None, label: str) -> list[int]:
    value = str(value).strip()
    if value == "all":
        if num_gpus is None:
            raise ValueError(f"{label} ranks cannot use 'all'")
        return list(range(int(num_gpus)))
    return parse_accelerator_range(value)


def _rank_map_to_process_resources(
    rank_map: dict[tuple[int, ...], list[int]]
) -> dict[int, list[int]]:
    process_resources: dict[int, list[int]] = {}
    for resource_ranks, process_ranks in rank_map.items():
        resources = list(resource_ranks)
        if len(resources) >= len(process_ranks):
            resources_per_process = len(resources) // len(process_ranks)
            for index, process_rank in enumerate(process_ranks):
                process_resources[process_rank] = resources[
                    index * resources_per_process : (index + 1) * resources_per_process
                ]
        else:
            processes_per_resource = len(process_ranks) // len(resources)
            for resource_index, resource_rank in enumerate(resources):
                start = resource_index * processes_per_resource
                stop = start + processes_per_resource
                for process_rank in process_ranks[start:stop]:
                    process_resources.setdefault(process_rank, []).append(resource_rank)
    return process_resources


def _component_placement_string(spec: object) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, Mapping):
        if "placement" not in spec:
            raise ValueError(f"component placement mapping missing 'placement': {spec}")
        return str(spec["placement"])
    return str(spec)


def _to_plain_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    try:
        from omegaconf import OmegaConf

        plain = OmegaConf.to_container(value, resolve=True)
    except Exception as exc:  # noqa: BLE001
        raise TypeError("component_placement must be a mapping") from exc
    if not isinstance(plain, Mapping):
        raise TypeError("component_placement must be a mapping")
    return plain


def _select(cfg: object, path: str, *, default: object = None) -> object:
    try:
        from omegaconf import OmegaConf

        value = OmegaConf.select(cfg, path, default=None)
        return default if value is None else value
    except Exception:  # noqa: BLE001
        cur = cfg
        for part in path.split("."):
            if isinstance(cur, Mapping):
                if part not in cur:
                    return default
                cur = cur[part]
            else:
                if not hasattr(cur, part):
                    return default
                cur = getattr(cur, part)
        return cur


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
