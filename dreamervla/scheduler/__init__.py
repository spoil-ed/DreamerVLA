"""Ray scheduler primitives (single-node subset, RLinf-style naming).

Opt-in distributed backend. Importing this package requires ``ray``; the
single-machine torchrun path must not import it (see S1 design spec).
"""

from __future__ import annotations

from dreamervla.scheduler.channel import AsyncWork, Channel
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.collective import AsyncResult, TorchCollectiveGroup
from dreamervla.scheduler.dynamic_scheduler import ComponentScheduler, ScheduledWork
from dreamervla.scheduler.hardware import (
    AcceleratorInfo,
    count_local_accelerators,
    discover_local_accelerators,
)
from dreamervla.scheduler.manager import DeviceLockManager, WorkerManager, WorkerRoute
from dreamervla.scheduler.node import NodeInfo, discover_ray_nodes, probe_local_node
from dreamervla.scheduler.placement import (
    ComponentPlacement,
    FlexiblePlacementStrategy,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Placement,
    PlacementStrategy,
    ResourceMapPlacementStrategy,
    parse_accelerator_range,
)
from dreamervla.scheduler.worker import Worker
from dreamervla.scheduler.worker_group import (
    WorkerGroup,
    WorkerGroupFunc,
    WorkerGroupFuncResult,
)

__all__ = [
    "Channel",
    "Cluster",
    "AcceleratorInfo",
    "AsyncResult",
    "AsyncWork",
    "ComponentPlacement",
    "ComponentScheduler",
    "DeviceLockManager",
    "FlexiblePlacementStrategy",
    "NodeInfo",
    "NodePlacementStrategy",
    "PackedPlacementStrategy",
    "Placement",
    "PlacementStrategy",
    "ResourceMapPlacementStrategy",
    "ScheduledWork",
    "TorchCollectiveGroup",
    "WorkerManager",
    "WorkerRoute",
    "count_local_accelerators",
    "discover_ray_nodes",
    "discover_local_accelerators",
    "parse_accelerator_range",
    "probe_local_node",
    "Worker",
    "WorkerGroup",
    "WorkerGroupFunc",
    "WorkerGroupFuncResult",
]
