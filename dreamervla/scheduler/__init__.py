"""Ray scheduler primitives (single-node subset, RLinf-style naming).

Opt-in distributed backend. Importing this package requires ``ray``; the
single-machine torchrun path must not import it (see S1 design spec).
"""

from __future__ import annotations

from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.collective import AsyncResult, TorchCollectiveGroup
from dreamervla.scheduler.hardware import (
    AcceleratorInfo,
    count_local_accelerators,
    discover_local_accelerators,
)
from dreamervla.scheduler.placement import (
    FlexiblePlacementStrategy,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Placement,
    PlacementStrategy,
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
    "FlexiblePlacementStrategy",
    "NodePlacementStrategy",
    "PackedPlacementStrategy",
    "Placement",
    "PlacementStrategy",
    "TorchCollectiveGroup",
    "count_local_accelerators",
    "discover_local_accelerators",
    "parse_accelerator_range",
    "Worker",
    "WorkerGroup",
    "WorkerGroupFunc",
    "WorkerGroupFuncResult",
]
