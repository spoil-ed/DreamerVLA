"""Ray scheduler primitives (single-node subset, RLinf-style naming).

Opt-in distributed backend. Importing this package requires ``ray``; the
single-machine torchrun path must not import it (see S1 design spec).
"""

from __future__ import annotations

from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import (
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Placement,
    PlacementStrategy,
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
    "NodePlacementStrategy",
    "PackedPlacementStrategy",
    "Placement",
    "PlacementStrategy",
    "Worker",
    "WorkerGroup",
    "WorkerGroupFunc",
    "WorkerGroupFuncResult",
]
