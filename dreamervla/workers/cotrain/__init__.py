from __future__ import annotations

import importlib

from dreamervla.workers.cotrain.placement import (
    ManualCotrainPlacementPlan,
    RolePlacement,
    build_manual_cotrain_placement,
)

_MESSAGE_EXPORTS = {
    "ObservationBatchMsg",
    "ObservationMsg",
    "RolloutResultBatchMsg",
    "RolloutResultMsg",
    "StopMsg",
    "TrajectoryBatch",
    "TrajectoryShard",
    "as_tensor",
    "collate_trajectory_shards",
}

__all__ = [
    "ManualCotrainPlacementPlan",
    "ObservationBatchMsg",
    "ObservationMsg",
    "RolePlacement",
    "RolloutResultBatchMsg",
    "RolloutResultMsg",
    "StopMsg",
    "TrajectoryBatch",
    "TrajectoryShard",
    "as_tensor",
    "build_manual_cotrain_placement",
    "collate_trajectory_shards",
]


def __getattr__(name: str) -> object:
    if name not in _MESSAGE_EXPORTS:
        raise AttributeError(name)

    messages = importlib.import_module("dreamervla.workers.cotrain.messages")
    value = getattr(messages, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_MESSAGE_EXPORTS])
