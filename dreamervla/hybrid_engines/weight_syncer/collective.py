"""Collective-capable weight synchronization."""

from __future__ import annotations

from typing import Any

import torch

from dreamervla.hybrid_engines.weight_syncer.base import WeightSyncer
from dreamervla.hybrid_engines.weight_syncer.objectstore import ObjectStoreWeightSyncer
from dreamervla.scheduler.collective import TorchCollectiveGroup


class CollectiveWeightSyncer(WeightSyncer):
    """Weight syncer with torch.distributed broadcast support.

    The versioned ``push``/``pull`` API remains compatible with the existing
    object-store implementation. Distributed callers that participate in the
    same process group can use ``broadcast_model`` for NCCL/Gloo synchronization.
    """

    def __init__(
        self,
        store_name: str = "DreamerVLACollectiveWeightStore",
        *,
        group: TorchCollectiveGroup | None = None,
    ) -> None:
        self.fallback = ObjectStoreWeightSyncer(store_name=store_name)
        self.group = group or TorchCollectiveGroup()

    def push(self, key: str, state_dict: dict[str, Any], version: int) -> None:
        self.fallback.push(key, state_dict, version)

    def pull(self, key: str, model: torch.nn.Module, local_version: int) -> int | None:
        return self.fallback.pull(key, model, local_version)

    def broadcast_model(
        self,
        model: torch.nn.Module,
        *,
        src: int = 0,
    ) -> None:
        state = self.group.broadcast_state_dict(model.state_dict(), src=src)
        model.load_state_dict(state)


__all__ = ["CollectiveWeightSyncer"]
