"""Base class for Ray workers in the optional online backend."""

from __future__ import annotations

import os
from typing import Any


class Worker:
    """Common rank/device metadata for Ray actor workers."""

    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_world_size = self.world_size
        self.visible_accelerators = self._visible_accelerators()
        self.device = "cuda:0" if self.visible_accelerators else "cpu"

    def init(self) -> None:
        """Hook for heavy actor-local initialization."""

    @classmethod
    def create_group(cls, *args: Any, **kwargs: Any):
        """Create an unlaunched WorkerGroup for this worker class."""

        from dreamervla.scheduler.worker_group import WorkerGroup

        return WorkerGroup(cls, *args, **kwargs)

    @staticmethod
    def _visible_accelerators() -> list[str]:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not raw or raw == "-1":
            return []
        return [part.strip() for part in raw.split(",") if part.strip()]
