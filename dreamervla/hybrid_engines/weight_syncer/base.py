"""Weight synchronization interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class WeightSyncer(ABC):
    """Versioned state_dict synchronization interface."""

    @abstractmethod
    def push(self, key: str, state_dict: dict[str, Any], version: int) -> None:
        """Publish a state_dict under ``key`` at ``version``."""

    @abstractmethod
    def pull(self, key: str, model: torch.nn.Module, local_version: int) -> int | None:
        """Load newer weights into ``model`` and return new version, or None."""
