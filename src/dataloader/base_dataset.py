from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class BaseDataset(Dataset[dict[str, Any]], ABC):
    """Common dataset contract for Dreamer-VLA training inputs."""

    @property
    @abstractmethod
    def data_spec(self) -> Any:
        """Structured metadata describing the dataset."""

    @abstractmethod
    def get_normalizer(self) -> Any:
        """Return dataset-side normalization metadata used by the workspace."""

    @property
    def spec(self) -> Any:
        """Backward-compatible alias for callers that still expect `spec`."""
        return self.data_spec
