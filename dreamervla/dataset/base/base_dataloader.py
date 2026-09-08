"""Common dataset interfaces and batch metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class DatasetLoaderBundle:
    """Dataset, PyTorch loader and normalization metadata returned by a factory."""

    dataset: Any
    dataloader: DataLoader
    dataset_statistics: dict[str, Any]


class BaseDataset(Dataset[dict[str, Any]], ABC):
    """Common dataset contract for DreamerVLA training inputs."""

    @property
    @abstractmethod
    def data_spec(self) -> Any:
        """Structured metadata describing the dataset."""

    @abstractmethod
    def get_normalizer(self) -> Any:
        """Return dataset-side normalization metadata used by the workspace."""

    @staticmethod
    def resolve_project_path(path: str | Path, base_dir: Path | None = None) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path.resolve()
        if base_dir is not None:
            candidate = (base_dir / path).resolve()
            if candidate.exists():
                return candidate
        return (PROJECT_ROOT / path).resolve()

    @staticmethod
    def pad_action_batch(
        actions: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_steps = max((int(action.shape[0]) for action in actions), default=0)
        action_dim = max(
            (int(action.shape[-1]) for action in actions if action.ndim == 2), default=0
        )
        padded = torch.zeros(len(actions), max_steps, action_dim, dtype=torch.float32)
        mask = torch.zeros(len(actions), max_steps, dtype=torch.bool)
        for idx, action in enumerate(actions):
            if action.numel() == 0:
                continue
            steps = int(action.shape[0])
            padded[idx, :steps] = action
            mask[idx, :steps] = True
        return padded, mask

    @staticmethod
    def pad_state_batch(
        states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_dim = max((int(state.numel()) for state in states), default=0)
        padded = torch.zeros(len(states), max_dim, dtype=torch.float32)
        mask = torch.zeros(len(states), max_dim, dtype=torch.bool)
        for idx, state in enumerate(states):
            if state.numel() == 0:
                continue
            dim = int(state.numel())
            padded[idx, :dim] = state.reshape(-1)
            mask[idx, :dim] = True
        return padded, mask

    @staticmethod
    def stack_long(values: list[int]) -> torch.Tensor:
        if not values:
            return torch.zeros(0, dtype=torch.long)
        return torch.tensor(values, dtype=torch.long)


__all__ = ["BaseDataset", "DatasetLoaderBundle"]
