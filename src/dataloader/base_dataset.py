from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    def discover_task_files(raw_data_dir: Path, task_suite_name: str) -> list[Path]:
        direct_files = sorted(raw_data_dir.glob("*_demo.hdf5"))
        if direct_files:
            return direct_files

        suite_dir = raw_data_dir / task_suite_name
        if suite_dir.is_dir():
            suite_files = sorted(suite_dir.glob("*_demo.hdf5"))
            if suite_files:
                return suite_files

        nested_files = sorted(raw_data_dir.glob("libero_*/*_demo.hdf5"))
        if nested_files:
            if task_suite_name == "all":
                return nested_files
            filtered = [path for path in nested_files if path.parent.name == task_suite_name]
            if filtered:
                return filtered

        return []

    @staticmethod
    def extract_demo_index(demo_key: str) -> int:
        match = re.fullmatch(r"demo_(\d+)", demo_key)
        if match is None:
            return -1
        return int(match.group(1))

    @classmethod
    def list_demo_keys(cls, data_group: h5py.Group) -> list[str]:
        demo_keys = [key for key in data_group.keys() if cls.extract_demo_index(key) >= 0]
        return sorted(demo_keys, key=cls.extract_demo_index)

    @staticmethod
    def image_from_array(rgb_frame: np.ndarray) -> Image.Image:
        return Image.fromarray(rgb_frame[::-1, ::-1].astype(np.uint8))

    @staticmethod
    def pad_action_batch(actions: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        max_steps = max((int(action.shape[0]) for action in actions), default=0)
        action_dim = max((int(action.shape[-1]) for action in actions if action.ndim == 2), default=0)
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
    def pad_state_batch(states: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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
