from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.dataloader.base_dataset import BaseDataset


@dataclass(frozen=True)
class PretokenizeDataSpec:
    config_path: str
    manifest_path: str
    num_samples: int
    max_token_length: int
    prompt_text: str | None = None


class PretokenizeDataset(BaseDataset):
    """Loads pretokenized VLA SFT samples produced by the preprocess scripts."""

    def __init__(self, config_path: str | Path) -> None:
        super().__init__()
        self.config_path = self.resolve_project_path(config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle) if self.config_path.suffix == ".json" else None
        if config is None:
            import yaml

            with self.config_path.open("r", encoding="utf-8") as handle:
                config = yaml.load(handle, Loader=yaml.FullLoader)

        meta_entries = config.get("META")
        if not isinstance(meta_entries, list) or not meta_entries:
            raise ValueError(f"PretokenizeDataset expects META to be a non-empty list in {self.config_path}")
        manifest_value = meta_entries[0].get("path")
        if manifest_value is None:
            raise ValueError(f"PretokenizeDataset META[0] is missing 'path' in {self.config_path}")

        self.manifest_path = self.resolve_project_path(manifest_value, base_dir=self.config_path.parent)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.records = json.load(handle)
        if not isinstance(self.records, list):
            raise ValueError(f"Pretokenize manifest must be a list: {self.manifest_path}")

        self.max_token_length = max((int(item.get("len", 0)) for item in self.records), default=0)
        self._data_spec = PretokenizeDataSpec(
            config_path=str(self.config_path),
            manifest_path=str(self.manifest_path),
            num_samples=len(self.records),
            max_token_length=self.max_token_length,
            prompt_text=config.get("prompt_text"),
        )

    @property
    def data_spec(self) -> PretokenizeDataSpec:
        return self._data_spec

    def get_normalizer(self) -> dict[str, dict[str, torch.Tensor]]:
        return {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        file_path = self.resolve_project_path(record["file"], base_dir=self.manifest_path.parent)
        with file_path.open("rb") as handle:
            payload = pickle.load(handle)

        input_ids = list(payload["token"])
        labels = list(payload["label"])
        meta = dict(record.get("meta", {}))
        if isinstance(payload, dict) and "meta" in payload and isinstance(payload["meta"], dict):
            meta.update(payload["meta"])

        image = list(payload.get("image", [])) if isinstance(payload, dict) else []
        action = list(payload.get("action", [])) if isinstance(payload, dict) else []
        state = list(payload.get("state", [])) if isinstance(payload, dict) else []
        next_obs = dict(payload.get("next_obs", {})) if isinstance(payload, dict) and isinstance(payload.get("next_obs"), dict) else {}
        reward_value = payload.get("reward") if isinstance(payload, dict) else None
        if reward_value is None:
            reward_value = meta.get("reward", 0.0)
        task_name = payload.get("task_name") if isinstance(payload, dict) else None
        if task_name is None:
            task_name = meta.get("task_name", "")
        wm_obs_input_ids = payload.get("wm_obs_input_ids") if isinstance(payload, dict) else None
        if not isinstance(wm_obs_input_ids, list):
            wm_obs_input_ids = list(input_ids)
        wm_next_obs_input_ids = payload.get("wm_next_obs_input_ids") if isinstance(payload, dict) else None
        if not isinstance(wm_next_obs_input_ids, list):
            wm_next_obs_input_ids = list(wm_obs_input_ids)

        wm_action = self._load_action_sequence(action)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "length": int(len(input_ids)),
            "image": image,
            "action": action,
            "state": state,
            "next_obs": next_obs,
            "reward": float(reward_value) if reward_value is not None else 0.0,
            "task_name": str(task_name),
            "wm_obs_input_ids": [int(x) for x in wm_obs_input_ids],
            "wm_next_obs_input_ids": [int(x) for x in wm_next_obs_input_ids],
            "wm_action": wm_action,
            "meta": meta,
            "file": str(file_path),
            "id": int(payload.get("id", record.get("id", index))),
        }

    @staticmethod
    def _load_action_sequence(action: list[Any]) -> torch.Tensor:
        values: list[np.ndarray] = []
        for entry in action:
            if isinstance(entry, str):
                path = Path(entry).expanduser()
                if path.is_file():
                    values.append(np.asarray(np.load(path), dtype=np.float32))
                continue
            values.append(np.asarray(entry, dtype=np.float32))
        if not values:
            return torch.zeros((0, 0), dtype=torch.float32)
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        return torch.tensor(array, dtype=torch.float32)

    @staticmethod
    def _pad_action_batch(actions: list[torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not actions:
            return None, None
        max_steps = max(int(tensor.shape[0]) for tensor in actions)
        max_dim = max(int(tensor.shape[1]) if tensor.ndim == 2 else 0 for tensor in actions)
        if max_steps <= 0 or max_dim <= 0:
            return None, None
        padded = torch.zeros(len(actions), max_steps, max_dim, dtype=torch.float32)
        mask = torch.zeros(len(actions), max_steps, dtype=torch.bool)
        for idx, tensor in enumerate(actions):
            if tensor.ndim != 2 or tensor.numel() == 0:
                continue
            steps = int(tensor.shape[0])
            dim = int(tensor.shape[1])
            padded[idx, :steps, :dim] = tensor
            mask[idx, :steps] = True
        return padded, mask

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        padded_action, action_mask = PretokenizeDataset._pad_action_batch([item["wm_action"] for item in batch])
        return {
            "input_ids": [list(item["input_ids"]) for item in batch],
            "labels": [list(item["labels"]) for item in batch],
            "lengths": torch.tensor([int(item["length"]) for item in batch], dtype=torch.long),
            "image": [item["image"] for item in batch],
            "action": padded_action,
            "action_mask": action_mask,
            "state": [item["state"] for item in batch],
            "next_obs": [item["next_obs"] for item in batch],
            "reward": torch.tensor([float(item["reward"]) for item in batch], dtype=torch.float32),
            "task_name": [str(item["task_name"]) for item in batch],
            "wm_obs_input_ids": [list(item["wm_obs_input_ids"]) for item in batch],
            "wm_next_obs_input_ids": [list(item["wm_next_obs_input_ids"]) for item in batch],
            "meta": [item["meta"] for item in batch],
            "file": [item["file"] for item in batch],
            "id": torch.tensor([int(item["id"]) for item in batch], dtype=torch.long),
        }


__all__ = ["PretokenizeDataSpec", "PretokenizeDataset"]
