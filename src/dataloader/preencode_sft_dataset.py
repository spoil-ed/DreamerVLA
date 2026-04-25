from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.dataloader.base_dataset import BaseDataset


@dataclass(frozen=True)
class PreencodeSFTDataSpec:
    manifest_path: str
    cache_dir: str
    num_samples: int
    num_shards: int
    hidden_dim: int
    action_dim: int
    embedding_dtype: str
    source_data_spec: dict[str, Any]


class PreencodeSFTDataset(BaseDataset):
    def __init__(
        self,
        manifest_path: str | Path = "/home/user01/liops/workspace/DreamerVLA/data/preencode/rynnvla_libero_object",
        max_open_shards: int = 2,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if self.manifest_path.is_dir():
            self.manifest_path = self.manifest_path / "manifest.pt"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Preencode manifest does not exist: {self.manifest_path}")

        self.cache_dir = self.manifest_path.parent
        self.manifest = torch.load(self.manifest_path, map_location="cpu")
        self.shards = list(self.manifest["shards"])
        self.num_samples = int(self.manifest["num_samples"])
        self.hidden_dim = int(self.manifest["hidden_dim"])
        self.embedding_dtype = str(self.manifest.get("embedding_dtype", "float32"))
        self.max_open_shards = max(1, int(max_open_shards))
        self._normalizer = self.manifest.get("normalizer", {})
        self._shard_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._global_to_shard: list[tuple[int, int]] = [(-1, -1)] * self.num_samples

        action_dim = 0
        for shard_idx, shard in enumerate(self.shards):
            start = int(shard["start_index"])
            end = int(shard["end_index"])
            shard_num = int(shard["num_samples"])
            for local_idx, global_idx in enumerate(range(start, end)):
                self._global_to_shard[global_idx] = (shard_idx, local_idx)
            if action_dim == 0 and shard_num > 0:
                payload = self._load_shard(shard_idx)
                action_dim = int(payload["action"].shape[-1])

        self._data_spec = PreencodeSFTDataSpec(
            manifest_path=str(self.manifest_path),
            cache_dir=str(self.cache_dir),
            num_samples=self.num_samples,
            num_shards=int(len(self.shards)),
            hidden_dim=self.hidden_dim,
            action_dim=int(action_dim),
            embedding_dtype=self.embedding_dtype,
            source_data_spec=dict(self.manifest.get("source_data_spec", {})),
        )

    @property
    def data_spec(self) -> PreencodeSFTDataSpec:
        return self._data_spec

    def get_normalizer(self) -> dict[str, Any]:
        return self._normalizer

    def __len__(self) -> int:
        return self.num_samples

    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        cached = self._shard_cache.get(shard_idx)
        if cached is not None:
            self._shard_cache.move_to_end(shard_idx)
            return cached

        shard_path = self.cache_dir / self.shards[shard_idx]["file"]
        payload = torch.load(shard_path, map_location="cpu")
        self._shard_cache[shard_idx] = payload
        self._shard_cache.move_to_end(shard_idx)
        while len(self._shard_cache) > self.max_open_shards:
            self._shard_cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_idx, local_idx = self._global_to_shard[index]
        if shard_idx < 0:
            raise IndexError(f"Sample index {index} is missing from the manifest mapping.")
        payload = self._load_shard(shard_idx)

        obs_embedding = payload["obs_embedding"][local_idx].float()
        next_obs_embedding = payload["next_obs_embedding"][local_idx].float()
        action = payload["action"][local_idx].float()
        action_mask = payload["action_mask"][local_idx].bool()
        reward = payload["reward"][local_idx].float()
        meta = payload["meta"][local_idx]

        return {
            "obs": {
                "task_type": meta.get("task_type"),
                "task_id": meta.get("task_id"),
            },
            "next_obs": {
                "task_type": meta.get("task_type"),
                "task_id": meta.get("task_id"),
            },
            "obs_embedding": obs_embedding,
            "next_obs_embedding": next_obs_embedding,
            "action": action,
            "action_mask": action_mask,
            "reward": reward,
            "meta": meta,
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        obs_embedding = torch.stack([item["obs_embedding"] for item in batch], dim=0)
        next_obs_embedding = torch.stack([item["next_obs_embedding"] for item in batch], dim=0)
        action = torch.stack([item["action"] for item in batch], dim=0)
        action_mask = torch.stack([item["action_mask"] for item in batch], dim=0)
        reward = torch.stack([item["reward"] for item in batch], dim=0)
        meta = [item["meta"] for item in batch]
        return {
            "obs": {
                "task_type": [item["obs"]["task_type"] for item in batch],
                "task_id": torch.tensor([int(item["obs"]["task_id"]) for item in batch], dtype=torch.long),
            },
            "next_obs": {
                "task_type": [item["next_obs"]["task_type"] for item in batch],
                "task_id": torch.tensor([int(item["next_obs"]["task_id"]) for item in batch], dtype=torch.long),
            },
            "obs_embedding": obs_embedding,
            "next_obs_embedding": next_obs_embedding,
            "action": action,
            "action_mask": action_mask,
            "reward": reward,
            "meta": meta,
        }

PreencodeRynnVLADataSpec = PreencodeSFTDataSpec
PreencodeRynnVLADataset = PreencodeSFTDataset

__all__ = [
    "PreencodeSFTDataSpec",
    "PreencodeSFTDataset",
    "PreencodeRynnVLADataSpec",
    "PreencodeRynnVLADataset",
]
