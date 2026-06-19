"""Bucketed weight synchronization.

Partitions a ``state_dict`` into size-bounded buckets so each transfer unit is
capped, mirroring RLinf's ``BucketWeightSyncer`` shape. Single-node verifiable:
the bucketing math is a pure function and the syncer is backed by the existing
object store (no NCCL required to test).
"""

from __future__ import annotations

from typing import Any

import ray
import torch

from dreamervla.hybrid_engines.weight_syncer.base import WeightSyncer
from dreamervla.hybrid_engines.weight_syncer.objectstore import (
    ObjectStoreWeightSyncer,
    _to_cpu_tensor,
)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def bucket_state_dict(
    state_dict: dict[str, torch.Tensor],
    bucket_bytes: int,
) -> list[dict[str, torch.Tensor]]:
    """Split ``state_dict`` into ordered buckets bounded by ``bucket_bytes``.

    Keys are preserved exactly once in their original order. A bucket holds as
    many consecutive tensors as fit under ``bucket_bytes``; a single tensor that
    exceeds the budget gets its own bucket (never split).
    """

    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be positive; got {bucket_bytes}")

    buckets: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for key, tensor in state_dict.items():
        size = _tensor_bytes(tensor)
        if current and current_bytes + size > bucket_bytes:
            buckets.append(current)
            current = {}
            current_bytes = 0
        current[key] = tensor
        current_bytes += size
    if current:
        buckets.append(current)
    return buckets


class BucketWeightSyncer(WeightSyncer):
    """Object-store weight sync that splits each push into size-bounded buckets.

    Mirrors RLinf's ``BucketWeightSyncer`` shape (bounded transfer units) while
    staying object-store-backed so it is verifiable on a single node without
    NCCL. Reuses the same detached ``_WeightStore`` actor as
    :class:`ObjectStoreWeightSyncer`; every bucket and the per-key metadata
    share one monotonic version.
    """

    def __init__(
        self,
        store_name: str = "DreamerVLABucketWeightStore",
        *,
        bucket_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        self.store_name = str(store_name)
        self.bucket_bytes = int(bucket_bytes)
        self._store = ObjectStoreWeightSyncer._get_or_create_store(self.store_name)

    def push(self, key: str, state_dict: dict[str, Any], version: int) -> None:
        cpu_state = {name: _to_cpu_tensor(value) for name, value in state_dict.items()}
        buckets = bucket_state_dict(cpu_state, self.bucket_bytes)
        for index, bucket in enumerate(buckets):
            ray.get(self._store.set.remote(_bucket_key(key, index), int(version), bucket))
        meta = {"num_buckets": torch.tensor(len(buckets), dtype=torch.int64)}
        ray.get(self._store.set.remote(_meta_key(key), int(version), meta))

    def pull(self, key: str, model: torch.nn.Module, local_version: int) -> int | None:
        item = ray.get(self._store.get.remote(_meta_key(key)))
        if item is None:
            return None
        version, meta = item
        if int(version) <= int(local_version):
            return None
        meta = _resolve(meta)
        num_buckets = int(meta["num_buckets"].item())
        merged: dict[str, torch.Tensor] = {}
        for index in range(num_buckets):
            bucket_item = ray.get(self._store.get.remote(_bucket_key(key, index)))
            if bucket_item is None:
                raise RuntimeError(
                    f"missing bucket {index} for key {key!r}; weight store is inconsistent"
                )
            merged.update(_resolve(bucket_item[1]))
        device = next(model.parameters(), torch.empty(0)).device
        model.load_state_dict({name: value.to(device) for name, value in merged.items()})
        return int(version)


def _bucket_key(key: str, index: int) -> str:
    return f"{key}::b{index}"


def _meta_key(key: str) -> str:
    return f"{key}::meta"


def _resolve(value: Any) -> Any:
    if isinstance(value, ray.ObjectRef):
        return ray.get(value)
    return value
