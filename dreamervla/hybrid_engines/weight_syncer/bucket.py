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
from dreamervla.hybrid_engines.weight_syncer.objectstore import ObjectStoreWeightSyncer


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
        # ActorGroup exports an independent CPU snapshot. Keep that storage and
        # let Ray serialize each bucket instead of cloning the complete model a
        # second time before bucketing.
        cpu_state = {
            name: (
                value.detach().cpu()
                if isinstance(value, torch.Tensor)
                else torch.as_tensor(value).detach().cpu()
            )
            for name, value in state_dict.items()
        }
        buckets = bucket_state_dict(cpu_state, self.bucket_bytes)
        bucket_refs = [
            self._store.set.remote(
                _bucket_key(key, index, version=int(version)),
                int(version),
                bucket,
            )
            for index, bucket in enumerate(buckets)
        ]
        ray.get(bucket_refs)
        self.commit(str(key), version=int(version), num_buckets=len(buckets))

    def push_bucket(
        self,
        key: str,
        bucket: dict[str, torch.Tensor],
        *,
        version: int,
        index: int,
    ) -> None:
        """Publish one independent CPU bucket without retaining a full state dict."""

        ray.get(
            self._store.set.remote(
                _bucket_key(str(key), int(index), version=int(version)),
                int(version),
                bucket,
            )
        )

    def commit(self, key: str, *, version: int, num_buckets: int) -> None:
        """Atomically expose a version after every bucket has been published."""

        if int(num_buckets) <= 0:
            raise ValueError("weight sync must publish at least one bucket")
        previous_item = ray.get(self._store.get.remote(_meta_key(str(key))))
        previous: tuple[int, int] | None = None
        if previous_item is not None:
            previous_version, previous_meta = previous_item
            previous_meta = _resolve(previous_meta)
            previous = (
                int(previous_version),
                int(previous_meta["num_buckets"].item()),
            )
        meta = {"num_buckets": torch.tensor(int(num_buckets), dtype=torch.int64)}
        ray.get(self._store.set.remote(_meta_key(str(key)), int(version), meta))
        if previous is not None and int(previous[0]) < int(version):
            old_version, old_bucket_count = previous
            ray.get(
                self._store.delete.remote(
                    [
                        _bucket_key(
                            str(key),
                            index,
                            version=int(old_version),
                        )
                        for index in range(int(old_bucket_count))
                    ]
                )
            )

    def pull(self, key: str, model: torch.nn.Module, local_version: int) -> int | None:
        item = ray.get(self._store.get.remote(_meta_key(key)))
        if item is None:
            return None
        version, meta = item
        if int(version) <= int(local_version):
            return None
        meta = _resolve(meta)
        num_buckets = int(meta["num_buckets"].item())
        bucket_items = ray.get(
            [
                self._store.get.remote(_bucket_key(key, index, version=int(version)))
                for index in range(num_buckets)
            ]
        )
        destination = model.state_dict()
        for index, bucket_item in enumerate(bucket_items):
            if bucket_item is None:
                raise RuntimeError(
                    f"missing bucket {index} for key {key!r}; weight store is inconsistent"
                )
            bucket_version, bucket = bucket_item
            if int(bucket_version) != int(version):
                raise RuntimeError(
                    f"bucket {index} for key {key!r} has version "
                    f"{bucket_version}, expected {version}"
                )
            for name, value in _resolve(bucket).items():
                if name not in destination:
                    raise KeyError(f"weight bucket contains unknown parameter {name!r}")
                destination[name].copy_(
                    value.to(
                        device=destination[name].device,
                        dtype=destination[name].dtype,
                    )
                )
        return int(version)

    def release(self, key: str, *, version: int) -> bool:
        """Release a committed snapshot after every Rollout worker applied it."""

        item = ray.get(self._store.get.remote(_meta_key(str(key))))
        if item is None or int(item[0]) != int(version):
            return False
        meta = _resolve(item[1])
        num_buckets = int(meta["num_buckets"].item())
        ray.get(
            self._store.delete.remote(
                [
                    _meta_key(str(key)),
                    *(
                        _bucket_key(str(key), index, version=int(version))
                        for index in range(num_buckets)
                    ),
                ]
            )
        )
        return True


def _bucket_key(key: str, index: int, *, version: int) -> str:
    return f"{key}::v{int(version)}::b{index}"


def _meta_key(key: str) -> str:
    return f"{key}::meta"


def _resolve(value: Any) -> Any:
    if isinstance(value, ray.ObjectRef):
        return ray.get(value)
    return value
