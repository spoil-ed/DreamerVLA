"""Patch-based weight synchronization helpers."""

from __future__ import annotations

from typing import Any

import ray
import torch

from dreamervla.hybrid_engines.weight_syncer.base import WeightSyncer
from dreamervla.hybrid_engines.weight_syncer.objectstore import (
    ObjectStoreWeightSyncer,
    _to_cpu_tensor,
)


def state_dict_delta(
    previous: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return tensors that are new or changed relative to ``previous``."""

    delta: dict[str, torch.Tensor] = {}
    for name, tensor in current.items():
        old = previous.get(name)
        if old is None or old.dtype != tensor.dtype or old.shape != tensor.shape:
            delta[name] = tensor
            continue
        if not torch.equal(old, tensor):
            delta[name] = tensor
    return delta


class PatchWeightSyncer(WeightSyncer):
    """Object-store syncer with single-step patch updates and full fallback.

    If the receiver is exactly one version behind, ``pull`` applies only the
    latest tensor delta to its current ``state_dict``. Older receivers fall back
    to the full snapshot stored alongside the patch metadata.
    """

    def __init__(self, store_name: str = "DreamerVLAPatchWeightStore") -> None:
        self.store_name = str(store_name)
        self._store = ObjectStoreWeightSyncer._get_or_create_store(self.store_name)

    def push(self, key: str, state_dict: dict[str, Any], version: int) -> None:
        version = int(version)
        full_key = _full_key(key)
        current = ray.get(self._store.get.remote(full_key))
        if current is not None and version < int(current[0]):
            return

        cpu_state = {name: _to_cpu_tensor(value) for name, value in state_dict.items()}
        previous = _resolve(current[1]) if current is not None else {}
        patch = state_dict_delta(previous, cpu_state)
        ray.get(self._store.set.remote(_patch_key(key, version), version, patch))
        ray.get(self._store.set.remote(full_key, version, cpu_state))
        ray.get(
            self._store.set.remote(
                _meta_key(key),
                version,
                {
                    "latest_version": torch.tensor(version, dtype=torch.int64),
                    "patch_keys": list(patch),
                },
            )
        )

    def pull(self, key: str, model: torch.nn.Module, local_version: int) -> int | None:
        item = ray.get(self._store.get.remote(_meta_key(key)))
        if item is None:
            return None
        version, meta = item
        version = int(version)
        if version <= int(local_version):
            return None

        device = next(model.parameters(), torch.empty(0)).device
        if int(local_version) == version - 1:
            patch_item = ray.get(self._store.get.remote(_patch_key(key, version)))
            if patch_item is not None:
                patch = _resolve(patch_item[1])
                state = model.state_dict()
                for name, value in patch.items():
                    state[name] = value.to(device)
                model.load_state_dict(state)
                return version

        full_item = ray.get(self._store.get.remote(_full_key(key)))
        if full_item is None:
            raise RuntimeError(f"missing full snapshot for key {key!r}")
        full_state = _resolve(full_item[1])
        model.load_state_dict({name: value.to(device) for name, value in full_state.items()})
        return version


def _full_key(key: str) -> str:
    return f"{key}::full"


def _meta_key(key: str) -> str:
    return f"{key}::meta"


def _patch_key(key: str, version: int) -> str:
    return f"{key}::patch::{int(version)}"


def _resolve(value: Any) -> Any:
    if isinstance(value, ray.ObjectRef):
        return ray.get(value)
    return value


__all__ = ["PatchWeightSyncer", "state_dict_delta"]
