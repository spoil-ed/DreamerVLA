"""Patch-based weight synchronization helpers."""

from __future__ import annotations

import time
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
        self.last_push_metrics: dict[str, float] = {}
        self.last_pull_metrics: dict[str, float] = {}

    def push(self, key: str, state_dict: dict[str, Any], version: int) -> None:
        version = int(version)
        full_key = _full_key(key)
        get_start = time.perf_counter()
        current = ray.get(self._store.get.remote(full_key))
        full_get_s = float(time.perf_counter() - get_start)
        if current is not None and version < int(current[0]):
            self.last_push_metrics = {
                "sync/patch_push_skipped": 1.0,
                "sync/patch_full_get_s": full_get_s,
            }
            return

        cpu_start = time.perf_counter()
        cpu_state = {name: _to_cpu_tensor(value) for name, value in state_dict.items()}
        cpu_state_s = float(time.perf_counter() - cpu_start)
        prev_start = time.perf_counter()
        previous = _resolve(current[1]) if current is not None else {}
        previous_resolve_s = float(time.perf_counter() - prev_start)
        delta_start = time.perf_counter()
        patch = state_dict_delta(previous, cpu_state)
        delta_s = float(time.perf_counter() - delta_start)
        patch_set_start = time.perf_counter()
        ray.get(self._store.set.remote(_patch_key(key, version), version, patch))
        patch_set_s = float(time.perf_counter() - patch_set_start)
        full_set_start = time.perf_counter()
        ray.get(self._store.set.remote(full_key, version, cpu_state))
        full_set_s = float(time.perf_counter() - full_set_start)
        meta_set_start = time.perf_counter()
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
        meta_set_s = float(time.perf_counter() - meta_set_start)
        self.last_push_metrics = {
            "sync/patch_push_skipped": 0.0,
            "sync/patch_full_get_s": full_get_s,
            "sync/patch_cpu_state_s": cpu_state_s,
            "sync/patch_previous_resolve_s": previous_resolve_s,
            "sync/patch_delta_s": delta_s,
            "sync/patch_patch_set_s": patch_set_s,
            "sync/patch_full_set_s": full_set_s,
            "sync/patch_meta_set_s": meta_set_s,
            "sync/patch_tensors": float(len(patch)),
            "sync/patch_bytes": float(_state_nbytes(patch)),
            "sync/patch_full_tensors": float(len(cpu_state)),
            "sync/patch_full_bytes": float(_state_nbytes(cpu_state)),
        }

    def pull(self, key: str, model: torch.nn.Module, local_version: int) -> int | None:
        meta_start = time.perf_counter()
        item = ray.get(self._store.get.remote(_meta_key(key)))
        meta_get_s = float(time.perf_counter() - meta_start)
        if item is None:
            self.last_pull_metrics = {
                "sync/patch_pull_meta_get_s": meta_get_s,
                "sync/patch_pull_updated": 0.0,
            }
            return None
        version, meta = item
        version = int(version)
        if version <= int(local_version):
            self.last_pull_metrics = {
                "sync/patch_pull_meta_get_s": meta_get_s,
                "sync/patch_pull_updated": 0.0,
                "sync/patch_pull_version": float(version),
            }
            return None

        device = next(model.parameters(), torch.empty(0)).device
        if int(local_version) == version - 1:
            patch_get_start = time.perf_counter()
            patch_item = ray.get(self._store.get.remote(_patch_key(key, version)))
            patch_get_s = float(time.perf_counter() - patch_get_start)
            if patch_item is not None:
                patch_resolve_start = time.perf_counter()
                patch = _resolve(patch_item[1])
                patch_resolve_s = float(time.perf_counter() - patch_resolve_start)
                patch_apply_start = time.perf_counter()
                state = model.state_dict()
                for name, value in patch.items():
                    state[name] = value.to(device)
                model.load_state_dict(state)
                patch_apply_s = float(time.perf_counter() - patch_apply_start)
                self.last_pull_metrics = {
                    "sync/patch_pull_meta_get_s": meta_get_s,
                    "sync/patch_pull_patch_get_s": patch_get_s,
                    "sync/patch_pull_patch_resolve_s": patch_resolve_s,
                    "sync/patch_pull_patch_apply_s": patch_apply_s,
                    "sync/patch_pull_used_full": 0.0,
                    "sync/patch_pull_updated": 1.0,
                    "sync/patch_pull_version": float(version),
                    "sync/patch_pull_tensors": float(len(patch)),
                    "sync/patch_pull_bytes": float(_state_nbytes(patch)),
                }
                return version

        full_get_start = time.perf_counter()
        full_item = ray.get(self._store.get.remote(_full_key(key)))
        full_get_s = float(time.perf_counter() - full_get_start)
        if full_item is None:
            raise RuntimeError(f"missing full snapshot for key {key!r}")
        full_resolve_start = time.perf_counter()
        full_state = _resolve(full_item[1])
        full_resolve_s = float(time.perf_counter() - full_resolve_start)
        full_load_start = time.perf_counter()
        model.load_state_dict({name: value.to(device) for name, value in full_state.items()})
        full_load_s = float(time.perf_counter() - full_load_start)
        self.last_pull_metrics = {
            "sync/patch_pull_meta_get_s": meta_get_s,
            "sync/patch_pull_full_get_s": full_get_s,
            "sync/patch_pull_full_resolve_s": full_resolve_s,
            "sync/patch_pull_full_load_s": full_load_s,
            "sync/patch_pull_used_full": 1.0,
            "sync/patch_pull_updated": 1.0,
            "sync/patch_pull_version": float(version),
            "sync/patch_pull_tensors": float(len(full_state)),
            "sync/patch_pull_bytes": float(_state_nbytes(full_state)),
        }
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


def _state_nbytes(state_dict: dict[str, Any]) -> int:
    total = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel() * value.element_size())
        else:
            tensor = torch.as_tensor(value)
            total += int(tensor.numel() * tensor.element_size())
    return total


__all__ = ["PatchWeightSyncer", "state_dict_delta"]
