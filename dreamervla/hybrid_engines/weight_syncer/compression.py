"""Tensor compression helpers for weight synchronization transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ray
import torch

from dreamervla.hybrid_engines.weight_syncer.base import WeightSyncer
from dreamervla.hybrid_engines.weight_syncer.objectstore import (
    ObjectStoreWeightSyncer,
    _to_cpu_tensor,
)


@dataclass(frozen=True)
class PackedTensor:
    """Compressed tensor payload with enough metadata to restore dtype."""

    tensor: torch.Tensor
    original_dtype: torch.dtype


class DTypeTensorCompressor:
    """Cast floating tensors to a transport dtype and restore on receive."""

    def __init__(self, transport_dtype: str | torch.dtype = "fp16") -> None:
        self.transport_dtype = _dtype_from_transport(transport_dtype)

    def compress_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> dict[str, PackedTensor]:
        packed: dict[str, PackedTensor] = {}
        for key, value in state_dict.items():
            tensor = _to_cpu_tensor(value)
            original_dtype = tensor.dtype
            payload = tensor
            if torch.is_floating_point(tensor):
                payload = tensor.to(self.transport_dtype)
            packed[key] = PackedTensor(
                tensor=payload.detach().cpu().clone(),
                original_dtype=original_dtype,
            )
        return packed

    def decompress_state_dict(
        self,
        packed: dict[str, PackedTensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: item.tensor.to(dtype=item.original_dtype)
            for key, item in packed.items()
        }


class CompressedWeightSyncer(WeightSyncer):
    """Object-store syncer that stores tensors in an explicit transport dtype."""

    def __init__(
        self,
        store_name: str = "DreamerVLACompressedWeightStore",
        *,
        transport_dtype: str | torch.dtype = "fp16",
    ) -> None:
        self.store_name = str(store_name)
        self.compressor = DTypeTensorCompressor(transport_dtype=transport_dtype)
        self._store = ObjectStoreWeightSyncer._get_or_create_store(self.store_name)

    def push(self, key: str, state_dict: dict[str, Any], version: int) -> None:
        packed = self.compressor.compress_state_dict(state_dict)
        ray.get(self._store.set.remote(str(key), int(version), packed))

    def pull(self, key: str, model: torch.nn.Module, local_version: int) -> int | None:
        item = ray.get(self._store.get.remote(str(key)))
        if item is None:
            return None
        version, packed = item
        if int(version) <= int(local_version):
            return None
        if isinstance(packed, ray.ObjectRef):
            packed = ray.get(packed)
        state = self.compressor.decompress_state_dict(packed)
        device = next(model.parameters(), torch.empty(0)).device
        model.load_state_dict({name: value.to(device) for name, value in state.items()})
        return int(version)


def _dtype_from_transport(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unsupported transport dtype: {value!r}")


__all__ = ["CompressedWeightSyncer", "DTypeTensorCompressor", "PackedTensor"]
