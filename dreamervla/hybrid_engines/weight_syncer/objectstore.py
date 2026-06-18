"""Ray object-store based weight synchronization."""

from __future__ import annotations

from typing import Any

import ray
import torch

from dreamervla.hybrid_engines.weight_syncer.base import WeightSyncer


@ray.remote
class _WeightStore:
    def __init__(self) -> None:
        self.items: dict[str, tuple[int, Any]] = {}

    def set(self, key: str, version: int, state_dict: dict[str, torch.Tensor]) -> None:
        current = self.items.get(str(key))
        if current is None or int(version) >= int(current[0]):
            self.items[str(key)] = (int(version), ray.put(state_dict))

    def get(self, key: str) -> tuple[int, Any] | None:
        return self.items.get(str(key))


class ObjectStoreWeightSyncer(WeightSyncer):
    """Store CPU state_dicts in Ray's object store with monotonic versions."""

    def __init__(self, store_name: str = "DreamerVLAWeightStore") -> None:
        self.store_name = str(store_name)
        self.store = self._get_or_create_store(self.store_name)

    def push(self, key: str, state_dict: dict[str, Any], version: int) -> None:
        cpu_state = {
            name: _to_cpu_tensor(value)
            for name, value in state_dict.items()
        }
        ray.get(self.store.set.remote(str(key), int(version), cpu_state))

    def pull(self, key: str, model: torch.nn.Module, local_version: int) -> int | None:
        item = ray.get(self.store.get.remote(str(key)))
        if item is None:
            return None
        version, state = item
        if int(version) <= int(local_version):
            return None
        if isinstance(state, ray.ObjectRef):
            state = ray.get(state)
        device = next(model.parameters(), torch.empty(0)).device
        model.load_state_dict({name: value.to(device) for name, value in state.items()})
        return int(version)

    @staticmethod
    def _get_or_create_store(name: str) -> Any:
        try:
            return ray.get_actor(name, namespace="DreamerVLA")
        except ValueError:
            actor = _WeightStore.options(
                name=name,
                namespace="DreamerVLA",
                lifetime="detached",
            ).remote()
            ray.get(actor.get.remote("__ready__"))
            return actor


def _to_cpu_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return torch.as_tensor(value).detach().cpu().clone()
