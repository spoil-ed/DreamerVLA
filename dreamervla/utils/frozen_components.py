"""Strict checkpoint and immutability helpers for frozen-model RL."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LoadedFrozenComponent:
    """One component state and the non-state metadata from its checkpoint."""

    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def load_frozen_component(
    path: str | Path,
    component: str,
) -> LoadedFrozenComponent:
    """Load a WM or classifier state from the supported training schemas."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{component} checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"{component} checkpoint payload must be a mapping, got {type(payload).__name__}"
        )

    state_dicts = payload.get("state_dicts", {})
    if not isinstance(state_dicts, Mapping):
        state_dicts = {}
    state = payload.get(component, state_dicts.get(component))
    if component == "classifier" and state is None:
        state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"checkpoint {checkpoint_path} has no non-empty {component} state")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError(f"{component} state contains non-tensor values")

    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in {"state_dicts", component, "model"}
    }
    metadata["checkpoint_path"] = str(checkpoint_path)
    return LoadedFrozenComponent(
        state_dict={str(key): value for key, value in state.items()},
        metadata=metadata,
    )


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash names, dtypes, shapes, and bytes for a complete tensor state."""

    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        for item in (name, str(value.dtype), repr(tuple(value.shape))):
            encoded = item.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="big"))
            digest.update(encoded)
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, byteorder="big"))
        digest.update(raw)
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    """Hash names, dtypes, shapes, and bytes for a complete module state."""

    return state_dict_sha256(module.state_dict())


def assert_module_frozen(module: nn.Module, *, name: str) -> None:
    """Fail when a supposedly frozen module can train or is in train mode."""

    if module.training:
        raise RuntimeError(f"{name} must remain in eval mode")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError(f"{name} exposes trainable parameters")


__all__ = [
    "LoadedFrozenComponent",
    "assert_module_frozen",
    "load_frozen_component",
    "module_state_sha256",
    "state_dict_sha256",
]
