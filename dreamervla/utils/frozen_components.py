"""Strict checkpoint and immutability helpers for frozen-model RL."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
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
    if component == "classifier" and state is None:
        state = state_dicts.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"checkpoint {checkpoint_path} has no non-empty {component} state")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError(f"{component} state contains non-tensor values")

    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in {"state_dicts", component, "model"}
    }
    if "config" not in metadata:
        runner_cfg = payload.get("cfg")
        if OmegaConf.is_config(runner_cfg):
            component_cfg = OmegaConf.select(runner_cfg, component, default=None)
        elif isinstance(runner_cfg, Mapping):
            component_cfg = runner_cfg.get(component)
        else:
            component_cfg = None
        component_cfg = _resolved_container(component_cfg)
        if isinstance(component_cfg, Mapping):
            metadata["config"] = {component: dict(component_cfg)}
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


def _resolved_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def require_component_config_match(
    metadata: Mapping[str, Any],
    *,
    component: str,
    active_cfg: Any,
) -> None:
    """Require checkpoint construction metadata to equal active Hydra config."""

    config = _resolved_container(metadata.get("config"))
    if not isinstance(config, Mapping) or component not in config:
        raise ValueError(
            f"{component} checkpoint must contain config.{component} metadata"
        )
    checkpoint_cfg = _resolved_container(config[component])
    resolved_active = _resolved_container(active_cfg)
    if checkpoint_cfg != resolved_active:
        raise ValueError(
            f"{component} checkpoint config does not match the active Hydra config"
        )


def resolve_classifier_threshold(
    metadata: Mapping[str, Any],
    *,
    configured: float | None = None,
) -> float:
    """Resolve the selected classifier threshold without silent drift."""

    checkpoint_value = metadata.get("threshold")
    if configured is None:
        if checkpoint_value is None:
            raise ValueError("classifier checkpoint must provide a validation threshold")
        resolved = float(checkpoint_value)
    else:
        if checkpoint_value is not None and float(configured) != float(checkpoint_value):
            raise ValueError(
                "configured classifier threshold must equal the selected checkpoint "
                f"threshold ({float(configured)} != {float(checkpoint_value)})"
            )
        resolved = float(configured)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(
            f"classifier threshold must be finite and within [0,1], got {resolved}"
        )
    return resolved


__all__ = [
    "LoadedFrozenComponent",
    "assert_module_frozen",
    "load_frozen_component",
    "module_state_sha256",
    "require_component_config_match",
    "resolve_classifier_threshold",
    "state_dict_sha256",
]
