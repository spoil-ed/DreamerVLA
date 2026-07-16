"""Checkpoint loading and hashing for trainable cotrain components."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


@dataclass(frozen=True)
class LoadedComponentCheckpoint:
    """One component state and the non-state metadata from its checkpoint."""

    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def load_component_checkpoint(
    path: str | Path,
    component: str,
) -> LoadedComponentCheckpoint:
    """Load a WM or classifier from a Torch file or HF-style directory."""

    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.is_dir():
        from dreamervla.utils.hf_module import load_module_pretrained

        module = load_module_pretrained(str(checkpoint_path), map_location="cpu")
        return LoadedComponentCheckpoint(
            state_dict={
                str(key): value.detach().cpu() for key, value in module.state_dict().items()
            },
            metadata={
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_format": "huggingface",
            },
        )
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
    metadata["checkpoint_format"] = "torch"
    return LoadedComponentCheckpoint(
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


def _resolved_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


__all__ = [
    "LoadedComponentCheckpoint",
    "load_component_checkpoint",
    "state_dict_sha256",
]
