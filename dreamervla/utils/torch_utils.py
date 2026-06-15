from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def resolve_device(device: str) -> torch.device:
    # Auto device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def freeze_module(module: nn.Module) -> None:
    # Eval mode
    module.eval()
    # No grads
    for parameter in module.parameters():
        parameter.requires_grad = False


def move_mapping_to_device(
    values: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    # Tensor move
    moved = {}
    for key, value in values.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def repeat_tensor_mapping(values: Mapping[str, Any], repeats: int) -> dict[str, Any]:
    # Tensor repeat
    repeated = {}
    for key, value in values.items():
        repeated[key] = _repeat_batch_like(value, repeats)
    return repeated


def _repeat_batch_like(value: Any, repeats: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    if isinstance(value, Mapping):
        return {key: _repeat_batch_like(item, repeats) for key, item in value.items()}
    if isinstance(value, list):
        expanded: list[Any] = []
        for item in value:
            expanded.extend([item] * repeats)
        return expanded
    if isinstance(value, tuple):
        expanded = []
        for item in value:
            expanded.extend([item] * repeats)
        return tuple(expanded)
    return value
