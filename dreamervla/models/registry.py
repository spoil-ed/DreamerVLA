"""Lightweight model registry for config-facing model selection.

Existing Hydra ``_target_`` configs remain supported by the runners. This
registry gives new configs a stable ``model_type`` indirection so model module
paths can move without editing every recipe.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig

ModelBuilder = Callable[[DictConfig, Any], Any]

_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(model_type: str, builder: ModelBuilder, *, replace: bool = False) -> None:
    """Register a model builder under a config-facing type name."""

    key = _normalize_model_type(model_type)
    if not replace and key in _MODEL_REGISTRY:
        raise ValueError(f"model_type {key!r} is already registered")
    _MODEL_REGISTRY[key] = builder


def get_model(cfg: DictConfig, torch_dtype: Any = None) -> Any:
    """Build a model from ``cfg.model_type`` using the registered builder."""

    model_type = _normalize_model_type(cfg.get("model_type", ""))
    if not model_type:
        raise ValueError("model config must include model_type")
    try:
        builder = _MODEL_REGISTRY[model_type]
    except KeyError as exc:
        raise ValueError(
            f"unknown model_type {model_type!r}; supported: {registered_model_types()}"
        ) from exc
    return builder(cfg, torch_dtype)


def registered_model_types() -> list[str]:
    """Return supported model types in deterministic order."""

    return sorted(_MODEL_REGISTRY)


def validate_model_type(model_type: str) -> None:
    """Fail fast if a config references an unknown registered model type."""

    key = _normalize_model_type(model_type)
    if key and key not in _MODEL_REGISTRY:
        raise ValueError(
            f"unknown model_type {key!r}; supported: {registered_model_types()}"
        )


def _normalize_model_type(model_type: str) -> str:
    return str(model_type).strip().lower().replace("-", "_")


def _lazy_get_model(module_name: str) -> ModelBuilder:
    def _builder(cfg: DictConfig, torch_dtype: Any = None) -> Any:
        module = importlib.import_module(module_name)
        return module.get_model(cfg, torch_dtype=torch_dtype)

    return _builder


register_model("openvla", _lazy_get_model("dreamervla.models.embodiment.openvla"))
register_model("openvla_oft", _lazy_get_model("dreamervla.models.embodiment.openvla_oft"))


__all__ = [
    "get_model",
    "register_model",
    "registered_model_types",
    "validate_model_type",
]
