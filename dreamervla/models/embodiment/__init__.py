"""Embodiment model implementations for VLA, encoders, and world models."""

from importlib import import_module
from typing import Any

_MODULES = {
    "BaseEncoder": "base_encoder",
    "ChunkAwareWorldModel": "world_model",
    "DinoTokenWorldModel": "world_model",
    "EncoderInputBatch": "protocol",
    "OpenVLAOFTPolicy": "openvla_oft_policy",
    "Pi05Policy": "pi05",
    "Pi05PrefixInputWorldModel": "world_model",
    "QwenGR00TPolicy": "qwen_groot",
    "WorldModel": "world_model",
    "build_encoder_input_batch": "protocol",
}

__all__ = list(_MODULES)


def __getattr__(name: str) -> Any:
    """Load a selected embodiment without importing unrelated model families."""
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
