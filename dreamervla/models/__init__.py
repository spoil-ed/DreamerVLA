"""Public model exports.

Prefer canonical subpackage imports for new code, for example
``dreamervla.models.embodiment`` or ``dreamervla.models.embodiment.world_model``.
"""

from importlib import import_module
from typing import Any

_MODULES = {
    "WorldModel": "embodiment.world_model",
    "get_model": "registry",
    "register_model": "registry",
    "registered_model_types": "registry",
    "validate_model_type": "registry",
}

__all__ = list(_MODULES)


def __getattr__(name: str) -> Any:
    """Load only the selected model or registry export."""
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
