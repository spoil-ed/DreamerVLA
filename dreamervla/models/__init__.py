"""Public model exports.

Prefer canonical subpackage imports for new code, for example
``dreamervla.models.embodiment`` or ``dreamervla.models.embodiment.world_model``.
"""

from dreamervla.models.embodiment.world_model import WorldModel
from dreamervla.models.registry import (
    get_model,
    register_model,
    registered_model_types,
    validate_model_type,
)

__all__ = [
    "WorldModel",
    "get_model",
    "register_model",
    "registered_model_types",
    "validate_model_type",
]
