# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch pytree registration used by the OpenPI SFT boundary.

This is the small SFT-relevant helper from RLinf's Apache-2.0
``rlinf/utils/pytree.py``.  Keeping it here avoids making the sibling RLinf
checkout a DreamerVLA runtime dependency.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from torch.utils import _pytree

_REGISTERED_PYTREE_DATACLASSES: set[type[Any]] = set()


def register_pytree_dataclass_type(cls: type[Any]) -> None:
    """Register one dataclass type so ``tree_map`` can recurse through it."""

    if cls in _REGISTERED_PYTREE_DATACLASSES or not is_dataclass(cls):
        return
    field_names = tuple(field.name for field in fields(cls))

    def flatten(instance: Any) -> tuple[list[Any], list[str]]:
        pairs = [(name, getattr(instance, name)) for name in field_names]
        return [value for _, value in pairs if value is not None], [
            name for name, value in pairs if value is not None
        ]

    def unflatten(values: list[Any], context: list[str]) -> Any:
        return cls(**dict(zip(context, values, strict=True)))

    try:
        _pytree.register_pytree_node(cls, flatten, unflatten)
    except ValueError:
        pass
    _REGISTERED_PYTREE_DATACLASSES.add(cls)


def register_pytree_dataclasses(obj: Any) -> None:
    """Recursively register dataclass instances contained in an object tree."""

    if is_dataclass(obj) and not isinstance(obj, type):
        register_pytree_dataclass_type(type(obj))
        for field in fields(obj):
            register_pytree_dataclasses(getattr(obj, field.name))
        return
    if isinstance(obj, dict):
        for value in obj.values():
            register_pytree_dataclasses(value)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            register_pytree_dataclasses(value)


__all__ = ["register_pytree_dataclass_type", "register_pytree_dataclasses"]
