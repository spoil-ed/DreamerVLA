"""Dreamer-VLA package."""

from __future__ import annotations

from dreamervla.config_resolvers import register_dreamervla_resolvers

register_dreamervla_resolvers()

__all__ = ["__version__"]

__version__ = "0.1.0"
