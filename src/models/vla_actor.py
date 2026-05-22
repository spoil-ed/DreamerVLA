"""Compatibility exports for actor implementations.

Canonical implementations live under ``src.models.actor``.  This module keeps
existing Hydra targets such as ``src.models.vla_actor.Pi0ActionHiddenActor``
working.
"""

from src.models.actor import BaseActor, Pi0ActionHiddenActor, VLAActionHeadActor

__all__ = [
    "BaseActor",
    "Pi0ActionHiddenActor",
    "VLAActionHeadActor",
]
