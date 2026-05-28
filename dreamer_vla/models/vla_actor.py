"""Compatibility exports for actor implementations.

Canonical implementations live under ``dreamer_vla.models.actor``.  This module keeps
existing Hydra targets such as ``dreamer_vla.models.vla_actor.Pi0ActionHiddenActor``
working.
"""

from dreamer_vla.models.actor import BaseActor, Pi0ActionHiddenActor, VLAActionHeadActor

__all__ = [
    "BaseActor",
    "Pi0ActionHiddenActor",
    "VLAActionHeadActor",
]
