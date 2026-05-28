"""Compatibility exports for DreamerVLA actor policy modules.

Canonical implementations live under ``dreamer_vla.models.actor``.
"""

from dreamer_vla.models.actor.vla_policy import SharedObservationEmbedding, VLAPolicy

__all__ = ["SharedObservationEmbedding", "VLAPolicy"]
