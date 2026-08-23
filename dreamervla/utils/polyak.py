"""Polyak (soft) parameter averaging — a generic target-network update.

Lives in utils, not in a specific critic model, so any actor-critic algorithm
can reuse it without importing a concrete model module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """In-place Polyak update: target ← (1-tau)·target + tau·source (buffers copied)."""
    for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
    for tb, sb in zip(target.buffers(), source.buffers(), strict=True):
        tb.data.copy_(sb.data)


__all__ = ["soft_update"]
