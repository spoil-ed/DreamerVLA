"""Protocol for swappable WMPO reward definitions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class RewardModel(Protocol):
    """Maps an imagined rollout's success outcome to a per-step reward tensor.

    The verifier emits ``(complete, finish_step)``; a ``RewardModel`` turns that
    into the ``[batch, max_steps]`` reward the WMPO advantage consumes. The default
    sparse-outcome form places ``float(complete)`` at ``finish_step``; dense /
    verifier-shaped forms may return a per-step signal instead.
    """

    name: str

    def build_reward(
        self,
        *,
        batch: int,
        max_steps: int,
        chunk_size: int,
        finish_step: torch.Tensor,
        complete: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a ``[batch, max_steps]`` float32 reward tensor on ``device``."""
        ...
