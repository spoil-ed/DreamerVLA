"""Thin torch.distributed collective group wrapper."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class TorchCollectiveGroup:
    """NCCL/Gloo-capable collective helpers.

    Calls are safe in single-process tests: if ``torch.distributed`` is not
    initialized, methods return cloned local values.
    """

    def __init__(self, process_group: Any | None = None) -> None:
        self.process_group = process_group

    @property
    def is_initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def broadcast_tensor(self, tensor: torch.Tensor, *, src: int = 0) -> torch.Tensor:
        out = tensor.detach().clone()
        if self.is_initialized:
            dist.broadcast(out, src=int(src), group=self.process_group)
        return out

    def broadcast_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        src: int = 0,
    ) -> dict[str, Any]:
        """Broadcast tensor values in a state dict.

        Non-tensor values are copied through unchanged. Tensor shapes must
        already match on all ranks, matching normal model state synchronization.
        """

        out: dict[str, Any] = {}
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                out[name] = self.broadcast_tensor(value, src=src)
            else:
                out[name] = value
        return out
