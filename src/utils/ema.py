"""FSDP-compatible EMA helper (per-parameter shadows)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EMAHelper:
    """Maintain exponential-moving-average shadows of a module's parameters.

    Works with FSDP-sharded parameters: each rank keeps shadows matching the
    shapes of its local shard, so updates and checkpointing stay rank-local.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
    ) -> None:
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.optimization_step = 0
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def step(self, model: nn.Module) -> None:
        self.optimization_step += 1
        if self.optimization_step <= self.update_after_step:
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].copy_(param.detach())
            return
        for name, param in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "optimization_step": self.optimization_step,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.update_after_step = int(state_dict.get("update_after_step", self.update_after_step))
        self.optimization_step = int(state_dict.get("optimization_step", 0))
        self.shadow = dict(state_dict.get("shadow", {}))


__all__ = ["EMAHelper"]
