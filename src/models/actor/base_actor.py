from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal


class BaseActor(nn.Module, ABC):
    """Base interface for DreamerVLA actor modules."""

    log_std: nn.Parameter
    min_log_std: float
    max_log_std: float

    def _normal_from_action_chunk(self, action_chunk: torch.Tensor) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        mean = action_chunk[:, 0, :].float()
        log_std = (
            self.log_std.clamp(min=self.min_log_std, max=self.max_log_std)
            .unsqueeze(0)
            .expand_as(mean)
        )
        std = log_std.exp()
        return Normal(mean, std), mean, std

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> Any:
        raise NotImplementedError


__all__ = ["BaseActor"]
