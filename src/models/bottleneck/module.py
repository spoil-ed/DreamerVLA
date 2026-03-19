from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BottleneckOutput:
    latent: torch.Tensor
    penalty: torch.Tensor


class LinearBottleneck(nn.Module):
    """Simplest bottleneck: one linear projection to a smaller latent."""

    def __init__(self, input_dim: int = 256, latent_dim: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, latent_dim)

    def forward(self, z_sem: torch.Tensor) -> BottleneckOutput:
        latent = self.proj(z_sem)
        penalty = latent.pow(2).mean(dim=-1)
        return BottleneckOutput(latent=latent, penalty=penalty)

    def project(self, latent: torch.Tensor) -> torch.Tensor:
        return self.forward(latent).latent
