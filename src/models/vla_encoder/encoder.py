from __future__ import annotations

from typing import Any

import torch
from torch import nn


class FrozenVLAEncoder(nn.Module):
    """Minimal frozen encoder that maps multi-modal observations into a semantic latent.

    It accepts either:
    - a tensor shaped `[..., semantic_dim]`, or
    - a dict with `image`, `proprio`, and `text` tensors shaped `[..., dim]`.
    """

    def __init__(
        self,
        image_dim: int = 48,
        proprio_dim: int = 8,
        text_dim: int = 16,
        semantic_dim: int = 256,
    ) -> None:
        super().__init__()
        self.input_dim = image_dim + proprio_dim + text_dim
        self.semantic_dim = semantic_dim
        self.proj = nn.Linear(self.input_dim, semantic_dim, bias=False)

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, obs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        if "semantic" in obs:
            return obs["semantic"]

        image = obs["image"]
        proprio = obs["proprio"]
        text = obs["text"]

        original_shape = image.shape[:-1]
        image = image.reshape(-1, image.shape[-1])
        proprio = proprio.reshape(-1, proprio.shape[-1])
        text = text.reshape(-1, text.shape[-1])

        fused = torch.cat([image, proprio, text], dim=-1)
        semantic = self.proj(fused)
        return semantic.reshape(*original_shape, self.semantic_dim)

    def encode(self, obs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(obs)

    def extra_repr(self) -> str:
        return f"semantic_dim={self.semantic_dim}, frozen=True"


def build_frozen_vla_encoder(config: Any) -> FrozenVLAEncoder:
    return FrozenVLAEncoder(
        image_dim=config.model.image_dim,
        proprio_dim=config.model.proprio_dim,
        text_dim=config.model.text_dim,
        semantic_dim=config.model.semantic_dim,
    )
