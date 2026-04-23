"""PyTorch port of DreamerV3's BlockLinear.

Reference: dreamerv3/embodied/jax/nets.py:254 (class BlockLinear).

Semantics: a grouped linear layer. The input's last dim is split into `groups`
equal-sized chunks; each chunk is mapped through its own independent weight
matrix; the outputs are concatenated back along the last dim.

This gives a more parameter-efficient linear (params = in*out/groups instead
of in*out) with a structured inductive bias: each group specialises on a
disjoint slice of the input feature space.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class BlockLinear(nn.Module):
    """Grouped linear layer (DreamerV3-style).

    out_features must be divisible by groups; same for in_features.
    Each group g has its own weight [in/groups, out/groups].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert in_features % groups == 0, (in_features, groups)
        assert out_features % groups == 0, (out_features, groups)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.groups = int(groups)
        in_g = in_features // groups
        out_g = out_features // groups
        # weight shape [groups, in_g, out_g] — each group has its own matrix
        self.weight = nn.Parameter(torch.empty(groups, in_g, out_g))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Trunc-normal-ish init scaled by fan_in (per-group fan_in = in/groups).
        # Matches DreamerV3 behaviour closely enough for our purposes.
        fan_in = self.in_features // self.groups
        std = 1.0 / math.sqrt(fan_in)
        nn.init.trunc_normal_(self.weight, std=std, a=-2.0 * std, b=2.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_features] → [..., out_features]."""
        *leading, in_dim = x.shape
        assert in_dim == self.in_features, (x.shape, self.in_features)
        # [..., groups, in_g]
        x = x.reshape(*leading, self.groups, self.in_features // self.groups)
        # einsum: '...g i, g i o -> ...g o'
        out = torch.einsum("...gi,gio->...go", x, self.weight.to(dtype=x.dtype))
        out = out.reshape(*leading, self.out_features)
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype)
        return out


__all__ = ["BlockLinear"]
