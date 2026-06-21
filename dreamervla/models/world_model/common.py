"""Model-agnostic primitives shared across DreamerVLA world models."""

from __future__ import annotations

import torch
import torch.nn as nn

from dreamervla.models.world_model.block_linear import BlockLinear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight.to(dtype=dtype)


class ChannelRMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.square().mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        weight = self.weight.to(dtype=dtype).view(1, -1, 1, 1)
        return (x32 * rms).to(dtype) * weight


def act(name: str) -> nn.Module:
    name = str(name).lower()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


def _module_ref_tensor(module: nn.Module) -> torch.Tensor | None:
    for tensor in module.parameters(recurse=True):
        return tensor
    for tensor in module.buffers(recurse=True):
        return tensor
    # DataParallel replicas can expose copied weights as plain Tensor
    # attributes instead of registered Parameters.
    for child in module.modules():
        for attr in ("weight", "bias"):
            tensor = getattr(child, attr, None)
            if isinstance(tensor, torch.Tensor):
                return tensor
    return None


def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    tensor = _module_ref_tensor(module)
    return tensor.dtype if tensor is not None else fallback


def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    tensor = _module_ref_tensor(module)
    return tensor.device if tensor is not None else fallback


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: int = 1,
        units: int = 1024,
        act: str = "silu",
        outscale: float = 1.0,
    ) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        cur = int(in_dim)
        for _ in range(int(layers)):
            activation = globals()["act"](act)
            mods.extend([nn.Linear(cur, int(units)), RMSNorm(int(units)), activation])
            cur = int(units)
        final = nn.Linear(cur, int(out_dim))
        if float(outscale) != 1.0:
            with torch.no_grad():
                final.weight.mul_(float(outscale))
                if final.bias is not None:
                    final.bias.mul_(float(outscale))
        mods.append(final)
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, dim: int, act: str) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = globals()["act"](act)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        return x + self.fc2(h)


class ResMLPHead(nn.Module):
    """ResNet-style MLP head: input projection, residual blocks, output projection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: int = 2,
        units: int = 8192,
        act: str = "silu",
        outscale: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(in_dim), int(units))
        self.blocks = nn.ModuleList([ResBlock(int(units), act) for _ in range(int(layers))])
        self.norm_out = RMSNorm(int(units))
        final = nn.Linear(int(units), int(out_dim))
        if float(outscale) != 1.0:
            with torch.no_grad():
                final.weight.mul_(float(outscale))
                if final.bias is not None:
                    final.bias.mul_(float(outscale))
        self.output_proj = final

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(self.norm_out(h))


__all__ = [
    "BlockLinear",
    "ChannelRMSNorm",
    "MLPHead",
    "RMSNorm",
    "ResBlock",
    "ResMLPHead",
    "_module_device",
    "_module_dtype",
    "act",
]
