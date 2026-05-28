from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _act(name: str) -> nn.Module:
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
            mods.extend([nn.Linear(cur, int(units)), RMSNorm(int(units)), _act(act)])
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


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class SymexpTwoHotHead(nn.Module):
    """DreamerV3 symexp-twohot scalar reward/value head."""

    def __init__(
        self,
        in_dim: int,
        bins: int = 255,
        layers: int = 1,
        units: int = 1024,
        act: str = "silu",
        outscale: float = 0.0,
    ) -> None:
        super().__init__()
        self.bins_count = int(bins)
        self.net = MLPHead(
            in_dim,
            self.bins_count,
            layers=layers,
            units=units,
            act=act,
            outscale=outscale,
        )

    def _bins(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.bins_count % 2 == 1:
            half = torch.linspace(
                -20.0, 0.0, (self.bins_count - 1) // 2 + 1, device=device, dtype=dtype
            )
            half = symexp(half)
            return torch.cat([half, -half[:-1].flip(0)], dim=0)
        half = torch.linspace(
            -20.0, 0.0, self.bins_count // 2, device=device, dtype=dtype
        )
        half = symexp(half)
        return torch.cat([half, -half.flip(0)], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(device=logits.device, dtype=logits.dtype).detach()
        bins = self._bins(logits.device, logits.dtype)
        below = (bins.view(*((1,) * target.ndim), -1) <= target.unsqueeze(-1)).sum(
            dim=-1
        ) - 1
        above = self.bins_count - (
            bins.view(*((1,) * target.ndim), -1) > target.unsqueeze(-1)
        ).sum(dim=-1)
        below = below.clamp(0, self.bins_count - 1).long()
        above = above.clamp(0, self.bins_count - 1).long()
        equal = below == above
        below_bins = bins[below]
        above_bins = bins[above]
        dist_to_below = torch.where(
            equal, torch.ones_like(target), (below_bins - target).abs()
        )
        dist_to_above = torch.where(
            equal, torch.ones_like(target), (above_bins - target).abs()
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total.clamp_min(1e-8)
        weight_above = dist_to_below / total.clamp_min(1e-8)
        target_dist = F.one_hot(below, self.bins_count).to(
            dtype=logits.dtype
        ) * weight_below.unsqueeze(-1) + F.one_hot(above, self.bins_count).to(
            dtype=logits.dtype
        ) * weight_above.unsqueeze(-1)
        return -(target_dist * F.log_softmax(logits, dim=-1)).sum(dim=-1)

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        bins = self._bins(logits.device, logits.dtype)
        n = logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = probs[..., :m]
            p2 = probs[..., m : m + 1]
            p3 = probs[..., m + 1 :]
            b1 = bins[..., :m]
            b2 = bins[..., m : m + 1]
            b3 = bins[..., m + 1 :]
            return (p2 * b2).sum(dim=-1) + ((p1 * b1).flip(-1) + (p3 * b3)).sum(dim=-1)
        p1 = probs[..., : n // 2]
        p2 = probs[..., n // 2 :]
        b1 = bins[..., : n // 2]
        b2 = bins[..., n // 2 :]
        return ((p1 * b1).flip(-1) + (p2 * b2)).sum(dim=-1)


class BinaryRewardHead(nn.Module):
    """Bernoulli reward head for sparse or terminal-window 0/1 labels."""

    def __init__(
        self,
        in_dim: int,
        layers: int = 1,
        units: int = 1024,
        act: str = "silu",
        init_logit: float = -5.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.net = MLPHead(in_dim, 1, layers=layers, units=units, act=act)
        self.pos_weight = None if pos_weight is None else float(pos_weight)
        final = self.net.net[-1]
        if isinstance(final, nn.Linear) and final.bias is not None:
            nn.init.constant_(final.bias, float(init_logit))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(-1).float()
        target = (
            target.to(device=logits.device, dtype=torch.float32)
            .detach()
            .clamp(0.0, 1.0)
        )
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = torch.tensor(
                self.pos_weight, device=logits.device, dtype=logits.dtype
            )
        return F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=pos_weight
        )

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits.squeeze(-1).float()).to(dtype=logits.dtype)


def _make_reward_head(
    feat_dim: int,
    reward_bins: int,
    hidden: int,
    act: str,
    reward_head_type: str = "twohot",
    reward_init_logit: float = -5.0,
    reward_pos_weight: float | None = None,
) -> nn.Module:
    reward_head_type = str(reward_head_type).lower()
    if reward_head_type in {"binary", "bernoulli", "sigmoid"}:
        return BinaryRewardHead(
            feat_dim,
            layers=1,
            units=hidden,
            act=act,
            init_logit=reward_init_logit,
            pos_weight=reward_pos_weight,
        )
    if reward_head_type not in {"twohot", "symexp_twohot", "regression", "mse"}:
        raise ValueError(f"Unsupported reward_head_type: {reward_head_type}")
    if int(reward_bins) <= 1:
        return MLPHead(feat_dim, 1, layers=1, units=hidden, act=act)
    return SymexpTwoHotHead(feat_dim, bins=reward_bins, layers=1, units=hidden, act=act)


def _reward_loss(
    head: nn.Module, pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    target = target.to(device=pred.device, dtype=pred.dtype)
    if hasattr(head, "loss"):
        return head.loss(pred, target).mean()
    return F.mse_loss(pred.squeeze(-1), target, reduction="none").mean()


def _reward_pred(head: nn.Module, pred: torch.Tensor) -> torch.Tensor:
    if hasattr(head, "pred"):
        return head.pred(pred)
    return pred.squeeze(-1)


__all__ = [
    "BinaryRewardHead",
    "SymexpTwoHotHead",
    "_make_reward_head",
    "_reward_loss",
    "_reward_pred",
    "symexp",
]
