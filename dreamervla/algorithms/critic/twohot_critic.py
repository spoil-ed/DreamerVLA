"""DreamerV3-style twohot critic with symlog-transformed value bins.

The critic outputs logits over `num_bins` bins whose centres are linearly spaced
in symlog space. The predicted value is the expectation of those bins mapped
back through `symexp`. The training target is a two-hot encoding of
`symlog(return)`, and the loss is -log_prob of that target under the categorical.

This matches Hafner et al., "Mastering Diverse Domains through World Models"
(DreamerV3), §B.2.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dreamervla.utils.polyak import soft_update


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight.to(dtype=dtype)


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported critic activation: {name}")


def _norm(name: str, dim: int) -> nn.Module | None:
    name = str(name).lower()
    if name in {"none", "identity", ""}:
        return None
    if name == "rms":
        return RMSNorm(dim)
    if name in {"layer", "layernorm"}:
        return nn.LayerNorm(dim)
    raise ValueError(f"Unsupported critic norm: {name}")


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwohotCritic(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        num_bins: int = 255,
        bin_min: float = -20.0,
        bin_max: float = 20.0,
        critic_layers: int = 1,
        activation: str = "gelu",
        norm: str = "none",
        outscale: float = 1.0,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim!r}")
        if int(critic_hidden_dim) <= 0:
            raise ValueError(f"critic_hidden_dim must be > 0, got {critic_hidden_dim!r}")
        if int(num_bins) < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins!r}")
        if not math.isfinite(float(bin_min)) or not math.isfinite(float(bin_max)):
            raise ValueError("bin_min and bin_max must be finite")
        if float(bin_min) >= float(bin_max):
            raise ValueError(f"bin_min must be < bin_max, got {bin_min!r} >= {bin_max!r}")
        if int(critic_layers) < 0:
            raise ValueError(f"critic_layers must be >= 0, got {critic_layers!r}")
        self.num_bins = int(num_bins)
        modules: list[nn.Module] = []
        cur_dim = int(hidden_dim)
        for _ in range(int(critic_layers)):
            modules.append(nn.Linear(cur_dim, int(critic_hidden_dim)))
            norm_layer = _norm(norm, int(critic_hidden_dim))
            if norm_layer is not None:
                modules.append(norm_layer)
            modules.append(_activation(activation))
            cur_dim = int(critic_hidden_dim)
        final = nn.Linear(cur_dim, int(num_bins))
        if float(outscale) != 1.0:
            with torch.no_grad():
                final.weight.mul_(float(outscale))
                if final.bias is not None:
                    final.bias.mul_(float(outscale))
        modules.append(final)
        self.backbone = nn.Sequential(*modules)
        bins = torch.linspace(float(bin_min), float(bin_max), int(num_bins))
        self.register_buffer("bins", bins, persistent=False)

    def logits(self, hidden: Tensor) -> Tensor:
        # Match param dtype (FSDP MixedPrecision casts gathered Linear weights
        # to bf16 inside forward).
        first_linear = next(module for module in self.backbone if isinstance(module, nn.Linear))
        weight_dtype = first_linear.weight.dtype
        hidden = hidden.to(dtype=weight_dtype)
        return self.backbone(hidden)

    def _expected_value(self, hidden: Tensor) -> Tensor:
        probs = F.softmax(self.logits(hidden).float(), dim=-1)
        expected_symlog_value = (probs * self.bins.to(probs.dtype)).sum(dim=-1)
        return symexp(expected_symlog_value)

    def forward(self, hidden):
        """FSDP-compatible dispatcher.

        Tensor input → expected value (existing behaviour, used by the target
        critic bootstrap path).
        Dict input with 'mode' key → routes to log_prob_of so FSDP's all-gather
        hook fires on the (otherwise custom) call.
        """
        if isinstance(hidden, dict):
            mode = hidden.get("mode")
            if mode == "log_prob":
                return self.log_prob_of(hidden["hidden"], hidden["values"])
            if mode == "value":
                return self._expected_value(hidden["hidden"])
            raise ValueError(f"Unknown TwohotCritic forward mode: {mode!r}")
        return self._expected_value(hidden)

    def twohot_targets(self, values: Tensor) -> Tensor:
        bins = self.bins.to(values.dtype).to(values.device)
        sym_values = symlog(values).clamp(min=bins[0], max=bins[-1])
        idx_upper = torch.bucketize(sym_values.contiguous(), bins)
        idx_upper = idx_upper.clamp(min=1, max=self.num_bins - 1)
        idx_lower = idx_upper - 1
        lower_bin = bins[idx_lower]
        upper_bin = bins[idx_upper]
        denom = (upper_bin - lower_bin).clamp_min(1e-8)
        weight_upper = ((sym_values - lower_bin) / denom).clamp(0.0, 1.0)
        weight_lower = 1.0 - weight_upper
        targets = torch.zeros(
            *values.shape, self.num_bins, device=values.device, dtype=values.dtype
        )
        targets.scatter_(-1, idx_lower.unsqueeze(-1), weight_lower.unsqueeze(-1))
        targets.scatter_add_(-1, idx_upper.unsqueeze(-1), weight_upper.unsqueeze(-1))
        return targets

    def log_prob_of(self, hidden: Tensor, values: Tensor) -> Tensor:
        log_probs = F.log_softmax(self.logits(hidden).float(), dim=-1)
        targets = self.twohot_targets(values).detach()
        return (targets * log_probs).sum(dim=-1)


class ReturnPercentileTracker:
    """EMA tracker for P95 − P5 of returns, used to normalise actor advantages.

    DreamerV3 §B.3: S = max(1, P95(R) − P5(R)); actor advantage
    uses (return − value baseline) / S. DreamerV3 metrics report
    normalised returns as (return - P5) / S.
    """

    def __init__(self, decay: float = 0.99, low: float = 0.05, high: float = 0.95) -> None:
        self.decay = float(decay)
        self.low = float(low)
        self.high = float(high)
        self._low_ema: float | None = None
        self._high_ema: float | None = None
        self._validate_geometry()

    def _validate_geometry(self) -> None:
        if not math.isfinite(self.decay) or not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"decay must be finite and in [0, 1], got {self.decay!r}")
        if not math.isfinite(self.low) or not 0.0 <= self.low <= 1.0:
            raise ValueError(f"low must be finite and in [0, 1], got {self.low!r}")
        if not math.isfinite(self.high) or not 0.0 <= self.high <= 1.0:
            raise ValueError(f"high must be finite and in [0, 1], got {self.high!r}")
        if self.low > self.high:
            raise ValueError(f"low must be <= high, got {self.low!r} > {self.high!r}")

    @torch.no_grad()
    def update(self, returns: Tensor) -> tuple[float, float]:
        flat = returns.detach().float().flatten()
        if flat.numel() == 0:
            raise ValueError("returns must be non-empty")
        if not bool(torch.isfinite(flat).all()):
            raise ValueError("returns must contain only finite values")
        low_q = float(torch.quantile(flat, self.low).item())
        high_q = float(torch.quantile(flat, self.high).item())
        if self._low_ema is None:
            self._low_ema, self._high_ema = low_q, high_q
        else:
            self._low_ema = self.decay * self._low_ema + (1.0 - self.decay) * low_q
            self._high_ema = self.decay * self._high_ema + (1.0 - self.decay) * high_q
        return self._low_ema, self._high_ema

    def scale(self) -> float:
        if self._low_ema is None or self._high_ema is None:
            return 1.0
        return max(1.0, self._high_ema - self._low_ema)

    def offset(self) -> float:
        if self._low_ema is None:
            return 0.0
        return self._low_ema

    def stats(self) -> tuple[float, float]:
        return self.offset(), self.scale()

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "low": self.low,
            "high": self.high,
            "low_ema": self._low_ema,
            "high_ema": self._high_ema,
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.low = float(state.get("low", self.low))
        self.high = float(state.get("high", self.high))
        self._low_ema = state.get("low_ema")
        self._high_ema = state.get("high_ema")
        self._validate_geometry()
        for name, value in (("low_ema", self._low_ema), ("high_ema", self._high_ema)):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when present, got {value!r}")


# soft_update now lives in dreamervla.utils.polyak (generic, model-independent);
# re-exported here for existing importers.
__all__ = ["ReturnPercentileTracker", "TwohotCritic", "soft_update", "symexp", "symlog"]
