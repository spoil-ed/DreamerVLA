"""DreamerV3-style twohot critic with symlog-transformed value bins.

The critic outputs logits over `num_bins` bins whose centres are linearly spaced
in symlog space. The predicted value is the expectation of those bins mapped
back through `symexp`. The training target is a two-hot encoding of
`symlog(return)`, and the loss is -log_prob of that target under the categorical.

This matches Hafner et al., "Mastering Diverse Domains through World Models"
(DreamerV3), §B.2.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
        targets = torch.zeros(*values.shape, self.num_bins, device=values.device, dtype=values.dtype)
        targets.scatter_(-1, idx_lower.unsqueeze(-1), weight_lower.unsqueeze(-1))
        targets.scatter_add_(-1, idx_upper.unsqueeze(-1), weight_upper.unsqueeze(-1))
        return targets

    def log_prob_of(self, hidden: Tensor, values: Tensor) -> Tensor:
        log_probs = F.log_softmax(self.logits(hidden).float(), dim=-1)
        targets = self.twohot_targets(values).detach()
        return (targets * log_probs).sum(dim=-1)


class ReturnPercentileTracker:
    """EMA tracker for P95 − P5 of returns, used to normalise actor advantages.

    DreamerV3 §B.3: S = max(1, P95(R) − P5(R)); advantage = R / S.
    """

    def __init__(self, decay: float = 0.99, low: float = 0.05, high: float = 0.95) -> None:
        self.decay = float(decay)
        self.low = float(low)
        self.high = float(high)
        self._low_ema: float | None = None
        self._high_ema: float | None = None

    @torch.no_grad()
    def update(self, returns: Tensor) -> tuple[float, float]:
        flat = returns.detach().float().flatten()
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

    def state_dict(self) -> dict:
        return {"decay": self.decay, "low": self.low, "high": self.high,
                "low_ema": self._low_ema, "high_ema": self._high_ema}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.low = float(state.get("low", self.low))
        self.high = float(state.get("high", self.high))
        self._low_ema = state.get("low_ema")
        self._high_ema = state.get("high_ema")


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
    for tb, sb in zip(target.buffers(), source.buffers()):
        tb.data.copy_(sb.data)


__all__ = ["TwohotCritic", "ReturnPercentileTracker", "symlog", "symexp", "soft_update"]
