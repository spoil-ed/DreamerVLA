"""Faithful TransDreamer dynamics Transformer.

Re-implements the architectural choices of Chen et al. 2022 (TransDreamer,
modules_transformer.py / transformer.py):

  * Custom multi-head self-attention with **relative positional bias**
    (Shaw et al. 2018-style learnable bias bucketed by clipped relative
    offset -- in the causal setting only positive offsets exist, so we use
    a `[n_heads, max_rel_pos+1]` parameter).
  * **GRU gating** on the residual paths of both MHA and FFN sublayers
    (Parisotto et al. 2020, "Stabilizing Transformers for Reinforcement
    Learning"). Init bias for the update gate keeps the block near the
    identity at start-up.
  * **Pre-LayerNorm** + final LayerNorm (matches TransDreamer's
    `pre_lnorm=True`).
  * **KV cache** support via per-layer `step(x_t, cache)` for imagination
    rollouts -- avoids the O(H * prefix^2) recompute pattern of the plain
    `CausalTransformerCell`.

External interface mirrors `CausalTransformerCell`:

    cell = TransDreamerTransformerCell(d_model=..., n_heads=..., ...)
    h_seq = cell(x)                   # [B, T, D] -> [B, T, D]   (training)

    cache = cell.init_cache()         # imagination
    h_t, cache = cell.step(x_t, cache)# [B, 1, D] -> [B, 1, D] + updated cache

Calling `step` T times starting from an empty cache produces the same output
(modulo numerical noise) as a single `forward` call on the concatenated
sequence.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ──────────────────────────────────────────────────────────


class GRUGate(nn.Module):
    """Parisotto et al. 2020 GRU gating block.

        r = sigmoid(W_r y + U_r x)
        z = sigmoid(W_z y + U_z x - b_g)
        h_hat = tanh(W_h y + U_h (r * x))
        out = (1 - z) * x + z * h_hat

    With `b_g > 0`, z starts near zero so the layer initialises near the
    identity (output ≈ x), giving the optimiser a stable starting point.
    """

    def __init__(self, dim: int, init_bias: float = 2.0) -> None:
        super().__init__()
        self.W_r = nn.Linear(dim, dim, bias=False)
        self.U_r = nn.Linear(dim, dim, bias=True)
        self.W_z = nn.Linear(dim, dim, bias=False)
        self.U_z = nn.Linear(dim, dim, bias=True)
        self.W_h = nn.Linear(dim, dim, bias=False)
        self.U_h = nn.Linear(dim, dim, bias=True)
        # Push the update gate's bias negative so z ≈ 0 at init -> identity.
        nn.init.constant_(self.U_z.bias, -float(init_bias))
        nn.init.zeros_(self.U_r.bias)
        nn.init.zeros_(self.U_h.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.W_r(y) + self.U_r(x))
        z = torch.sigmoid(self.W_z(y) + self.U_z(x))
        h = torch.tanh(self.W_h(y) + self.U_h(r * x))
        return (1.0 - z) * x + z * h


class RelativeMultiheadAttention(nn.Module):
    """Causal MHA with Shaw-style learnable relative-position bias.

    Positive offsets only (causal: query at q can only attend to keys at k
    where k <= q, so q - k >= 0). Offsets > max_rel_pos are clipped to the
    "max" bucket so attention to very distant context still has a learnable
    bias.

    Bias shape: `[n_heads, max_rel_pos + 1]`.

    Supports KV cache via a `kv_cache` dict carrying `k` and `v` tensors of
    shape `[B, H, T_past, head_dim]`. When provided, the new tokens' K/V
    are appended to the cache before attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        max_rel_pos: int = 128,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.max_rel_pos = int(max_rel_pos)

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # `rel_bias[h, i]` = bias added to attention score where (q - k) == i.
        self.rel_bias = nn.Parameter(torch.zeros(self.n_heads, self.max_rel_pos + 1))

    def _rel_bias(self, T_q: int, T_k: int, q_offset: int, device) -> torch.Tensor:
        q_idx = torch.arange(T_q, device=device).unsqueeze(1) + q_offset   # [T_q, 1]
        k_idx = torch.arange(T_k, device=device).unsqueeze(0)              # [1, T_k]
        rel = (q_idx - k_idx).clamp(min=0, max=self.max_rel_pos)            # [T_q, T_k]
        return self.rel_bias[:, rel]                                        # [H, T_q, T_k]

    def forward(
        self,
        x: torch.Tensor,                    # [B, T_new, D]
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        B, T_new, _ = x.shape
        qkv = self.qkv_proj(x)                                             # [B, T_new, 3D]
        q, k_new, v_new = qkv.chunk(3, dim=-1)
        q = q.view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)   # [B,H,T_new,dh]
        k_new = k_new.view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None and kv_cache.get("k") is not None:
            k = torch.cat([kv_cache["k"], k_new], dim=2)
            v = torch.cat([kv_cache["v"], v_new], dim=2)
        else:
            k = k_new
            v = v_new
        T_k = k.shape[2]
        q_offset = T_k - T_new                                              # past length

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)         # [B,H,T_new,T_k]
        rel_bias = self._rel_bias(T_new, T_k, q_offset, x.device)
        scores = scores + rel_bias.unsqueeze(0)

        # Causal mask: query at position (q_offset + i) sees keys 0..(q_offset + i).
        q_pos = torch.arange(T_new, device=x.device).unsqueeze(1) + q_offset  # [T_new,1]
        k_pos = torch.arange(T_k, device=x.device).unsqueeze(0)                # [1,T_k]
        causal = (k_pos > q_pos)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v                                                       # [B,H,T_new,dh]
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.d_model)
        out = self.out_dropout(self.out_proj(out))

        new_cache = {"k": k, "v": v} if kv_cache is not None else None
        return out, new_cache


class TransDreamerBlock(nn.Module):
    """Pre-LN MHA + GRU-gate, then pre-LN FFN + GRU-gate."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_rel_pos: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelativeMultiheadAttention(d_model, n_heads, dropout, max_rel_pos)
        self.gate1 = GRUGate(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.gate2 = GRUGate(d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        a, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache)
        h = self.gate1(x, a)
        f = self.ffn(self.norm2(h))
        out = self.gate2(h, f)
        return out, new_cache


# ── Public cell ──────────────────────────────────────────────────────────────


class TransDreamerTransformerCell(nn.Module):
    """Faithful TransDreamer dynamics transformer.

    Drop-in replacement for `CausalTransformerCell` from causal_transformer.py
    (matches the [B, T, d_model] -> [B, T, d_model] interface) but adds:
      - relative positional bias (so the prior can localise in time even when
        token content is similar across steps)
      - GRU gating on every residual path (RL-stable training)
      - explicit `step()` with KV cache for fast imagination rollouts.

    Args:
        d_model:      token / hidden dimension
        n_heads:      number of attention heads (must divide d_model)
        n_layers:     number of stacked TransDreamerBlocks
        d_ff:         inner dim of the position-wise feed-forward
        dropout:      dropout in attention, FFN, and residual outputs
        max_rel_pos:  largest relative offset assigned its own bias bucket;
                      offsets beyond this are clipped to the same bucket
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_rel_pos: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.layers = nn.ModuleList([
            TransDreamerBlock(d_model, n_heads, d_ff, dropout, max_rel_pos)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training-time full-sequence forward.

        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model] — h_t at position t encodes z_{<=t}, a_{<=t}.
        """
        for layer in self.layers:
            x, _ = layer(x, kv_cache=None)
        return self.final_norm(x)

    def init_cache(self) -> list[dict]:
        """Empty KV cache (one dict per layer) for `step()` rollout."""
        return [{"k": None, "v": None} for _ in range(self.n_layers)]

    def step(
        self,
        x_t: torch.Tensor,
        caches: list[dict],
    ) -> tuple[torch.Tensor, list[dict]]:
        """Single-step (or chunk) forward with KV cache update.

        Args:
            x_t:    [B, T_new, d_model] new tokens (typically T_new=1)
            caches: list of n_layers dicts produced by `init_cache()` or a
                    previous `step()`
        Returns:
            (out, new_caches) where out has shape [B, T_new, d_model] and
            new_caches[i]['k'/'v'] has shape [B, H, T_past + T_new, head_dim].
        """
        if len(caches) != self.n_layers:
            raise ValueError(
                f"caches length {len(caches)} != n_layers {self.n_layers}"
            )
        new_caches: list[dict] = []
        for layer, cache in zip(self.layers, caches):
            x_t, new_cache = layer(x_t, kv_cache=cache)
            new_caches.append(new_cache)  # type: ignore[arg-type]
        return self.final_norm(x_t), new_caches


__all__ = [
    "GRUGate",
    "RelativeMultiheadAttention",
    "TransDreamerBlock",
    "TransDreamerTransformerCell",
]
