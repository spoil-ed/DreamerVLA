"""CausalTransformerCell V2 — minimal upgrades over the original.

Same compute envelope as `causal_transformer.CausalTransformerCell` (4 layers,
d_model=512, pre-LN), but with the three architectural fixes flagged in our
TransDreamer-vs-this-repo audit:

  + **Learnable absolute positional embeddings** (`pos_embed[1, max_seq_len, d]`)
    added once at the input.  Without this, the original cell has *no* way to
    distinguish position from content -- when latent z collapses, all tokens
    look identical to attention.

  + **Optional GRU gating** on residual paths (`use_gru_gate=True` switches it
    on; default False keeps the parameter count close to the original cell).
    Init bias keeps gates near identity at start-up.

  + **KV-cache step() interface** for imagination rollouts -- avoids the
    O(H * prefix^2) recompute pattern.

Designed as a strict drop-in replacement for `CausalTransformerCell`: the
original cell's `(d_model, n_heads, n_layers, d_ff, dropout)` constructor
keywords work unchanged.  The two new keywords (`max_seq_len`, `use_gru_gate`)
have safe defaults.

Compared to `transdreamer_transformer.TransDreamerTransformerCell`:
  - V2 keeps absolute (not relative) positional info -- simpler, fewer params.
  - V2 makes GRU gating optional rather than mandatory.
  - V2 attention uses fused SDPA (`scaled_dot_product_attention`) for speed;
    relative-bias attention in the TransDreamer cell is hand-rolled.

Use V2 if you want the smallest possible delta from the current code; use the
TransDreamer cell if you want a faithful re-implementation of Chen et al. 2022.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ──────────────────────────────────────────────────────────


class _GRUGate(nn.Module):
    """Parisotto et al. 2020 GRU gating block (init near identity)."""

    def __init__(self, dim: int, init_bias: float = 2.0) -> None:
        super().__init__()
        self.W_r = nn.Linear(dim, dim, bias=False)
        self.U_r = nn.Linear(dim, dim, bias=True)
        self.W_z = nn.Linear(dim, dim, bias=False)
        self.U_z = nn.Linear(dim, dim, bias=True)
        self.W_h = nn.Linear(dim, dim, bias=False)
        self.U_h = nn.Linear(dim, dim, bias=True)
        nn.init.constant_(self.U_z.bias, -float(init_bias))
        nn.init.zeros_(self.U_r.bias)
        nn.init.zeros_(self.U_h.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.W_r(y) + self.U_r(x))
        z = torch.sigmoid(self.W_z(y) + self.U_z(x))
        h = torch.tanh(self.W_h(y) + self.U_h(r * x))
        return (1.0 - z) * x + z * h


class _CausalSelfAttention(nn.Module):
    """Standard causal MHA via fused SDPA, with KV cache support.

    Training path: T_q == T_k, `is_causal=True` engages SDPA's causal kernel.
    Step path: T_q small (typically 1), KV cache appended to past, mask
    automatically holds because the new query is the latest position.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=True)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout_p = float(dropout)

    def forward(
        self,
        x: torch.Tensor,                    # [B, T_new, D]
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        B, T_new, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k_new, v_new = qkv.chunk(3, dim=-1)
        q = q.view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = k_new.view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None and kv_cache.get("k") is not None:
            k = torch.cat([kv_cache["k"], k_new], dim=2)
            v = torch.cat([kv_cache["v"], v_new], dim=2)
        else:
            k = k_new
            v = v_new

        # is_causal works only when T_q == T_k *and* there is no KV history.
        # In the step() path the new query is the latest position so the mask
        # is automatically satisfied without SDPA's causal flag.
        is_causal = (kv_cache is None) and (T_new > 1)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.d_model)
        out = self.out_proj(out)

        new_cache = {"k": k, "v": v} if kv_cache is not None else None
        return out, new_cache


class _Block(nn.Module):
    """Pre-LN MHA + (residual or GRU gate), then pre-LN FFN + (residual or GRU gate)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        use_gru_gate: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.use_gru = bool(use_gru_gate)
        if self.use_gru:
            self.gate1 = _GRUGate(d_model)
            self.gate2 = _GRUGate(d_model)
        else:
            self.gate1 = None
            self.gate2 = None

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        a, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache)
        h = self.gate1(x, a) if self.use_gru else (x + a)
        f = self.ffn(self.norm2(h))
        out = self.gate2(h, f) if self.use_gru else (h + f)
        return out, new_cache


# ── Public cell ──────────────────────────────────────────────────────────────


class CausalTransformerCellV2(nn.Module):
    """Lightweight causal transformer with positional embeddings + KV cache.

    Drop-in replacement for `causal_transformer.CausalTransformerCell` with
    the same constructor keywords plus two new ones:

        max_seq_len:   maximum sequence length expected at forward time;
                       size of the learnable positional embedding table.
                       Must be >= max(T_train, T_imagine).
        use_gru_gate:  when True, replaces the standard residual sums with
                       Parisotto-style GRU gates on both attention and FFN
                       sublayers.

    Args:
        d_model:        token / hidden dimension
        n_heads:        number of attention heads (must divide d_model)
        n_layers:       number of stacked blocks
        d_ff:           inner dim of the position-wise feed-forward
        dropout:        dropout in attention, FFN, and residual outputs
        max_seq_len:    table size for the absolute positional embedding
        use_gru_gate:   enable GRU gating on residual paths
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        use_gru_gate: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.max_seq_len = int(max_seq_len)
        self.use_gru_gate = bool(use_gru_gate)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            _Block(d_model, n_heads, d_ff, dropout, use_gru_gate)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training-time full-sequence forward.

        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        B, T, _ = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len={self.max_seq_len}"
            )
        x = x + self.pos_embed[:, :T]
        for layer in self.layers:
            x, _ = layer(x, kv_cache=None)
        return self.final_norm(x)

    def init_cache(self) -> dict:
        """Empty KV cache + position counter for `step()` rollout."""
        return {
            "layers": [{"k": None, "v": None} for _ in range(self.n_layers)],
            "pos": 0,
        }

    def step(
        self,
        x_t: torch.Tensor,
        cache: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Single- or chunk-step forward with KV cache update.

        Args:
            x_t:   [B, T_new, d_model] new tokens (typically T_new=1)
            cache: dict from `init_cache()` or a previous `step()`
        Returns:
            (out, new_cache) — out has shape [B, T_new, d_model]; new_cache
            holds K/V history of length pos+T_new and the updated position.
        """
        pos = int(cache.get("pos", 0))
        B, T_new, _ = x_t.shape
        if pos + T_new > self.max_seq_len:
            raise ValueError(
                f"step position {pos + T_new} exceeds max_seq_len={self.max_seq_len}"
            )
        x_t = x_t + self.pos_embed[:, pos : pos + T_new]
        layer_caches = cache.get("layers")
        if layer_caches is None or len(layer_caches) != self.n_layers:
            raise ValueError("cache.layers missing or wrong length; use init_cache().")
        new_layer_caches: list[dict] = []
        for layer, layer_cache in zip(self.layers, layer_caches):
            x_t, new_layer_cache = layer(x_t, kv_cache=layer_cache)
            new_layer_caches.append(new_layer_cache)  # type: ignore[arg-type]
        return self.final_norm(x_t), {"layers": new_layer_caches, "pos": pos + T_new}


__all__ = ["CausalTransformerCellV2"]
