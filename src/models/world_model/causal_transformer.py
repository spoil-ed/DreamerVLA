"""
Causal Transformer dynamics backbone for TransDreamer-style world model.

Replaces the large WAM backbone (ChameleonXLLMX) used in TSSMWorldModel.
Equivalent to TransDreamer's `self.cell = Transformer(cfg)` in
modules_transformer.py:259.

Input:  [B, T, d_model]   — sequence of (z_t ⊕ a_t) token embeddings
Output: [B, T, d_model]   — h_t at position t encodes history z_{0:t}, a_{0:t}
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CausalTransformerCell(nn.Module):
    """
    Lightweight causal Transformer used as the sequence dynamics backbone.

    Architecture:
        - Standard PyTorch TransformerEncoder with pre-LayerNorm (norm_first=True)
        - Upper-triangular causal mask: position t attends only to 0..t
        - batch_first=True so shapes are [B, T, d_model] throughout

    Comparison with TransDreamer (transformer.py):
        TransDreamer uses a custom MultiheadAttention + GRU gating + positional
        embeddings. This implementation uses PyTorch's built-in encoder for
        simplicity and better hardware utilisation via SDPA.

    Args:
        d_model:  token / hidden dimension
        n_heads:  number of attention heads  (d_model must be divisible by n_heads)
        n_layers: number of Transformer layers
        d_ff:     feedforward inner dimension (typically 4 * d_model)
        dropout:  dropout probability applied in attention and FFN
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # [B, T, d_model]
            norm_first=True,   # pre-LayerNorm: more stable, matches TransDreamer's pre_lnorm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """
        Build an upper-triangular boolean causal mask.

        True  = position is masked (not attended to).
        False = position is visible.

        TransDreamer equivalent:
            transformer.py _generate_square_subsequent_mask (line 215)

        Shape: [T, T]
        """
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]  — input token sequence

        Returns:
            [B, T, d_model]  — contextualised output where position t has
                               attended to all positions 0..t (causal).
        """
        T = x.size(1)
        mask = self._causal_mask(T, x.device)
        return self.encoder(x, mask=mask, is_causal=True)


__all__ = ["CausalTransformerCell"]
