"""Discrete-token I/O for the TSSM world model (io_mode="token").

Counterpart of the hidden-space Route-B pipeline:

  hidden mode:  bpe ids → [frozen Chameleon] → [B, N_img, 4096]  → ConvEncoderStem
  token  mode:  bpe ids → [bpe→img_idx map] → [ImageTokenEmbedder] → [B, N_img, d_embed] → ConvEncoderStem

On the decode side, the existing BspaceConvDecoderHead is reused with
``out_channels = num_image_tokens_vocab`` so its per-spatial-position output
is already the image-vocab logits (no frozen lm_head needed).
"""
from __future__ import annotations

import torch
from torch import nn


class ImageTokenEmbedder(nn.Module):
    """Image bpe ids (already mapped to image-vocab indices) → spatial token embeddings.

    Input:  img_idx  [..., N_img]  long, values in [0, num_image_tokens_vocab)
    Output: embeds   [..., N_img, d_embed]

    The caller is responsible for converting raw BPE ids to image-vocab
    indices via the ``_bpe_to_img_idx`` buffer maintained on the WM (same
    mapping already used by the hidden-mode CE loss).
    """

    def __init__(
        self,
        num_image_tokens_vocab: int,
        d_embed: int = 512,
        spatial: tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        self.num_image_tokens_vocab = int(num_image_tokens_vocab)
        self.d_embed = int(d_embed)
        self.spatial = (int(spatial[0]), int(spatial[1]))
        n_img = self.spatial[0] * self.spatial[1]

        self.token_embed = nn.Embedding(self.num_image_tokens_vocab, self.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_img, self.d_embed))

        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, img_idx: torch.Tensor) -> torch.Tensor:
        if img_idx.dtype != torch.long:
            img_idx = img_idx.long()
        x = self.token_embed(img_idx)
        # pos_embed is [1, N_img, d_embed]; broadcasts over any leading dims.
        return x + self.pos_embed


__all__ = ["ImageTokenEmbedder"]
