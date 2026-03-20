from __future__ import annotations

import torch
from torch import Tensor, nn

from .vla_encoder.encoder import MultimodalEncoder, MultimodalEncoderOutput


class DreamerVLA(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        image_channels: int = 3,
        vocab_size: int = 32000,
        max_language_length: int = 128,
        proprio_dim: int = 16,
        embed_dim: int = 256,
        fused_dim: int = 256,
        image_depth: int = 4,
        language_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        proprio_hidden_dim: int = 256,
        dropout: float = 0.0,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = MultimodalEncoder(
            image_size=image_size,
            patch_size=patch_size,
            image_channels=image_channels,
            vocab_size=vocab_size,
            max_language_length=max_language_length,
            proprio_dim=proprio_dim,
            embed_dim=embed_dim,
            image_depth=image_depth,
            language_depth=language_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            proprio_hidden_dim=proprio_hidden_dim,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
        )

    def forward(
        self,
        image: Tensor,
        language: Tensor,
        proprio: Tensor,
        language_attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor | MultimodalEncoderOutput]:
        encoder_output = self.encoder(
            image=image,
            language=language,
            proprio=proprio,
            language_attention_mask=language_attention_mask,
        )

        pooled_embedding = torch.cat(
            [
                encoder_output.image_embedding,
                encoder_output.language_embedding,
                encoder_output.proprio_embedding,
            ],
            dim=-1,
        )
        multimodal_tokens = torch.cat(
            [
                encoder_output.image_tokens,
                encoder_output.language_tokens,
                encoder_output.proprio_tokens,
            ],
            dim=1,
        )

        return {
            "encoder_output": encoder_output,
            "pooled_embedding": pooled_embedding,
            "multimodal_tokens": multimodal_tokens,
            "latent": self.fusion(pooled_embedding),
        }
