from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence


def _masked_mean(hidden_states: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(hidden_states.dtype)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (hidden_states * weights).sum(dim=1) / denom


@dataclass
class MultimodalEncoderOutput:
    image_embedding: Tensor
    language_embedding: Tensor
    proprio_embedding: Tensor
    image_tokens: Tensor
    language_tokens: Tensor
    proprio_tokens: Tensor


class SimpleViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.image_size = image_size
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        if image.ndim != 4:
            raise ValueError("image must have shape [batch, channels, height, width].")

        _, _, height, width = image.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected image size [{self.image_size}, {self.image_size}], "
                f"got [{height}, {width}]."
            )

        patches = self.patch_embed(image)
        patches = patches.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(image.shape[0], -1, -1)
        hidden_states = torch.cat([cls_token, patches], dim=1)
        hidden_states = hidden_states + self.pos_embed[:, : hidden_states.shape[1]]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.norm(hidden_states)

        pooled = hidden_states[:, 0]
        tokens = hidden_states[:, 1:]
        return pooled, tokens


class SimpleLanguageEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 128,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        pad_token_id: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.max_length = max_length
        self.pad_token_id = pad_token_id

        self.token_embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_token_id,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        language: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if language.ndim != 2:
            raise ValueError("language must have shape [batch, sequence_length].")

        _, seq_len = language.shape
        if seq_len > self.max_length:
            raise ValueError(
                f"language length {seq_len} exceeds configured max_length {self.max_length}."
            )

        if attention_mask is None:
            attention_mask = language.ne(self.pad_token_id)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        hidden_states = self.token_embed(language)
        hidden_states = hidden_states + self.pos_embed[:, :seq_len]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.encoder(
            hidden_states,
            src_key_padding_mask=~attention_mask,
        )
        hidden_states = self.norm(hidden_states)

        pooled = _masked_mean(hidden_states, attention_mask)
        return pooled, hidden_states


class ProprioMLPEncoder(nn.Module):
    def __init__(
        self,
        proprio_dim: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, proprio: Tensor) -> tuple[Tensor, Tensor]:
        if proprio.ndim != 2:
            raise ValueError("proprio must have shape [batch, proprio_dim].")

        pooled = self.encoder(proprio)
        tokens = pooled.unsqueeze(1)
        return pooled, tokens


class MultimodalEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        image_channels: int = 3,
        vocab_size: int = 32000,
        max_language_length: int = 128,
        proprio_dim: int = 16,
        embed_dim: int = 256,
        image_depth: int = 4,
        language_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        proprio_hidden_dim: int = 256,
        dropout: float = 0.0,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.image_encoder = SimpleViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=image_channels,
            embed_dim=embed_dim,
            depth=image_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.language_encoder = SimpleLanguageEncoder(
            vocab_size=vocab_size,
            max_length=max_language_length,
            embed_dim=embed_dim,
            depth=language_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pad_token_id=pad_token_id,
            dropout=dropout,
        )
        self.proprio_encoder = ProprioMLPEncoder(
            proprio_dim=proprio_dim,
            embed_dim=embed_dim,
            hidden_dim=proprio_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        image: Tensor,
        language: Tensor,
        proprio: Tensor,
        language_attention_mask: Optional[Tensor] = None,
    ) -> MultimodalEncoderOutput:
        image_embedding, image_tokens = self.image_encoder(image)
        language_embedding, language_tokens = self.language_encoder(
            language=language,
            attention_mask=language_attention_mask,
        )
        proprio_embedding, proprio_tokens = self.proprio_encoder(proprio)

        return MultimodalEncoderOutput(
            image_embedding=image_embedding,
            language_embedding=language_embedding,
            proprio_embedding=proprio_embedding,
            image_tokens=image_tokens,
            language_tokens=language_tokens,
            proprio_tokens=proprio_tokens,
        )


