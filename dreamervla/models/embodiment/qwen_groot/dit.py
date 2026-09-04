# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion Transformer blocks migrated from SiPAI.

Source: ``SIPAI@sipai-main-9672af6/sipai/models/base_modules/dit.py``.
The module stays local so Qwen-GR00T has no runtime dependency on SiPAI.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    """Embed discrete diffusion timesteps."""

    def __init__(self, embedding_dim: int, compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        del compute_dtype
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=1,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))


class AdaLayerNorm(nn.Module):
    """Adaptive layer norm conditioned on timestep embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ) -> None:
        super().__init__()
        self.chunk_dim = chunk_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(
            embedding_dim,
            norm_eps,
            norm_elementwise_affine,
        )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if temb is None:
            raise ValueError("AdaLayerNorm requires a timestep embedding")
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    """Transformer block with optional cross attention and adaptive norm."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: str | None = None,
        num_positional_embeddings: int | None = None,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ) -> None:
        super().__init__()
        del attention_type
        self.dim = dim
        self.norm_type = norm_type
        if positional_embeddings and num_positional_embeddings is None:
            raise ValueError("num_positional_embeddings is required with positional embeddings")
        self.pos_embed = (
            SinusoidalPositionalEmbedding(
                dim,
                max_seq_length=num_positional_embeddings,
            )
            if positional_embeddings == "sinusoidal"
            else None
        )
        self.norm1 = (
            AdaLayerNorm(dim)
            if norm_type == "ada_norm"
            else nn.LayerNorm(
                dim,
                elementwise_affine=norm_elementwise_affine,
                eps=norm_eps,
            )
        )
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        norm_hidden_states = (
            self.norm1(hidden_states, temb)
            if self.norm_type == "ada_norm"
            else self.norm1(hidden_states)
        )
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)
        attention_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        if self.final_dropout is not None:
            attention_output = self.final_dropout(attention_output)
        hidden_states = attention_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    """SiPAI/GR00T diffusion transformer used by the action head."""

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: int | None = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype: torch.dtype = torch.float32,
        final_dropout: bool = True,
        positional_embeddings: str | None = "sinusoidal",
        interleave_self_attention: bool = False,
        cross_attention_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del num_embeds_ada_norm, kwargs
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.gradient_checkpointing = False
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim,
            compute_dtype=compute_dtype,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=(
                        None if idx % 2 == 1 and interleave_self_attention else cross_attention_dim
                    ),
                )
                for idx in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if timestep is None:
            raise ValueError("DiT requires timesteps")
        timestep_embedding = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        all_hidden_states = [hidden_states]
        for index, block in enumerate(self.transformer_blocks):
            self_attention = index % 2 == 1 and self.config.interleave_self_attention
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=None if self_attention else encoder_hidden_states,
                encoder_attention_mask=(None if self_attention else encoder_attention_mask),
                temb=timestep_embedding,
            )
            all_hidden_states.append(hidden_states)
        shift, scale = self.proj_out_1(F.silu(timestep_embedding)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        output = self.proj_out_2(hidden_states)
        return (output, all_hidden_states) if return_all_hidden_states else output


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    """Self-attention-only transformer used to refine Qwen tokens."""

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: int | None = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype: torch.dtype = torch.float32,
        final_dropout: bool = True,
        positional_embeddings: str | None = "sinusoidal",
        interleave_self_attention: bool = False,
    ) -> None:
        super().__init__()
        del output_dim, num_embeds_ada_norm, compute_dtype, interleave_self_attention
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.gradient_checkpointing = False
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)
        return (hidden_states, all_hidden_states) if return_all_hidden_states else hidden_states


__all__ = [
    "AdaLayerNorm",
    "BasicTransformerBlock",
    "DiT",
    "SelfAttentionTransformer",
    "TimestepEncoder",
]
