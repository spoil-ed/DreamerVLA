from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dreamervla.algorithms.actor.openvla_discrete_token_actor import (
    OpenVLADiscreteTokenActor,
)


class LatentToOpenVLADiscreteTokenActor(OpenVLADiscreteTokenActor):
    """Bridge query-before latents to OpenVLA discrete action tokens.

    The actor accepts source VLA/backbone tokens, uses learned action-query slots
    plus a Transformer decoder bridge to produce action-hidden slots, then reuses
    the OpenVLA LM-head categorical action-token decoder. No L1 action head is
    constructed on this route.
    """

    def __init__(
        self,
        hidden_dim: int | None = None,
        source_token_count: int | None = None,
        source_token_dim: int = 4096,
        action_hidden_dim: int = 4096,
        action_dim: int = 7,
        time_horizon: int = 8,
        bridge_hidden_dim: int = 1024,
        num_bridge_layers: int = 2,
        num_bridge_heads: int = 8,
        bridge_dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        action_token_count = int(time_horizon) * int(action_dim)
        super().__init__(
            hidden_dim=action_token_count * int(action_hidden_dim),
            action_hidden_dim=action_hidden_dim,
            action_dim=action_dim,
            time_horizon=time_horizon,
            **kwargs,
        )
        self.source_token_count = (
            int(source_token_count) if source_token_count is not None else None
        )
        self.source_token_dim = int(source_token_dim)
        self.bridge_hidden_dim = int(bridge_hidden_dim)
        if self.bridge_hidden_dim % int(num_bridge_heads) != 0:
            raise ValueError(
                "bridge_hidden_dim must be divisible by num_bridge_heads: "
                f"{self.bridge_hidden_dim} % {int(num_bridge_heads)} != 0"
            )

        expected_flat = (
            None
            if self.source_token_count is None
            else self.source_token_count * self.source_token_dim
        )
        self.hidden_dim = (
            int(hidden_dim)
            if hidden_dim is not None
            else int(expected_flat)
            if expected_flat is not None
            else None
        )
        if (
            expected_flat is not None
            and self.hidden_dim is not None
            and self.hidden_dim != expected_flat
        ):
            raise ValueError(
                "LatentToOpenVLADiscreteTokenActor flat hidden dim mismatch: "
                f"hidden_dim={self.hidden_dim}, expected source_token_count * "
                f"source_token_dim = {expected_flat}"
            )

        self.source_proj = (
            nn.Identity()
            if self.source_token_dim == self.bridge_hidden_dim
            else nn.Linear(self.source_token_dim, self.bridge_hidden_dim)
        )
        self.action_queries = nn.Parameter(
            torch.randn(action_token_count, self.bridge_hidden_dim) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.bridge_hidden_dim,
            nhead=int(num_bridge_heads),
            dim_feedforward=self.bridge_hidden_dim * 4,
            dropout=float(bridge_dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.bridge = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(num_bridge_layers),
            norm=nn.LayerNorm(self.bridge_hidden_dim),
        )
        self.action_hidden_proj = (
            nn.Identity()
            if self.bridge_hidden_dim == self.action_hidden_dim
            else nn.Linear(self.bridge_hidden_dim, self.action_hidden_dim)
        )

    def _source_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 2:
            if self.hidden_dim is not None and int(hidden.shape[-1]) != self.hidden_dim:
                raise ValueError(
                    f"flat hidden dim mismatch: got {hidden.shape[-1]}, expected {self.hidden_dim}"
                )
            if self.source_token_count is None:
                if int(hidden.shape[-1]) % self.source_token_dim != 0:
                    raise ValueError(
                        "flat hidden dim must be divisible by source_token_dim when "
                        "source_token_count is omitted"
                    )
                token_count = int(hidden.shape[-1]) // self.source_token_dim
            else:
                token_count = self.source_token_count
            return hidden.reshape(hidden.shape[0], token_count, self.source_token_dim)
        if hidden.ndim == 3:
            if int(hidden.shape[-1]) != self.source_token_dim:
                raise ValueError(
                    f"source token dim mismatch: got {hidden.shape[-1]}, expected {self.source_token_dim}"
                )
            if self.source_token_count is not None and int(hidden.shape[1]) != int(
                self.source_token_count
            ):
                raise ValueError(
                    "source token count mismatch: "
                    f"got {hidden.shape[1]}, expected {self.source_token_count}"
                )
            return hidden
        raise ValueError(
            f"hidden must be flat [B,N*D] or tokenized [B,N,D], got {tuple(hidden.shape)}"
        )

    def _action_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        source = self._source_tokens(hidden)
        dtype = self.action_queries.dtype
        source = self.source_proj(source.to(dtype=dtype))
        queries = self.action_queries.to(device=source.device, dtype=source.dtype)
        queries = queries.unsqueeze(0).expand(source.shape[0], -1, -1)
        bridged = self.bridge(tgt=queries, memory=source)
        action_hidden = self.action_hidden_proj(bridged)
        adapted = self.adapter(action_hidden)
        if self.adapter_type == "residual_mlp":
            adapted = action_hidden + adapted
        return adapted


__all__ = ["LatentToOpenVLADiscreteTokenActor"]
