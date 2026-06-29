from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dreamervla.models.actor._load import extract_state_dict
from dreamervla.models.actor.base_actor import BaseActor
from dreamervla.models.embodiment.chameleon_model.modeling_xllmx_chameleon_ck_action_head import (
    L1RegressionActionHead,
)
from dreamervla.utils.hf_checkpoint import is_hf_checkpoint, load_hf_prefixed_tensors


class LatentToActionHiddenActor(BaseActor):
    """Bridge tokenized WM latents to VLA action-head slots.

    Scheme-B world models predict input-side image-token latents, not
    action-query latents.  This actor keeps the WM token boundary:
    it accepts ``[B,N,D]`` or flat ``[B,N*D]`` latents, lets learned action
    queries attend to those source tokens, and decodes the resulting action
    slots with the usual L1 action output head.
    """

    def __init__(
        self,
        hidden_dim: int | None = None,
        source_token_count: int | None = None,
        source_token_dim: int = 4096,
        action_hidden_dim: int = 1024,
        action_dim: int = 7,
        time_horizon: int = 5,
        bridge_hidden_dim: int = 1024,
        num_bridge_layers: int = 2,
        num_bridge_heads: int = 8,
        bridge_dropout: float = 0.1,
        freeze_output_projection: bool = True,
        initial_log_std: float = -3.0,
        min_log_std: float = -5.0,
        max_log_std: float = -2.0,
        freeze_log_std: bool = False,
        init_action_head_ckpt: str | None = None,
        head_type: str = "legacy",
        **_: Any,
    ) -> None:
        super().__init__()
        self.source_token_count = (
            int(source_token_count) if source_token_count is not None else None
        )
        self.source_token_dim = int(source_token_dim)
        self.action_hidden_dim = int(action_hidden_dim)
        self.action_dim = int(action_dim)
        self.time_horizon = int(time_horizon)
        self.action_token_count = self.time_horizon * self.action_dim
        self.bridge_hidden_dim = int(bridge_hidden_dim)
        self.head_type = str(head_type).lower()
        if self.head_type not in {"legacy", "oft_l1_regression"}:
            raise ValueError(
                "LatentToActionHiddenActor head_type must be 'legacy' or 'oft_l1_regression'"
            )
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
                "LatentToActionHiddenActor flat hidden dim mismatch: "
                f"hidden_dim={self.hidden_dim}, expected source_token_count * "
                f"source_token_dim = {expected_flat}"
            )

        self.source_proj = (
            nn.Identity()
            if self.source_token_dim == self.bridge_hidden_dim
            else nn.Linear(self.source_token_dim, self.bridge_hidden_dim)
        )
        self.action_queries = nn.Parameter(
            torch.randn(self.action_token_count, self.bridge_hidden_dim) * 0.02
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
        self.output_projection = L1RegressionActionHead(
            self.action_hidden_dim,
            self.action_hidden_dim,
            self.time_horizon,
            self.action_dim,
        )
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(initial_log_std)),
            requires_grad=not bool(freeze_log_std),
        )
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.freeze_log_std = bool(freeze_log_std)

        if init_action_head_ckpt:
            self._load_output_projection(str(init_action_head_ckpt))
        elif self.head_type == "oft_l1_regression" and bool(freeze_output_projection):
            # Freezing a randomly-initialised OFT L1 head produces garbage actions
            # silently. The discrete OpenVLA-OFT one-trajectory checkpoint has no
            # `action_head--*.pt`, so fail loudly with guidance instead.
            raise ValueError(
                "LatentToActionHiddenActor(head_type='oft_l1_regression', "
                "freeze_output_projection=True) needs an L1 action head: set "
                "`policy.init_action_head_ckpt` to a checkpoint that contains "
                "`action_head--*.pt` tensors. The discrete OFT one-trajectory "
                "checkpoint has no L1 head — either supply an L1-finetuned OFT "
                "checkpoint, or set freeze_output_projection=false to train the "
                "head from scratch."
            )
        if bool(freeze_output_projection):
            for param in self.output_projection.parameters():
                param.requires_grad = False

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
        return self.action_hidden_proj(bridged)

    def _action_chunk(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_hidden = self._action_hidden(hidden)
        param_dtype = next(self.output_projection.parameters()).dtype
        actions = self.output_projection(action_hidden.to(dtype=param_dtype))
        chunk = actions.reshape(action_hidden.shape[0], self.time_horizon, self.action_dim).float()
        return chunk, action_hidden.float()

    def _load_output_projection(self, ckpt_path: str) -> None:
        payload: Any | None = None
        tensors: dict[str, torch.Tensor] = {}
        if is_hf_checkpoint(ckpt_path):
            tensors = load_hf_prefixed_tensors(
                ckpt_path, "action_head.output_projection."
            )
        else:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            tensors = self._extract_output_projection_state_dict(payload)

        if not tensors:
            raise RuntimeError(f"No compatible action output projection found: {ckpt_path}")
        missing, unexpected = self.output_projection.load_state_dict(tensors, strict=False)
        print(
            "[LatentToActionHiddenActor] loaded "
            f"{len(tensors)} output_projection tensors from {ckpt_path}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        del payload

    def _extract_output_projection_state_dict(self, payload: Any) -> dict[str, torch.Tensor]:
        if not isinstance(payload, dict):
            return {}
        candidates: list[dict[str, Any]] = []
        for key in ("state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                candidates.append(value)
        encoder_sd = payload.get("state_dicts", {}).get("encoder")
        if isinstance(encoder_sd, dict):
            prefix = "backbone.action_head.output_projection."
            candidates.append(
                {
                    key[len(prefix) :]: value
                    for key, value in encoder_sd.items()
                    if isinstance(key, str) and key.startswith(prefix)
                }
            )
        candidates.append(payload)

        return extract_state_dict(
            candidates,
            ("module.", "output_projection.", "action_head."),
            set(self.output_projection.state_dict().keys()),
        )

    def reference_action_chunk(self, hidden: torch.Tensor) -> torch.Tensor:
        action_hidden = self._action_hidden(hidden)
        param_dtype = next(self.output_projection.parameters()).dtype
        actions = self.output_projection(action_hidden.to(dtype=param_dtype))
        return actions.reshape(action_hidden.shape[0], self.time_horizon, self.action_dim).float()

    def forward(self, batch: dict[str, Any]) -> Any:
        mode = batch.get("mode")
        hidden = batch["hidden"]
        action_chunk, action_hidden = self._action_chunk(hidden)
        dist, mean, std = self._normal_from_action_chunk(action_chunk)
        chunk_dist, mean_chunk, std_chunk = self._normal_from_full_action_chunk(action_chunk)
        if mode == "sample":
            deterministic = bool(batch.get("deterministic", False))
            extra = {
                "mean": mean,
                "std": std,
                "mean_chunk": mean_chunk,
                "std_chunk": std_chunk,
                "action_chunk": mean_chunk,
                "action_hidden": action_hidden,
            }
            if bool(batch.get("return_chunk", False)):
                action = mean_chunk if deterministic else chunk_dist.rsample()
                log_prob = chunk_dist.log_prob(action).sum(dim=(-1, -2))
                return action, log_prob, extra
            action = mean if deterministic else dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, extra
        if mode == "evaluate":
            action = batch["action"]
            if action.ndim == 3:
                log_prob = chunk_dist.log_prob(action).sum(dim=(-1, -2))
                entropy = chunk_dist.entropy().sum(dim=(-1, -2))
                return (
                    log_prob,
                    entropy,
                    {
                        "mean": mean,
                        "std": std,
                        "mean_chunk": mean_chunk,
                        "std_chunk": std_chunk,
                        "action_hidden": action_hidden,
                    },
                )
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return (
                log_prob,
                entropy,
                {
                    "mean": mean,
                    "std": std,
                    "mean_chunk": mean_chunk,
                    "std_chunk": std_chunk,
                    "action_hidden": action_hidden,
                },
            )
        raise ValueError(f"Unknown LatentToActionHiddenActor forward mode: {mode!r}")


__all__ = ["LatentToActionHiddenActor"]
