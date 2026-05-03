"""VLA-as-actor: cotrain actor that reuses VLA's trained ActionHead.

Replaces the small 5M ``VLAPolicy`` MLP with the VLA's own ``ActionHead`` (a
trained Transformer-encoder + L1 regression head). The VLA Chameleon backbone
is *not* used here — imagination has no real obs at t>=1, so we project the
WM feature into the action_head's expected hidden space via a small adapter.

  WM feat [B, 768]
        │
        ▼ adapter (small, trainable, random init)
  fake hidden [B, 1, 4096]
        │
        ▼ hidden_projection (from VLA action_head)
  [B, 1, 1024] context
        │
        ▼ + action_token_embeddings (from VLA action_head)
  [B, 71, 1024]  (1 context + 70 = time_horizon × action_dim slots)
        │
        ▼ transformer_encoder (from VLA action_head)
  [B, 71, 1024]
        │
        ▼ output_projection (L1Regression from VLA action_head)
  predicted_action_chunk [B, time_horizon=10, action_dim=7]
        │
        ▼ take first step
  action mean [B, 7]  +  log_std (learnable)  →  Normal → rsample / log_prob

Provides the same ``forward(batch={'mode': 'sample'/'evaluate', ...})``
dispatcher that the existing cotrain algorithm and FSDP wrap expect, so
nothing else in the cotrain pipeline needs to change.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.models.chameleon_model.modeling_xllmx_chameleon_ck_action_head import (
    L1RegressionActionHead,
)


class VLAActionHeadActor(nn.Module):
    """Reuses VLA's ActionHead modules + a WM-feat adapter.

    Init kwarg ``init_action_head_ckpt`` (a workspace-format VLA ``.ckpt`` —
    e.g. the user's ``epoch=005-train_vla_loss=2.840.ckpt``) is mined for
    ``state_dicts.encoder.backbone.action_head.*`` keys, which are loaded
    here as warm-start.
    """

    def __init__(
        self,
        hidden_dim: int = 768,                 # WM feat dim = latent + d_model
        action_dim: int = 7,
        time_horizon: int = 10,                # must match VLA action_head's time_horizon
        vla_hidden_size: int = 4096,           # Chameleon hidden size; action_head expects this
        hidden_size_factor: float = 0.25,      # → reduced_hidden_size = 1024
        num_encoder_layers: int = 2,           # match VLA action_head
        adapter_hidden_dim: int = 1024,
        initial_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        init_action_head_ckpt: str | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.time_horizon = int(time_horizon)
        self.hidden_size = int(vla_hidden_size)
        self.reduced_hidden_size = int(self.hidden_size * hidden_size_factor)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        # ── Adapter: WM feat → action_head's expected hidden space ───────────
        self.adapter = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(adapter_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(adapter_hidden_dim), self.hidden_size),
        )

        # ── VLA ActionHead components (loaded from ckpt below) ──────────────
        # name + dtype must match exactly so the state_dict load lines up.
        self.action_token_embeddings = nn.Embedding(
            1, self.time_horizon * self.action_dim * self.hidden_size,
        )
        nn.init.normal_(self.action_token_embeddings.weight, std=0.02)

        self.hidden_projection = nn.Linear(self.hidden_size, self.reduced_hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.reduced_hidden_size,
            nhead=4,
            dim_feedforward=self.reduced_hidden_size * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_encoder_layers),
            norm=nn.LayerNorm(self.reduced_hidden_size),
        )
        self.output_projection = L1RegressionActionHead(
            self.reduced_hidden_size,
            self.reduced_hidden_size,
            self.time_horizon,
            self.action_dim,
        )

        # ── Gaussian std for reparameterized exploration ─────────────────────
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(initial_log_std)))

        # ── Optional warm-start from VLA ckpt ────────────────────────────────
        if init_action_head_ckpt:
            self._load_action_head_from_vla_ckpt(str(init_action_head_ckpt))

    # ──────────────────────────────────────────────────────────────────────
    # Warm-start loader
    # ──────────────────────────────────────────────────────────────────────

    def _load_action_head_from_vla_ckpt(self, ckpt_path: str) -> None:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        encoder_sd = payload.get("state_dicts", {}).get("encoder")
        if encoder_sd is None:
            print(f"[VLAActor] no state_dicts.encoder in {ckpt_path}; adapter+head stay random init")
            return
        prefix = "backbone.action_head."
        action_head_sd = {
            k[len(prefix):]: v
            for k, v in encoder_sd.items()
            if k.startswith(prefix)
        }
        if not action_head_sd:
            print(f"[VLAActor] no '{prefix}' keys in encoder state_dict; nothing to warm-start")
            return
        missing, unexpected = self.load_state_dict(action_head_sd, strict=False)
        # `missing` includes adapter / log_std (intentional random init); not a problem.
        non_adapter_missing = [k for k in missing if not k.startswith("adapter.") and k != "log_std"]
        print(
            f"[VLAActor] loaded {len(action_head_sd)} action_head tensors from VLA ckpt; "
            f"unexpected={len(unexpected)}, missing-action-head-only={len(non_adapter_missing)}"
        )
        if non_adapter_missing:
            print(f"[VLAActor] WARN missing action_head tensors (first 5): {non_adapter_missing[:5]}")
        if unexpected:
            print(f"[VLAActor] WARN unexpected (first 5): {unexpected[:5]}")
        del payload

    # ──────────────────────────────────────────────────────────────────────
    # Mean prediction
    # ──────────────────────────────────────────────────────────────────────

    def _action_mean(self, wm_feat: torch.Tensor) -> torch.Tensor:
        """WM feat [B, hidden_dim] → predicted current-step action mean [B, action_dim]."""
        param_dtype = self.adapter[1].weight.dtype
        wm_feat = wm_feat.to(dtype=param_dtype)
        bs = wm_feat.shape[0]

        # 1. Adapt WM feat to look like a length-1 VLA hidden context
        context = self.adapter(wm_feat).unsqueeze(1)                              # [B, 1, 4096]
        context_red = self.hidden_projection(context)                              # [B, 1, 1024]

        # 2. Reuse action_head's learnable action token embeddings (70 = T*A slots)
        action_tokens = self.action_token_embeddings.weight.view(
            1, self.time_horizon * self.action_dim, self.hidden_size,
        ).expand(bs, -1, -1)                                                       # [B, 70, 4096]
        action_tokens_red = self.hidden_projection(action_tokens)                  # [B, 70, 1024]

        # 3. Concat [context, action_tokens] and run the action_head transformer
        combined = torch.cat([context_red, action_tokens_red], dim=1)              # [B, 71, 1024]
        out = self.transformer_encoder(combined)                                   # [B, 71, 1024]

        # 4. Project the action-token outputs to a [B, time_horizon, action_dim] chunk
        action_part = out[:, 1:, :]                                                # [B, 70, 1024]
        action_chunk = self.output_projection(action_part)                          # [B, T, A]

        # 5. RL imagination uses one action per step → take first step of the chunk
        return action_chunk[:, 0, :].float()                                        # [B, A]

    # ──────────────────────────────────────────────────────────────────────
    # FSDP-compatible dispatcher (matches VLAPolicy interface)
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Any]) -> Any:
        """Routes through __call__ so FSDP's auto-gather hook fires."""
        mode = batch.get("mode")
        hidden = batch["hidden"]
        mean = self._action_mean(hidden)
        log_std = (
            self.log_std.clamp(min=self.min_log_std, max=self.max_log_std)
            .unsqueeze(0)
            .expand_as(mean)
        )
        std = log_std.exp()
        dist = Normal(mean, std)
        if mode == "sample":
            deterministic = bool(batch.get("deterministic", False))
            action = mean if deterministic else dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, {"mean": mean, "std": std}
        if mode == "evaluate":
            action = batch["action"]
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, entropy, {"mean": mean, "std": std}
        raise ValueError(f"Unknown VLAActionHeadActor forward mode: {mode!r}")


__all__ = ["VLAActionHeadActor"]
