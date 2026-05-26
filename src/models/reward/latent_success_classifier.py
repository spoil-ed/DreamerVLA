"""Transformer classifier over a window of DINO/VLA-hidden latent frames.

Mirrors the VideoMAE classifier in WMPO/reward_model/videomae.py at the
interface level — sliding W-frame window over a [T, latent_dim] sequence,
earliest window with p(success) >= threshold defines finish_step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LatentSuccessClassifierConfig:
    latent_dim: int = 35840
    window: int = 8
    hidden_dim: int = 1024
    num_layers: int = 8
    num_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    # head_type ∈ {transformer, linear, mlp2}. "transformer" is the original
    # 8-layer 137 M model. "linear" is a single nn.Linear(L*W, 2) — the
    # sklearn-LR-equivalent low-capacity head shown to hit F1≈0.87 on real
    # hidden (CLAUDE.md). "mlp2" is a 2-layer GELU MLP between the two.
    head_type: str = "transformer"
    # Time granularity at which the classifier consumes its window:
    #   "action": W consecutive env-step hiddens (the original WMPO setup).
    #   "chunk":  W consecutive chunk-aggregated hiddens, where each chunk
    #             covers ``chunk_size`` env-steps.  Aggregation is controlled
    #             by ``chunk_pool``: "last" (chunk-boundary frame), "first"
    #             (chunk-start frame), or "mean" (average over K frames).
    # Architecture is identical for both modes; only the data layer differs
    # (dataset stride at train time, video subsample at predict_success time).
    granularity: str = "action"
    chunk_size: int = 1
    chunk_pool: str = "last"


class LatentSuccessClassifier(nn.Module):
    """Binary success classifier over a window of latent frames.

    Input shape contract: ``[B, W, latent_dim]`` where W == cfg.window.
    Output: ``[B, 2]`` logits.

    ``cfg.head_type`` selects the architecture:
        - ``transformer``: original 8-layer Transformer (~137 M params)
        - ``linear``: single Linear(L*W, 2) — sklearn-LR equivalent
        - ``mlp2``: Linear(L*W, hidden_dim) → GELU → Dropout → Linear(hidden_dim, 2)
    """

    def __init__(self, cfg: Optional[LatentSuccessClassifierConfig] = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            cfg = LatentSuccessClassifierConfig(**kwargs)
        self.cfg = cfg
        gran = str(getattr(cfg, "granularity", "action"))
        if gran not in ("action", "chunk"):
            raise ValueError(f"unknown granularity: {gran!r} (action|chunk)")
        if gran == "chunk":
            if int(cfg.chunk_size) < 1:
                raise ValueError(f"chunk granularity requires chunk_size >= 1, got {cfg.chunk_size}")
            if str(cfg.chunk_pool) not in ("last", "first", "mean"):
                raise ValueError(f"chunk_pool must be last|first|mean, got {cfg.chunk_pool!r}")
        ht = str(getattr(cfg, "head_type", "transformer"))
        if ht == "transformer":
            self.input_proj = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, cfg.window + 1, cfg.hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=cfg.num_heads,
                dim_feedforward=int(cfg.hidden_dim * cfg.mlp_ratio),
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
            self.head = nn.Linear(cfg.hidden_dim, 2)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif ht == "linear":
            self.head = nn.Linear(cfg.latent_dim * cfg.window, 2)
        elif ht == "mlp2":
            self.head = nn.Sequential(
                nn.Linear(cfg.latent_dim * cfg.window, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, 2),
            )
        else:
            raise ValueError(f"unknown head_type: {ht!r} (transformer|linear|mlp2)")

    def forward(self, latent_window: torch.Tensor) -> torch.Tensor:
        """latent_window: [B, W, latent_dim] -> logits [B, 2]."""
        if latent_window.shape[1] != self.cfg.window:
            raise ValueError(
                f"expected window={self.cfg.window}, got {latent_window.shape[1]}"
            )
        ht = str(getattr(self.cfg, "head_type", "transformer"))
        if ht == "transformer":
            x = self.input_proj(latent_window.to(self.input_proj.weight.dtype))
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1) + self.pos_embed
            x = self.encoder(x)
            return self.head(x[:, 0])
        # linear / mlp2: flatten window into a single feature vector
        B = latent_window.shape[0]
        flat = latent_window.reshape(B, -1).to(next(self.head.parameters()).dtype)
        return self.head(flat)

    def _chunk_aggregate(self, latent_video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Subsample / pool an env-step granular video to chunk granularity.

        Returns:
            chunk_video: ``[B, T_chunk, latent_dim]`` where ``T_chunk = T // K``.
            chunk_end_env_step: ``[T_chunk]`` long — for each chunk index ``c``,
                the env-step index that chunk ``c`` ends at (i.e. ``(c+1)*K-1``).
                Used to translate chunk-unit finish_step back to env-step units
                so downstream code sees a consistent env-step index.
        """
        B, T, D = latent_video.shape
        K = int(self.cfg.chunk_size)
        T_chunk = T // K
        if T_chunk < 1:
            raise ValueError(
                f"chunk classifier needs T >= chunk_size={K} env-step frames, got T={T}"
            )
        pool = str(self.cfg.chunk_pool)
        usable = T_chunk * K
        reshaped = latent_video[:, :usable].reshape(B, T_chunk, K, D)
        if pool == "last":
            chunk_video = reshaped[:, :, -1]
        elif pool == "first":
            chunk_video = reshaped[:, :, 0]
        else:  # mean
            chunk_video = reshaped.mean(dim=2)
        chunk_end_env_step = torch.arange(
            K - 1, usable, K, device=latent_video.device, dtype=torch.long
        )
        return chunk_video, chunk_end_env_step

    @torch.no_grad()
    def predict_success(
        self,
        latent_video: torch.Tensor,
        threshold: float,
        stride: int = 1,
        min_steps: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Earliest-window success scan over a latent video.

        Args:
            latent_video: [B, T, latent_dim] in ENV-STEP granularity. For
                chunk-level classifiers the video is subsampled internally
                using ``self.cfg.chunk_size`` + ``self.cfg.chunk_pool`` so
                callers never have to know about granularity.
            threshold: probability threshold for the success class.
            stride: window stride (in classifier's NATIVE unit: env-step for
                action granularity, chunk for chunk granularity).
            min_steps: earliest window-end position in ENV-STEP units (so the
                online WMPO config can keep one consistent number across both
                granularities).  For chunk classifiers this is converted to
                ``ceil(min_steps / chunk_size)`` chunk units.

        Returns:
            dict with:
                ``complete``: [B] bool
                ``finish_step``: [B] long — earliest window-end index that
                    fired in ENV-STEP units; ``T - 1`` if no window fired.
        """
        B, T, _ = latent_video.shape
        W = self.cfg.window
        device = latent_video.device
        gran = str(getattr(self.cfg, "granularity", "action"))

        if gran == "chunk":
            scan_video, chunk_end_env_step = self._chunk_aggregate(latent_video)
            K = int(self.cfg.chunk_size)
            scan_min_steps = (int(min_steps) + K - 1) // K
        else:
            scan_video = latent_video
            chunk_end_env_step = None
            scan_min_steps = int(min_steps)

        T_scan = scan_video.shape[1]
        complete = torch.zeros(B, dtype=torch.bool, device=device)
        finish_step_scan = torch.full((B,), T_scan - 1, dtype=torch.long, device=device)

        first_end = max(W, scan_min_steps + W)
        ends = list(range(first_end, T_scan + 1, stride))
        for end in ends:
            window = scan_video[:, end - W : end]
            logits = self.forward(window)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            hit = (probs >= threshold) & (~complete)
            if hit.any():
                finish_step_scan = torch.where(
                    hit, torch.full_like(finish_step_scan, end - 1), finish_step_scan
                )
                complete = complete | hit
                if complete.all():
                    break

        if gran == "chunk":
            # Map chunk-index finish_step back to env-step index using the
            # chunk-end env-step lookup. Unfired entries keep T - 1.
            assert chunk_end_env_step is not None
            unfired = ~complete
            finish_env = chunk_end_env_step[finish_step_scan.clamp(max=chunk_end_env_step.shape[0] - 1)]
            finish_env = torch.where(unfired, torch.full_like(finish_env, T - 1), finish_env)
            return {"complete": complete, "finish_step": finish_env}
        return {"complete": complete, "finish_step": finish_step_scan}


__all__ = ["LatentSuccessClassifier", "LatentSuccessClassifierConfig"]
