"""Transformer classifier over a window of DINO/VLA-hidden latent frames.

Mirrors the VideoMAE classifier in WMPO/reward_model/videomae.py at the
interface level — sliding W-frame window over a [T, latent_dim] sequence,
earliest window with p(success) >= threshold defines finish_step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LatentSuccessClassifierConfig:
    latent_dim: int | None = None
    action_dim: int = 7
    time_horizon: int = 5
    token_dim: int = 1024
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
    # Tokenized frame windows [B,W,N,D] default to the historical flattened
    # boundary. Scheme-B input-token / backbone latents can set "mean" to keep
    # classifier size tied to token_dim instead of N*token_dim.
    token_pool: str = "flat"
    # When the latent is stored FLAT ([B,W,N*token_dim], the online/replay form)
    # and token_pool="mean", token_count lets forward() reshape flat -> tokens
    # before pooling, so the input projection stays token_dim-sized instead of
    # the (huge) N*token_dim flat dim. None keeps the historical flat behaviour.
    token_count: int | None = None


class LatentSuccessClassifier(nn.Module):
    """Binary success classifier over a window of latent frames.

    Input shape contract: ``[B, W, latent_dim]`` where W == cfg.window. Tokenized
    windows such as ``[B, W, N, token_dim]`` are accepted and flattened at this
    boundary.
    Output: ``[B, 2]`` logits.

    ``cfg.head_type`` selects the architecture:
        - ``transformer``: original 8-layer Transformer (~137 M params)
        - ``linear``: single Linear(L*W, 2) — sklearn-LR equivalent
        - ``mlp2``: Linear(L*W, hidden_dim) → GELU → Dropout → Linear(hidden_dim, 2)
    """

    def __init__(self, cfg: LatentSuccessClassifierConfig | None = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            cfg = LatentSuccessClassifierConfig(**kwargs)
        if cfg.latent_dim is None and str(getattr(cfg, "token_pool", "flat")) == "mean":
            cfg.latent_dim = int(cfg.token_dim)
        if cfg.latent_dim is None:
            cfg.latent_dim = int(cfg.time_horizon) * int(cfg.action_dim) * int(cfg.token_dim)
        self.cfg = cfg
        gran = str(getattr(cfg, "granularity", "action"))
        if gran not in ("action", "chunk"):
            raise ValueError(f"unknown granularity: {gran!r} (action|chunk)")
        if gran == "chunk":
            if int(cfg.chunk_size) < 1:
                raise ValueError(
                    f"chunk granularity requires chunk_size >= 1, got {cfg.chunk_size}"
                )
            if str(cfg.chunk_pool) not in ("last", "first", "mean"):
                raise ValueError(f"chunk_pool must be last|first|mean, got {cfg.chunk_pool!r}")
        token_pool = str(getattr(cfg, "token_pool", "flat"))
        if token_pool not in {"flat", "mean"}:
            raise ValueError(f"token_pool must be flat|mean, got {token_pool!r}")
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
        """latent_window: [B, W, latent_dim] or [B, W, ...] -> logits [B, 2]."""
        if latent_window.shape[1] != self.cfg.window:
            raise ValueError(f"expected window={self.cfg.window}, got {latent_window.shape[1]}")
        if latent_window.ndim > 3:
            token_pool = str(getattr(self.cfg, "token_pool", "flat"))
            if token_pool == "flat":
                latent_window = latent_window.reshape(
                    latent_window.shape[0], latent_window.shape[1], -1
                )
            else:
                latent_window = latent_window.reshape(
                    latent_window.shape[0],
                    latent_window.shape[1],
                    -1,
                    latent_window.shape[-1],
                ).mean(dim=2)
        elif (
            latent_window.ndim == 3
            and str(getattr(self.cfg, "token_pool", "flat")) == "mean"
            and getattr(self.cfg, "token_count", None)
            and int(latent_window.shape[-1]) != int(self.cfg.latent_dim)
        ):
            # FLAT-stored tokenized latent ([B,W,N*token_dim], the online/replay
            # form): reshape to tokens and mean-pool so input_proj stays token_dim.
            tc = int(self.cfg.token_count)
            latent_window = latent_window.reshape(
                latent_window.shape[0], latent_window.shape[1], tc, -1
            ).mean(dim=2)
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

    def _chunk_aggregate(self, latent_video: torch.Tensor) -> torch.Tensor:
        """Subsample / pool an env-step granular video to chunk granularity.

        Returns ``[B, T_chunk, ...]`` where ``T_chunk = T // K``.
        Pooling is controlled by ``self.cfg.chunk_pool`` (last|first|mean).
        """
        B, T = latent_video.shape[:2]
        trailing_shape = latent_video.shape[2:]
        K = int(self.cfg.chunk_size)
        T_chunk = T // K
        if T_chunk < 1:
            raise ValueError(
                f"chunk classifier needs T >= chunk_size={K} env-step frames, got T={T}"
            )
        pool = str(self.cfg.chunk_pool)
        usable = T_chunk * K
        reshaped = latent_video[:, :usable].reshape(B, T_chunk, K, *trailing_shape)
        if pool == "last":
            return reshaped[:, :, -1]
        if pool == "first":
            return reshaped[:, :, 0]
        return reshaped.mean(dim=2)

    @torch.no_grad()
    def predict_success(
        self,
        latent_video: torch.Tensor,
        threshold: float,
        stride: int = 1,
        min_steps: int = 0,
        pre_pooled: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Earliest-window success scan over a latent video.

        Unit convention: ``min_steps``, ``stride``, and the returned
        ``finish_step`` are ALL in the classifier's NATIVE unit:
            - action granularity → env-step
            - chunk granularity  → chunk (one chunk = ``chunk_size`` env-steps)

        ``latent_video`` is always env-step granular (callers don't need to
        pre-pool); chunk classifiers pool internally via
        ``self.cfg.chunk_size`` + ``self.cfg.chunk_pool``. Callers that need
        env-step finish_step must convert at the boundary
        (``finish_chunk * chunk_size + (chunk_size - 1)`` for ``chunk_pool=last``).

        Args:
            latent_video: ``[B, T, latent_dim]`` or ``[B, T, ...]``,
                ENV-STEP granular.
            threshold:    p(success) threshold for the positive class.
            stride:       window stride, NATIVE unit.
            min_steps:    earliest window-end position, NATIVE unit.

        Returns:
            ``complete``    : ``[B]`` bool
            ``finish_step`` : ``[B]`` long — earliest window-end index in
                              NATIVE unit; ``T_scan - 1`` if no window fired
                              (``T_scan = T // chunk_size`` for chunk,
                               ``T`` for action).
        """
        B, T, _ = latent_video.shape
        W = self.cfg.window
        device = latent_video.device
        gran = str(getattr(self.cfg, "granularity", "action"))
        # ``pre_pooled``: caller already aggregated the video to the classifier's
        # native granularity (e.g. WMPO imagination pools each chunk as it is
        # generated, storing 1/K the frames). Skip the internal aggregate so we
        # don't pool twice. Pooling at generation time with the same chunk_pool
        # is identical to ``_chunk_aggregate`` here, so the scan is unchanged.
        scan_video = (
            latent_video
            if (gran != "chunk" or pre_pooled)
            else self._chunk_aggregate(latent_video)
        )

        T_scan = scan_video.shape[1]
        complete = torch.zeros(B, dtype=torch.bool, device=device)
        finish_step = torch.full((B,), T_scan - 1, dtype=torch.long, device=device)

        first_end = max(W, int(min_steps) + W)
        for end in range(first_end, T_scan + 1, stride):
            window = scan_video[:, end - W : end]
            logits = self.forward(window)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            hit = (probs >= threshold) & (~complete)
            if hit.any():
                finish_step = torch.where(hit, torch.full_like(finish_step, end - 1), finish_step)
                complete = complete | hit
                if complete.all():
                    break
        return {"complete": complete, "finish_step": finish_step}


__all__ = ["LatentSuccessClassifier", "LatentSuccessClassifierConfig"]
