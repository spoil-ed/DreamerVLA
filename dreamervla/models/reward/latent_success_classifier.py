"""Transformer classifier over a window of DINO/VLA-hidden latent frames.

Mirrors the VideoMAE classifier in upstream reward_model/videomae.py at the
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
    #   "action": W consecutive env-step hiddens (the original LUMOS setup).
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
    proprio_dim: int = 0
    proprio_emb_dim: int = 0
    num_proprio_repeat: int = 1
    lang_dim: int = 0
    lang_emb_dim: int = 0
    num_lang_repeat: int = 1
    vision_input_norm: bool = True
    task_conditioning: dict | None = None


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
        self.proprio_dim = int(getattr(cfg, "proprio_dim", 0) or 0)
        self.proprio_emb_dim = int(getattr(cfg, "proprio_emb_dim", 0) or 0)
        self.num_proprio_repeat = int(getattr(cfg, "num_proprio_repeat", 1) or 1)
        if self.proprio_emb_dim < 0:
            raise ValueError(f"proprio_emb_dim must be >= 0, got {self.proprio_emb_dim}")
        if self.num_proprio_repeat < 1:
            raise ValueError(
                f"num_proprio_repeat must be >= 1, got {self.num_proprio_repeat}"
            )
        self.proprio_condition_dim = self.proprio_emb_dim * self.num_proprio_repeat
        if self.proprio_condition_dim > 0 and self.proprio_dim < 1:
            raise ValueError("proprio_emb_dim>0 requires proprio_dim>=1")

        self.lang_dim = int(getattr(cfg, "lang_dim", 0) or 0)
        self.lang_emb_dim = int(getattr(cfg, "lang_emb_dim", 0) or 0)
        self.num_lang_repeat = int(getattr(cfg, "num_lang_repeat", 1) or 1)
        if self.lang_emb_dim < 0:
            raise ValueError(f"lang_emb_dim must be >= 0, got {self.lang_emb_dim}")
        if self.num_lang_repeat < 1:
            raise ValueError(f"num_lang_repeat must be >= 1, got {self.num_lang_repeat}")
        self.lang_condition_dim = self.lang_emb_dim * self.num_lang_repeat
        if self.lang_condition_dim > 0 and self.lang_dim < 1:
            raise ValueError("lang_emb_dim>0 requires lang_dim>=1")

        self.obs_token_dim = int(cfg.token_dim) + self.proprio_condition_dim
        self.state_token_dim = self.obs_token_dim + self.lang_condition_dim
        self.supports_proprio_conditioning = self.proprio_condition_dim > 0
        self.supports_language_conditioning = self.lang_condition_dim > 0
        if cfg.latent_dim is None and str(getattr(cfg, "token_pool", "flat")) == "mean":
            cfg.latent_dim = int(self.state_token_dim)
        if cfg.latent_dim is None:
            cfg.latent_dim = int(cfg.time_horizon) * int(cfg.action_dim) * int(cfg.token_dim)
        if (
            self.supports_proprio_conditioning or self.supports_language_conditioning
        ) and int(cfg.latent_dim) != int(self.state_token_dim):
            raise ValueError(
                "LatentSuccessClassifier WM conditioning requires "
                "latent_dim == token_dim + proprio_emb_dim * num_proprio_repeat "
                "+ lang_emb_dim * num_lang_repeat, "
                f"got latent_dim={int(cfg.latent_dim)}, token_dim={int(cfg.token_dim)}, "
                f"proprio_emb_dim={self.proprio_emb_dim}, "
                f"num_proprio_repeat={self.num_proprio_repeat}, "
                f"lang_emb_dim={self.lang_emb_dim}, "
                f"num_lang_repeat={self.num_lang_repeat}"
            )
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
        if self.supports_proprio_conditioning:
            self.proprio_encoder: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.proprio_dim),
                nn.Linear(self.proprio_dim, self.proprio_emb_dim),
            )
        else:
            self.proprio_encoder = None
        if self.supports_language_conditioning:
            self.lang_proj: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.lang_dim),
                nn.Linear(self.lang_dim, self.lang_emb_dim),
            )
        else:
            self.lang_proj = None
        task_cfg = dict(getattr(cfg, "task_conditioning", None) or {})
        self.task_conditioning_enabled = bool(task_cfg.get("enabled", False))
        self.supports_task_conditioning = bool(self.task_conditioning_enabled)
        if self.task_conditioning_enabled:
            num_tasks = int(task_cfg.get("num_tasks", 0) or 0)
            embedding_dim = int(task_cfg.get("embedding_dim", 0) or 0)
            if num_tasks <= 0 or embedding_dim <= 0:
                raise ValueError(
                    "classifier.task_conditioning requires positive num_tasks and embedding_dim"
                )
            if embedding_dim != int(cfg.latent_dim):
                raise ValueError(
                    "LatentSuccessClassifier task_conditioning.embedding_dim must match "
                    f"latent_dim ({embedding_dim} != {int(cfg.latent_dim)})"
                )
            self.task_embedding = nn.Embedding(num_tasks, int(cfg.latent_dim))
        else:
            self.task_embedding = None
        ht = str(getattr(cfg, "head_type", "transformer"))
        if ht == "spatial_tf":
            token_count = int(getattr(cfg, "token_count", 0) or 0)
            if token_count < 1:
                raise ValueError("head_type='spatial_tf' requires token_count >= 1")
            self.vision_proj = nn.Linear(int(cfg.token_dim), int(cfg.hidden_dim))
            self.vision_norm = (
                nn.LayerNorm(int(cfg.token_dim))
                if bool(getattr(cfg, "vision_input_norm", True))
                else nn.Identity()
            )
            self.spatial_cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
            self.frame_pos_embed = nn.Parameter(
                torch.zeros(1, int(cfg.window), 1, int(cfg.hidden_dim))
            )
            self.token_pos_embed = nn.Parameter(
                torch.zeros(1, 1, token_count, int(cfg.hidden_dim))
            )
            self.proprio_token_proj = (
                nn.Linear(self.proprio_condition_dim, int(cfg.hidden_dim))
                if self.supports_proprio_conditioning
                else None
            )
            self.proprio_type_embed = (
                nn.Parameter(torch.zeros(1, 1, 1, int(cfg.hidden_dim)))
                if self.supports_proprio_conditioning
                else None
            )
            self.lang_token_proj = (
                nn.Linear(self.lang_condition_dim, int(cfg.hidden_dim))
                if self.supports_language_conditioning
                else None
            )
            self.lang_type_embed = (
                nn.Parameter(torch.zeros(1, 1, int(cfg.hidden_dim)))
                if self.supports_language_conditioning
                else None
            )
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
            nn.init.trunc_normal_(self.spatial_cls_token, std=0.02)
            nn.init.trunc_normal_(self.frame_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.token_pos_embed, std=0.02)
            if self.proprio_type_embed is not None:
                nn.init.trunc_normal_(self.proprio_type_embed, std=0.02)
            if self.lang_type_embed is not None:
                nn.init.trunc_normal_(self.lang_type_embed, std=0.02)
        elif ht == "transformer":
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
            raise ValueError(
                f"unknown head_type: {ht!r} (spatial_tf|transformer|linear|mlp2)"
            )

    def _encode_proprio(self, proprio: torch.Tensor | None, *, window: int) -> torch.Tensor | None:
        if not self.supports_proprio_conditioning:
            return None
        if proprio is None:
            raise ValueError("proprio is required when proprio_emb_dim>0")
        if proprio.ndim != 3:
            raise ValueError(f"proprio must be [B,W,{self.proprio_dim}], got {tuple(proprio.shape)}")
        if int(proprio.shape[1]) != int(window) or int(proprio.shape[-1]) != self.proprio_dim:
            raise ValueError(
                f"proprio shape mismatch: got {tuple(proprio.shape)}, "
                f"expected [B,{window},{self.proprio_dim}]"
            )
        if self.proprio_encoder is None:
            raise RuntimeError("proprio conditioning is enabled without an encoder")
        first = next(self.proprio_encoder.parameters())
        emb = self.proprio_encoder(proprio.to(device=first.device, dtype=first.dtype))
        if self.num_proprio_repeat > 1:
            emb = emb.repeat(1, 1, self.num_proprio_repeat)
        return emb

    def _project_lang(self, lang_emb: torch.Tensor | None) -> torch.Tensor | None:
        if not self.supports_language_conditioning:
            return None
        if lang_emb is None:
            raise ValueError("lang_emb is required when lang_emb_dim>0")
        if lang_emb.ndim != 2 or int(lang_emb.shape[-1]) != self.lang_dim:
            raise ValueError(
                f"lang_emb shape mismatch: got {tuple(lang_emb.shape)}, "
                f"expected [B,{self.lang_dim}]"
            )
        if self.lang_proj is None:
            raise RuntimeError("language conditioning is enabled without a projection")
        first = next(self.lang_proj.parameters())
        emb = self.lang_proj(lang_emb.to(device=first.device, dtype=first.dtype))
        if self.num_lang_repeat > 1:
            emb = emb.repeat(1, self.num_lang_repeat)
        return emb

    def _vector_window(
        self,
        latent_window: torch.Tensor,
        *,
        proprio: torch.Tensor | None,
        lang_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        if latent_window.ndim > 3:
            token_pool = str(getattr(self.cfg, "token_pool", "flat"))
            if token_pool == "flat":
                if self.supports_proprio_conditioning or self.supports_language_conditioning:
                    raise ValueError(
                        "WM proprio/lang classifier inputs must preserve token "
                        "structure or use token_pool='mean' fallback; token_pool='flat' "
                        "would flatten the token grid."
                    )
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
            tc = int(self.cfg.token_count)
            latent_window = latent_window.reshape(
                latent_window.shape[0], latent_window.shape[1], tc, -1
            ).mean(dim=2)
        proprio_emb = self._encode_proprio(
            proprio, window=int(latent_window.shape[1])
        )
        lang_proj = self._project_lang(lang_emb)
        parts = [latent_window]
        if proprio_emb is not None:
            parts.append(proprio_emb.to(device=latent_window.device, dtype=latent_window.dtype))
        if lang_proj is not None:
            lang_window = lang_proj[:, None, :].expand(
                -1, latent_window.shape[1], -1
            )
            parts.append(lang_window.to(device=latent_window.device, dtype=latent_window.dtype))
        out = torch.cat(parts, dim=-1) if len(parts) > 1 else latent_window
        if int(out.shape[-1]) != int(self.cfg.latent_dim):
            raise ValueError(
                f"classifier vector width mismatch: got {int(out.shape[-1])}, "
                f"expected latent_dim={int(self.cfg.latent_dim)}"
            )
        return out

    def _vision_tokens(self, latent_window: torch.Tensor) -> torch.Tensor:
        if latent_window.ndim == 4:
            tokens = latent_window
        elif (
            latent_window.ndim == 3
            and getattr(self.cfg, "token_count", None)
            and int(latent_window.shape[-1]) == int(self.cfg.token_count) * int(self.cfg.token_dim)
        ):
            tokens = latent_window.reshape(
                latent_window.shape[0],
                latent_window.shape[1],
                int(self.cfg.token_count),
                int(self.cfg.token_dim),
            )
        else:
            raise ValueError(
                "head_type='spatial_tf' requires vision tokens shaped "
                f"[B,W,N,{int(self.cfg.token_dim)}] or flat [B,W,N*D], "
                f"got {tuple(latent_window.shape)}"
            )
        if int(tokens.shape[-1]) == self.obs_token_dim:
            tokens = tokens[..., : int(self.cfg.token_dim)]
        if int(tokens.shape[-1]) != int(self.cfg.token_dim):
            raise ValueError(
                f"vision token dim mismatch: got {int(tokens.shape[-1])}, "
                f"expected {int(self.cfg.token_dim)}"
            )
        return tokens

    def _spatial_forward(
        self,
        latent_window: torch.Tensor,
        *,
        proprio: torch.Tensor | None,
        lang_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = self._vision_tokens(latent_window)
        bsz, window, token_count = int(tokens.shape[0]), int(tokens.shape[1]), int(tokens.shape[2])
        if window != int(self.cfg.window):
            raise ValueError(f"expected window={self.cfg.window}, got {window}")
        if token_count > int(self.token_pos_embed.shape[2]):
            raise ValueError(
                f"token_count {token_count} exceeds configured token_count "
                f"{int(self.token_pos_embed.shape[2])}"
            )
        weight = self.vision_proj.weight
        normed = self.vision_norm(tokens.to(device=weight.device, dtype=weight.dtype))
        vision = self.vision_proj(normed)
        vision = vision + self.frame_pos_embed[:, :window].to(vision.dtype)
        vision = vision + self.token_pos_embed[:, :, :token_count].to(vision.dtype)
        seq_parts = [vision.reshape(bsz, window * token_count, -1)]

        proprio_emb = self._encode_proprio(proprio, window=window)
        if proprio_emb is not None:
            if self.proprio_token_proj is None or self.proprio_type_embed is None:
                raise RuntimeError("proprio spatial token modules are missing")
            p = self.proprio_token_proj(
                proprio_emb.to(
                    device=self.proprio_token_proj.weight.device,
                    dtype=self.proprio_token_proj.weight.dtype,
                )
            )
            p = p[:, :, None, :] + self.frame_pos_embed[:, :window].to(p.dtype)
            p = p + self.proprio_type_embed.to(p.dtype)
            seq_parts.append(p.reshape(bsz, window, -1))

        lang_proj = self._project_lang(lang_emb)
        prefix_parts: list[torch.Tensor] = []
        if lang_proj is not None:
            if self.lang_token_proj is None or self.lang_type_embed is None:
                raise RuntimeError("language spatial token modules are missing")
            lang_token = self.lang_token_proj(
                lang_proj.to(
                    device=self.lang_token_proj.weight.device,
                    dtype=self.lang_token_proj.weight.dtype,
                )
            )[:, None, :]
            prefix_parts.append(lang_token + self.lang_type_embed.to(lang_token.dtype))

        seq = torch.cat(seq_parts, dim=1)
        cls = self.spatial_cls_token.expand(bsz, -1, -1).to(seq.dtype)
        x = torch.cat([cls, *prefix_parts, seq], dim=1)
        x = self.encoder(x)
        return self.head(x[:, 0])

    def forward(
        self,
        latent_window: torch.Tensor,
        *,
        task_ids: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        lang_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """latent_window: [B, W, latent_dim] or token grid -> logits [B, 2]."""
        if latent_window.shape[1] != self.cfg.window:
            raise ValueError(f"expected window={self.cfg.window}, got {latent_window.shape[1]}")
        ht = str(getattr(self.cfg, "head_type", "transformer"))
        if ht == "spatial_tf":
            if self.task_conditioning_enabled:
                raise ValueError("task_id conditioning is not supported by spatial_tf")
            return self._spatial_forward(
                latent_window,
                proprio=proprio,
                lang_emb=lang_emb,
            )
        latent_window = self._vector_window(
            latent_window,
            proprio=proprio,
            lang_emb=lang_emb,
        )
        if self.task_conditioning_enabled:
            if task_ids is None:
                raise ValueError(
                    "task_ids are required when classifier task conditioning is enabled"
                )
            if self.task_embedding is None:
                raise RuntimeError("task conditioning is enabled without an embedding")
            task_emb = self.task_embedding(task_ids.to(latent_window.device).long())
            latent_window = latent_window + task_emb[:, None, :].to(latent_window.dtype)
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
        task_ids: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        lang_emb: torch.Tensor | None = None,
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
            ``score``       : ``[B]`` float — max ``p(success)`` seen by the
                              sliding scan. This gives LUMOS a continuous value
                              source when sparse threshold outcomes are
                              all-success or all-fail inside a GRPO group.
            ``score_step``  : ``[B]`` long — window-end index for ``score`` in
                              the classifier's NATIVE unit.
        """
        if latent_video.ndim < 3:
            raise ValueError(
                f"latent_video must be [B,T,...], got {tuple(latent_video.shape)}"
            )
        B, T = int(latent_video.shape[0]), int(latent_video.shape[1])
        W = self.cfg.window
        device = latent_video.device
        gran = str(getattr(self.cfg, "granularity", "action"))
        # ``pre_pooled``: caller already aggregated the video to the classifier's
        # native granularity (e.g. LUMOS imagination pools each chunk as it is
        # generated, storing 1/K the frames). Skip the internal aggregate so we
        # don't pool twice. Pooling at generation time with the same chunk_pool
        # is identical to ``_chunk_aggregate`` here, so the scan is unchanged.
        scan_video = (
            latent_video
            if (gran != "chunk" or pre_pooled)
            else self._chunk_aggregate(latent_video)
        )
        scan_proprio = None
        if proprio is not None:
            if proprio.ndim != 3:
                raise ValueError(
                    f"proprio video must be [B,T,{self.proprio_dim}], got {tuple(proprio.shape)}"
                )
            if int(proprio.shape[0]) != B or int(proprio.shape[1]) != T:
                raise ValueError(
                    f"proprio video time shape {tuple(proprio.shape[:2])} does not "
                    f"match latent_video {(B, T)}"
                )
            scan_proprio = (
                proprio
                if (gran != "chunk" or pre_pooled)
                else self._chunk_aggregate(proprio)
            )

        T_scan = scan_video.shape[1]
        complete = torch.zeros(B, dtype=torch.bool, device=device)
        finish_step = torch.full((B,), T_scan - 1, dtype=torch.long, device=device)
        score = torch.zeros(B, dtype=torch.float32, device=device)
        score_step = torch.full((B,), T_scan - 1, dtype=torch.long, device=device)

        first_end = max(W, int(min_steps) + W)
        for end in range(first_end, T_scan + 1, stride):
            window = scan_video[:, end - W : end]
            proprio_window = (
                scan_proprio[:, end - W : end] if scan_proprio is not None else None
            )
            logits = self.forward(
                window,
                task_ids=task_ids,
                proprio=proprio_window,
                lang_emb=lang_emb,
            )
            probs = torch.softmax(logits, dim=-1)[:, 1]
            better = probs > score
            if better.any():
                score = torch.where(better, probs.float(), score)
                score_step = torch.where(
                    better,
                    torch.full_like(score_step, end - 1),
                    score_step,
                )
            hit = (probs >= threshold) & (~complete)
            if hit.any():
                finish_step = torch.where(
                    hit,
                    torch.full_like(finish_step, end - 1),
                    finish_step,
                )
                complete = complete | hit
        return {
            "complete": complete,
            "finish_step": finish_step,
            "score": score,
            "score_step": score_step,
        }


__all__ = ["LatentSuccessClassifier", "LatentSuccessClassifierConfig"]
