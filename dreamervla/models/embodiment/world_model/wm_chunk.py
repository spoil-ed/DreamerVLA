from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dreamervla.models.embodiment.world_model.wm import WorldModel


class _WMStyleFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _WMStyleAttention(nn.Module):
    """WM-style attention with residual dim independent of QKV inner dim."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        attn_impl: str = "manual",
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if dim_head < 1:
            raise ValueError(f"dim_head must be >= 1, got {dim_head}")
        if attn_impl not in ("manual", "sdpa"):
            raise ValueError(
                f"attn_impl must be 'manual' or 'sdpa', got {attn_impl!r}"
            )
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.attn_impl = str(attn_impl)
        self.dropout_p = float(dropout)
        self.scale = float(dim_head) ** -0.5
        inner_dim = self.heads * self.dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _dim = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.reshape(bsz, seq_len, self.heads, self.dim_head).transpose(1, 2)
            for t in qkv
        )
        if self.attn_impl == "sdpa":
            attn_mask = (
                mask.to(device=q.device, dtype=q.dtype)[None, None]
                if mask is not None
                else None
            )
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=self.scale,
            )
            out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
            return self.to_out(out)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            dots = dots + mask.to(device=dots.device, dtype=dots.dtype)[None, None]
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.to_out(out)


class _WMStyleTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
        attn_impl: str = "manual",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _WMStyleAttention(
                            dim,
                            heads=int(heads),
                            dim_head=int(dim_head),
                            dropout=float(dropout),
                            attn_impl=str(attn_impl),
                        ),
                        _WMStyleFeedForward(
                            dim, int(mlp_dim), dropout=float(dropout)
                        ),
                    ]
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return self.norm(x)


class ChunkAwareWorldModel(WorldModel):
    """Chunk WM over OpenVLA-OFT hidden tokens with WM-style conditioning.

    The transition model keeps each observation token in source token space and
    concatenates an encoded action to every observation token channel, matching
    the default WM ``concat_dim=1`` pattern.  A chunk is rolled out
    autoregressively: every step predicts ``e_{t+1}`` from the latest
    ``num_hist`` latent frames conditioned on the current action, then slides
    the predicted observation tokens into the next history.
    """

    def __init__(
        self,
        *args: Any,
        chunk_size: int = 8,
        chunk_rollout_chunks: int = 1,
        chunk_rollout_loss_scale: float = 0.0,
        grad_checkpoint: bool = False,
        action_emb_dim: int = 10,
        num_action_repeat: int = 1,
        proprio_dim: int = 0,
        proprio_emb_dim: int = 0,
        num_proprio_repeat: int = 1,
        proprio_reconstruction_loss_scale: float = 0.0,
        lang_dim: int = 0,
        lang_emb_dim: int = 0,
        num_lang_repeat: int = 1,
        dim_head: int = 64,
        attn_impl: str = "manual",
        task_conditioning: dict | None = None,
        **kwargs: Any,
    ) -> None:
        args_list = list(args)
        requested_model_dim = (
            args_list[5] if len(args_list) > 5 else kwargs.get("model_dim")
        )
        token_dim_hint = int(args_list[3] if len(args_list) > 3 else kwargs.get("token_dim", 4096))
        heads_hint = int(args_list[7] if len(args_list) > 7 else kwargs.get("heads", 8))
        safe_parent_model_dim = max(token_dim_hint, heads_hint)
        if safe_parent_model_dim % heads_hint != 0:
            safe_parent_model_dim += heads_hint - (safe_parent_model_dim % heads_hint)
        if len(args_list) > 5:
            args_list[5] = safe_parent_model_dim
        else:
            kwargs = dict(kwargs)
            kwargs["model_dim"] = safe_parent_model_dim

        super().__init__(*args_list, **kwargs)
        self.action_emb_dim = int(action_emb_dim)
        self.num_action_repeat = int(num_action_repeat)
        self.action_condition_dim = self.action_emb_dim * self.num_action_repeat
        if self.action_emb_dim < 1:
            raise ValueError(f"action_emb_dim must be >= 1, got {action_emb_dim}")
        if self.num_action_repeat < 1:
            raise ValueError(
                f"num_action_repeat must be >= 1, got {num_action_repeat}"
            )
        self.proprio_dim = int(proprio_dim)
        self.proprio_emb_dim = int(proprio_emb_dim)
        self.num_proprio_repeat = int(num_proprio_repeat)
        self.proprio_reconstruction_loss_scale = float(
            proprio_reconstruction_loss_scale
        )
        if self.proprio_reconstruction_loss_scale < 0:
            raise ValueError(
                "proprio_reconstruction_loss_scale must be >= 0, got "
                f"{proprio_reconstruction_loss_scale}"
            )
        if self.proprio_emb_dim < 0:
            raise ValueError(
                f"proprio_emb_dim must be >= 0, got {proprio_emb_dim}"
            )
        if self.num_proprio_repeat < 1:
            raise ValueError(
                f"num_proprio_repeat must be >= 1, got {num_proprio_repeat}"
            )
        self.proprio_condition_dim = self.proprio_emb_dim * self.num_proprio_repeat
        if self.proprio_condition_dim > 0:
            if self.proprio_dim < 1:
                raise ValueError("proprio_emb_dim>0 requires proprio_dim>=1")
            self.proprio_encoder: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.proprio_dim),
                nn.Linear(self.proprio_dim, self.proprio_emb_dim),
            )
            self.proprio_decoder: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.proprio_condition_dim),
                nn.Linear(self.proprio_condition_dim, self.proprio_dim),
            )
        else:
            self.proprio_encoder = None
            self.proprio_decoder = None
        self.obs_token_dim = self.token_dim + self.proprio_condition_dim

        self.lang_dim = int(lang_dim)
        self.lang_emb_dim = int(lang_emb_dim)
        self.num_lang_repeat = int(num_lang_repeat)
        if self.lang_emb_dim < 0:
            raise ValueError(f"lang_emb_dim must be >= 0, got {lang_emb_dim}")
        if self.num_lang_repeat < 1:
            raise ValueError(f"num_lang_repeat must be >= 1, got {num_lang_repeat}")
        self.lang_condition_dim = self.lang_emb_dim * self.num_lang_repeat
        if self.lang_condition_dim > 0:
            if self.lang_dim < 1:
                raise ValueError("lang_emb_dim>0 requires lang_dim>=1")
            self.lang_proj: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.lang_dim),
                nn.Linear(self.lang_dim, self.lang_emb_dim),
            )
        else:
            self.lang_proj = None

        expected_model_dim = (
            self.obs_token_dim + self.lang_condition_dim + self.action_condition_dim
        )
        if requested_model_dim is None:
            requested_model_dim = expected_model_dim
        self.model_dim = int(requested_model_dim)
        if self.model_dim != expected_model_dim:
            raise ValueError(
                "ChunkAwareWorldModel uses WM concat conditioning; "
                "set model_dim == token_dim + proprio_emb_dim * num_proprio_repeat "
                "+ lang_emb_dim * num_lang_repeat + action_emb_dim * "
                "num_action_repeat, "
                f"got model_dim={self.model_dim}, token_dim={self.token_dim}, "
                f"proprio_emb_dim={self.proprio_emb_dim}, "
                f"num_proprio_repeat={self.num_proprio_repeat}, "
                f"lang_emb_dim={self.lang_emb_dim}, "
                f"num_lang_repeat={self.num_lang_repeat}, "
                f"action_emb_dim={self.action_emb_dim}, "
                f"num_action_repeat={self.num_action_repeat}"
            )
        if int(chunk_size) < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if int(chunk_rollout_chunks) < 1:
            raise ValueError(
                f"chunk_rollout_chunks must be >= 1, got {chunk_rollout_chunks}"
            )
        if float(chunk_rollout_loss_scale) < 0:
            raise ValueError(
                f"chunk_rollout_loss_scale must be >= 0, got {chunk_rollout_loss_scale}"
            )
        self.chunk_size = int(chunk_size)
        # Close-loop multi-chunk rollout (anti-drift, à la Dreamer rollout loss):
        # train the model to predict chunk c+1..c+N-1 from its OWN predicted
        # history rolled forward from chunk 0.  chunk 0 is teacher-forced (covered
        # by the base chunk_loss).  Set ``chunk_rollout_chunks`` > 1 AND
        # ``chunk_rollout_loss_scale`` > 0 to enable.
        self.chunk_rollout_chunks = int(chunk_rollout_chunks)
        self.chunk_rollout_loss_scale = float(chunk_rollout_loss_scale)
        # Recompute each autoregressive step's activations in backward instead of
        # storing them — cuts the chunk rollout's activation memory from O(N*K) to
        # O(1). Opt-in; numerically identical to the plain path. See predict_next_chunk.
        self.grad_checkpoint = bool(grad_checkpoint)
        self.dim_head = int(dim_head)
        self.attn_impl = str(attn_impl)
        task_cfg = dict(task_conditioning or {})
        self.task_conditioning_enabled = bool(task_cfg.get("enabled", False))
        self.supports_task_conditioning = bool(self.task_conditioning_enabled)
        if self.task_conditioning_enabled:
            num_tasks = int(task_cfg.get("num_tasks", 0) or 0)
            embedding_dim = int(task_cfg.get("embedding_dim", 0) or 0)
            if num_tasks <= 0 or embedding_dim <= 0:
                raise ValueError(
                    "world_model.task_conditioning requires positive num_tasks and embedding_dim"
                )
            if embedding_dim != int(self.token_dim):
                raise ValueError(
                    "ChunkAwareWorldModel task_conditioning.embedding_dim must match "
                    f"token_dim ({embedding_dim} != {int(self.token_dim)})"
                )
            self.task_embedding = nn.Embedding(num_tasks, int(self.token_dim))
        else:
            self.task_embedding = None
        self.slots_per_step = self.token_count
        self.pos_context_len = self.num_hist
        self.obs_norm = nn.Identity()
        self.obs_proj = nn.Identity()
        self.action_proj = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.action_emb_dim),
        )
        if self.reward_enabled:
            self.reward_norm = nn.LayerNorm(self.obs_token_dim)
            self.reward_head = nn.Sequential(
                nn.Linear(self.obs_token_dim, self.reward_hidden_dim),
                nn.GELU(),
                nn.Linear(self.reward_hidden_dim, 1),
            )
            final = self.reward_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.constant_(final.bias, self.reward_init_logit)
        if self.success_return_enabled:
            self.success_return_norm = nn.LayerNorm(self.obs_token_dim)
            self.success_return_head = nn.Sequential(
                nn.Linear(self.obs_token_dim, self.success_return_hidden_dim),
                nn.GELU(),
                nn.Linear(self.success_return_hidden_dim, 1),
            )
            final = self.success_return_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.constant_(final.bias, self.success_return_init_logit)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.pos_context_len * self.slots_per_step, self.model_dim)
            * 0.02
        )
        self.predictor = _WMStyleTransformer(
            dim=self.model_dim,
            depth=int(kwargs.get("depth", 6) if len(args_list) <= 6 else args_list[6]),
            heads=int(kwargs.get("heads", 8) if len(args_list) <= 7 else args_list[7]),
            dim_head=self.dim_head,
            mlp_dim=int(kwargs.get("mlp_dim", 2048) if len(args_list) <= 8 else args_list[8]),
            dropout=float(kwargs.get("dropout", 0.1) if len(args_list) <= 9 else args_list[9]),
            attn_impl=self.attn_impl,
        )
        self.out_norm = nn.Identity()
        self.out_proj = nn.Identity()
        if self.freeze_input_embeddings_requested:
            self.freeze_input_embeddings()

    # ------------------------------------------------------------------ #
    # WM-style action concat transition                                  #
    # ------------------------------------------------------------------ #
    def _module_dtype(self) -> torch.dtype:
        return self.action_proj[-1].weight.dtype

    def _module_device(self) -> torch.device:
        return self.action_proj[-1].weight.device

    def _apply_task_conditioning(
        self,
        obs_tokens: torch.Tensor,
        task_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.task_conditioning_enabled:
            return obs_tokens
        if task_ids is None:
            raise ValueError(
                "task_ids are required when world model task conditioning is enabled"
            )
        if self.task_embedding is None:
            raise RuntimeError("task conditioning is enabled without an embedding")
        task_emb = self.task_embedding(task_ids.to(obs_tokens.device).long())
        while task_emb.ndim < obs_tokens.ndim:
            task_emb = task_emb.unsqueeze(1)
        return obs_tokens + task_emb.to(obs_tokens.dtype)

    def _obs_tokens_from_obs(
        self, obs: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        """Normalize raw vision tokens or expanded observation tokens."""
        obs_embedding = self._obs_embedding_from_obs(obs)
        if obs_embedding.ndim == 3 and obs_embedding.shape[1:] == (
            self.token_count,
            self.obs_token_dim,
        ):
            tokens = obs_embedding[:, None]
        elif obs_embedding.ndim == 4 and obs_embedding.shape[-2] == self.token_count:
            if obs_embedding.shape[-1] not in {self.token_dim, self.obs_token_dim}:
                raise ValueError(
                    "tokenized observation width mismatch: got "
                    f"{obs_embedding.shape[-1]}, expected {self.token_dim} "
                    f"or {self.obs_token_dim}"
                )
            tokens = obs_embedding
        else:
            tokens = self.obs_to_tokens(obs_embedding)
        if int(tokens.shape[1]) > self.max_seq_len:
            raise ValueError(
                f"sequence length {tokens.shape[1]} exceeds max_seq_len={self.max_seq_len}"
            )
        return tokens.to(device=self._module_device(), dtype=self._module_dtype())

    def _observation_tokens(
        self,
        vision_tokens: torch.Tensor,
        proprio_raw: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fold encoded proprio into every observation token channel."""
        if self.proprio_condition_dim == 0:
            return vision_tokens
        if proprio_raw is None:
            raise ValueError("proprio is required when proprio_emb_dim>0")
        if self.proprio_encoder is None:
            raise RuntimeError("proprio encoder is missing")
        proprio = proprio_raw.to(device=self._module_device(), dtype=self._module_dtype())
        emb = self.proprio_encoder(proprio)
        if self.num_proprio_repeat > 1:
            emb = emb.repeat(1, 1, self.num_proprio_repeat)
        tiled = emb[:, :, None, :].expand(-1, -1, vision_tokens.shape[2], -1)
        return torch.cat([vision_tokens, tiled], dim=-1)

    def _proprio_for_steps(
        self,
        proprio_raw: torch.Tensor | None,
        steps: int,
    ) -> torch.Tensor | None:
        if proprio_raw is None:
            return None
        proprio = proprio_raw.to(device=self._module_device(), dtype=self._module_dtype())
        if proprio.ndim == 2:
            return proprio[:, None].expand(-1, int(steps), -1)
        if proprio.ndim != 3:
            raise ValueError(
                f"proprio must be [B,P] or [B,T,P], got {tuple(proprio.shape)}"
            )
        if proprio.shape[1] == int(steps):
            return proprio
        if proprio.shape[1] > int(steps):
            return proprio[:, -int(steps) :]
        pad = proprio[:, :1].expand(-1, int(steps) - proprio.shape[1], -1)
        return torch.cat([pad, proprio], dim=1)

    def _raw_proprio_from_obs_tokens(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        """Decode predicted proprio tokens back to raw proprio for classifier scoring."""
        if self.proprio_condition_dim == 0:
            raise RuntimeError("raw proprio decoding requires proprio_emb_dim>0")
        if self.proprio_decoder is None:
            raise RuntimeError("proprio decoder is missing")
        if obs_tokens.shape[-1] < self.obs_token_dim:
            raise ValueError(
                f"obs token width {obs_tokens.shape[-1]} is smaller than obs_token_dim={self.obs_token_dim}"
            )
        proprio_emb = obs_tokens[..., self.token_dim : self.obs_token_dim].mean(dim=-2)
        return self.proprio_decoder(proprio_emb)

    def _condition_tokens(
        self,
        obs_tokens: torch.Tensor,
        lang_emb: torch.Tensor | None,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if actions is None:
            actions = lang_emb
            lang_emb = None
        if actions is None:
            raise ValueError("actions are required for WM conditioning")
        actions = self._validate_actions(actions, int(obs_tokens.shape[1]))
        parts = [obs_tokens]
        if self.lang_condition_dim > 0:
            if lang_emb is None:
                raise ValueError("lang_emb is required when lang_emb_dim>0")
            if self.lang_proj is None:
                raise RuntimeError("language projection is missing")
            lang = lang_emb.to(device=self._module_device(), dtype=self._module_dtype())
            le = self.lang_proj(lang)
            if self.num_lang_repeat > 1:
                le = le.repeat(1, self.num_lang_repeat)
            lang_tokens = le[:, None, None, :].expand(
                -1, obs_tokens.shape[1], obs_tokens.shape[2], -1
            )
            parts.append(lang_tokens)
        action_emb = self.action_proj(actions)
        if self.num_action_repeat > 1:
            action_emb = action_emb.repeat(1, 1, self.num_action_repeat)
        action_tokens = action_emb[:, :, None, :].expand(
            -1, -1, obs_tokens.shape[2], -1
        )
        parts.append(action_tokens)
        return torch.cat(parts, dim=-1)

    def observe_sequence(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Encode replay windows into per-step latent starts for imagination."""
        vision_tokens = self.obs_to_tokens(self._obs_embedding_from_obs(batch))
        vision_tokens = self._apply_task_conditioning(
            vision_tokens, batch.get("task_ids")
        )
        obs_tokens = self._observation_tokens(vision_tokens, batch.get("proprio"))
        bsz, steps = obs_tokens.shape[:2]
        actions = self._actions_or_zeros(batch.get("actions"), bsz, steps)

        histories: list[torch.Tensor] = []
        action_histories: list[torch.Tensor] = []
        for step in range(steps):
            indices = torch.arange(
                step - self.num_hist + 1,
                step + 1,
                device=obs_tokens.device,
            ).clamp_min(0)
            histories.append(obs_tokens.index_select(1, indices))
            action_histories.append(actions.index_select(1, indices))

        latent = {
            "hidden": obs_tokens,
            "history": torch.stack(histories, dim=1),
            "actions": torch.stack(action_histories, dim=1),
            "proprio": batch.get("proprio"),
            "lang": batch.get("lang_emb"),
        }
        return {"latent": latent}

    def encode(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        act: torch.Tensor,
        lang: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs_tokens = self._obs_tokens_from_obs(obs)
        proprio = obs.get("proprio") if isinstance(obs, dict) else None
        if obs_tokens.shape[-1] == self.token_dim and self.proprio_condition_dim > 0:
            obs_tokens = self._observation_tokens(
                obs_tokens,
                self._proprio_for_steps(proprio, int(obs_tokens.shape[1])),
            )
        z = self._condition_tokens(obs_tokens, lang, act)
        bsz, steps, slots, dim = z.shape
        flat = z.reshape(bsz, steps * slots, dim)
        if flat.shape[1] > self.pos_embedding.shape[1]:
            raise ValueError(
                "ChunkAwareWorldModel WM predictor is configured for "
                f"num_hist={self.pos_context_len} frames; got {steps} frames"
            )
        flat = flat + self.pos_embedding[:, : flat.shape[1]].to(
            device=flat.device, dtype=flat.dtype
        )
        return flat.reshape(bsz, steps, slots, dim)

    def separate_emb(
        self,
        z: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        visual = z[..., : self.token_dim]
        proprio = z[..., self.token_dim : self.obs_token_dim]
        cond_emb = z[..., self.obs_token_dim :].mean(dim=2)
        return {"visual": visual, "proprio": proprio}, cond_emb

    def replace_actions_from_z(
        self,
        z: torch.Tensor,
        act: torch.Tensor,
        lang: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs_tokens = z[..., : self.obs_token_dim]
        return self._condition_tokens(obs_tokens, lang, act)

    def actor_input(
        self, latent: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        """Return the visual token segment consumed by VLA action actors."""
        hidden = self._latent_hidden(latent)
        return hidden[..., : self.token_dim]

    def critic_input(
        self, latent: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        """Mean-pool visual tokens for critics while retaining proprio for WM losses."""
        hidden = self._latent_hidden(latent)
        tokens = self._obs_tokens_from_obs(hidden)[..., : self.token_dim]
        pooled = tokens.mean(dim=2)
        if tokens.shape[1] == 1:
            return pooled[:, 0]
        return pooled

    def _latent_lang(
        self, latent: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor | None:
        if isinstance(latent, dict) and isinstance(latent.get("lang"), torch.Tensor):
            return latent["lang"]
        return None

    def _latent_proprio(
        self, latent: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor | None:
        if isinstance(latent, dict) and isinstance(latent.get("proprio"), torch.Tensor):
            return latent["proprio"]
        return None

    def _latent_hidden(
        self, latent: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(latent, torch.Tensor):
            return self._obs_tokens_from_obs(latent)[:, -1]
        hidden = latent.get("hidden") if isinstance(latent, dict) else None
        if isinstance(hidden, torch.Tensor):
            return self._obs_tokens_from_obs(hidden)[:, -1]
        history = latent.get("history") if isinstance(latent, dict) else None
        if isinstance(history, torch.Tensor):
            if history.ndim == 5:
                history = history[:, -1]
            if history.ndim == 4 and history.shape[-1] == self.obs_dim:
                history = history[:, -1]
            return self._obs_tokens_from_obs(history)[:, -1]
        raise KeyError("VLA latent must contain `hidden` or `history`.")

    def _latent_history(
        self, latent: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(latent, dict) and isinstance(latent.get("history"), torch.Tensor):
            history = latent["history"]
            if history.ndim == 5:
                history = history[:, -1]
            elif history.ndim == 4 and history.shape[-1] == self.obs_dim:
                history = history[:, -1]
            tokens = self._obs_tokens_from_obs(history)
        else:
            tokens = self._obs_tokens_from_obs(self._latent_hidden(latent))
        if tokens.shape[1] >= self.num_hist:
            return tokens[:, -self.num_hist :]
        pad = tokens[:, :1].expand(-1, self.num_hist - tokens.shape[1], -1, -1)
        return torch.cat([pad, tokens], dim=1)

    def predict_next(
        self,
        latent: dict[str, torch.Tensor] | torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history = self._latent_history(latent)
        bsz = int(history.shape[0])
        lang = self._latent_lang(latent)
        proprio = self._latent_proprio(latent)
        action = actions[:, 0] if actions.ndim == 3 else actions
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Dreamer action must be [B,{self.action_dim}], got {tuple(actions.shape)}"
            )

        action_history = self._latent_actions(latent, bsz).clone()
        action_history[:, -1] = action.to(
            device=action_history.device, dtype=action_history.dtype
        )
        model_history = history
        if (
            proprio is not None
            and model_history.shape[-1] == self.token_dim
            and self.proprio_condition_dim > 0
        ):
            model_history = self._observation_tokens(
                model_history,
                self._proprio_for_steps(proprio, int(model_history.shape[1])),
            )
        z = self.encode(model_history, action_history, lang)
        pred_z = self.predict(z)
        next_hidden = pred_z[:, -1][..., : self.obs_token_dim]
        next_proprio = (
            self._raw_proprio_from_obs_tokens(next_hidden)
            if self.proprio_condition_dim > 0
            else None
        )

        if self.num_hist > 1:
            next_history = torch.cat(
                [model_history[:, 1:], next_hidden[:, None]],
                dim=1,
            )
            next_action_history = torch.cat(
                [
                    action_history[:, 1:],
                    action_history.new_zeros(bsz, 1, self.action_dim),
                ],
                dim=1,
            )
        else:
            next_history = next_hidden[:, None]
            next_action_history = action_history.new_zeros(bsz, 1, self.action_dim)
        out = {
            "hidden": next_hidden,
            "history": next_history,
            "actions": next_action_history,
            "lang": lang,
        }
        if next_proprio is not None:
            out["proprio"] = next_proprio
        return out

    def _predict_next_step(
        self,
        cur: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """One autoregressive step, optionally gradient-checkpointed.

        When ``grad_checkpoint`` is on (and we are building a graph), the step's
        activations are recomputed in backward instead of stored. Numerically
        identical to the plain path; ``use_reentrant=False`` preserves RNG so
        dropout matches on recompute.
        """
        if not (self.grad_checkpoint and self.training and torch.is_grad_enabled()):
            return self.predict_next(cur, action)

        lang = cur.get("lang")

        proprio = cur.get("proprio")
        if not isinstance(proprio, torch.Tensor):
            proprio = cur["hidden"].new_zeros(cur["hidden"].shape[0], 0)

        def _fn(hidden, history, actions, proprio_raw, act):
            latent = {
                "hidden": hidden,
                "history": history,
                "actions": actions,
                "lang": lang,
            }
            if proprio_raw.shape[-1] > 0:
                latent["proprio"] = proprio_raw
            out = self.predict_next(latent, act)
            out_proprio = out.get("proprio")
            if not isinstance(out_proprio, torch.Tensor):
                out_proprio = proprio_raw
            return out["hidden"], out["history"], out["actions"], out_proprio

        hidden, history, actions, proprio_out = checkpoint(
            _fn,
            cur["hidden"],
            cur["history"],
            cur["actions"],
            proprio,
            action,
            use_reentrant=False,
        )
        out = {"hidden": hidden, "history": history, "actions": actions, "lang": lang}
        if proprio_out.shape[-1] > 0:
            out["proprio"] = proprio_out
        return out

    def predict_next_chunk(
        self,
        latent: dict[str, torch.Tensor] | torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Advance the WM by ``chunk_size`` env-steps autoregressively.

        Args:
            latent: dict with ``history`` [B,H,N,token_dim], ``actions`` [B,H,A],
                and tokenized ``hidden`` [B,N,token_dim].
            action_chunk: [B, K, A] where K == ``self.chunk_size``.

        Returns dict with:
            ``hidden``     [B, N, token_dim]    — last predicted frame h_K.
            ``hidden_seq`` [B, K, N, token_dim] — all K predicted frames h_1..h_K.
            ``history``    [B, H, N, token_dim] — rolled history after K steps.
            ``actions``    [B, H, A]            — rolled action history, last slot zero.
        """
        if action_chunk.ndim != 3 or action_chunk.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_chunk must be [B,K,{self.action_dim}], got {tuple(action_chunk.shape)}"
            )
        if action_chunk.shape[1] != self.chunk_size:
            raise ValueError(
                f"action_chunk time dim {action_chunk.shape[1]} != chunk_size {self.chunk_size}"
            )
        K = self.chunk_size
        bsz = int(action_chunk.shape[0])

        history = self._latent_history(latent)
        action_history = self._latent_actions(latent, bsz).clone()
        lang = self._latent_lang(latent)

        device = self._module_device()
        dtype = self._module_dtype()
        action_chunk_v = action_chunk.to(device=device, dtype=dtype)
        cur: dict[str, torch.Tensor] = {
            "hidden": history[:, -1],
            "history": history,
            "actions": action_history,
            "lang": lang,
        }
        if isinstance(latent, dict) and isinstance(latent.get("proprio"), torch.Tensor):
            cur["proprio"] = latent["proprio"].to(device=device, dtype=dtype)
        preds: list[torch.Tensor] = []
        proprio_preds: list[torch.Tensor] = []
        for step in range(K):
            cur = self._predict_next_step(cur, action_chunk_v[:, step])
            preds.append(cur["hidden"])
            if isinstance(cur.get("proprio"), torch.Tensor):
                proprio_preds.append(cur["proprio"])
        hidden_seq = torch.stack(preds, dim=1)

        out = {
            "hidden": cur["hidden"],
            "hidden_seq": hidden_seq,
            "history": cur["history"],
            "actions": cur["actions"],
            "lang": cur.get("lang"),
        }
        if proprio_preds:
            out["proprio"] = cur["proprio"]
            out["proprio_seq"] = torch.stack(proprio_preds, dim=1)
        return out

    # ------------------------------------------------------------------ #
    # Chunk-objective training loss                                      #
    # ------------------------------------------------------------------ #
    def chunk_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Train end-to-end as a K-step chunk predictor.

        Per sampled window of length T >= H + K:
          - first H frames                  → h_history
          - actions[:, H-1 : H-1+K]         → chunk a_0..a_{K-1}
          - obs[:, H : H + K]               → targets h_1..h_K

        Reward / success-return heads (if enabled) are trained on TARGET
        hiddens, matching parent's convention.
        """
        obs = batch["obs_embedding"]
        actions = batch.get("current_actions")
        if not isinstance(actions, torch.Tensor):
            actions = batch["actions"]
        H = self.num_hist
        K = self.chunk_size
        vision_tokens = self.obs_to_tokens(obs)
        vision_tokens = self._apply_task_conditioning(
            vision_tokens, batch.get("task_ids")
        )
        obs_tokens = self._observation_tokens(vision_tokens, batch.get("proprio"))
        lang_emb = batch.get("lang_emb")
        T = int(obs_tokens.shape[1])
        if T < H + K:
            raise ValueError(f"chunk_loss requires T >= H+K = {H + K}, got T={T}")
        if actions.shape[1] < H - 1 + K:
            raise ValueError(
                "action sequence length "
                f"{actions.shape[1]} too short for chunk loss "
                f"(need >= {H - 1 + K})"
            )

        actions = self._validate_actions(actions, int(actions.shape[1]))
        bsz = int(obs_tokens.shape[0])

        history = obs_tokens[:, :H]
        chunk_actions = actions[:, H - 1 : H - 1 + K]
        hidden_target = obs_tokens[:, H : H + K].detach()

        action_history = torch.zeros(
            bsz,
            H,
            self.action_dim,
            device=self._module_device(),
            dtype=self._module_dtype(),
        )
        if H > 1:
            action_history[:, : H - 1] = actions[:, : H - 1]
        action_history[:, -1] = chunk_actions[:, 0]

        latent = {
            "hidden": history[:, -1],
            "history": history,
            "actions": action_history,
            "lang": lang_emb,
        }
        if isinstance(batch.get("proprio"), torch.Tensor):
            latent["proprio"] = batch["proprio"][:, H - 1]
        out = self.predict_next_chunk(latent, chunk_actions)
        hidden_pred = out["hidden_seq"]
        self._last_hidden_target_width = int(hidden_target.shape[-1])

        loss, hidden_mse, hidden_cosine = self._hidden_loss_terms(
            hidden_pred, hidden_target
        )
        proprio_out: dict[str, torch.Tensor] = {}
        if self.proprio_condition_dim > 0 and isinstance(out.get("proprio_seq"), torch.Tensor):
            proprio_target = batch.get("proprio")
            if not isinstance(proprio_target, torch.Tensor):
                if self.proprio_reconstruction_loss_scale > 0:
                    raise KeyError(
                        "proprio_reconstruction_loss_scale > 0 requires batch['proprio']"
                    )
            else:
                target = proprio_target[:, H : H + K].to(
                    device=out["proprio_seq"].device,
                    dtype=out["proprio_seq"].dtype,
                )
                proprio_loss = F.mse_loss(out["proprio_seq"], target)
                if self.proprio_reconstruction_loss_scale > 0:
                    loss = loss + self.proprio_reconstruction_loss_scale * proprio_loss
                proprio_out = {
                    "proprio_reconstruction_loss": proprio_loss.detach(),
                    "proprio_pred_norm": out["proprio_seq"].detach().float().norm(dim=-1).mean(),
                    "proprio_target_norm": target.detach().float().norm(dim=-1).mean(),
                }

        # --- Close-loop multi-chunk rollout loss (anti-drift) ---
        # Continue rolling forward N-1 MORE chunks from chunk 0's output, feeding
        # the predicted hidden as next chunk's history (no teacher forcing).
        # Actions are still REAL demo chunk actions throughout — this loss only
        # cures WM-internal drift, not actor sensitivity to drift.
        rollout_out: dict[str, torch.Tensor] = {}
        if self.chunk_rollout_chunks > 1 and self.chunk_rollout_loss_scale > 0.0:
            N = self.chunk_rollout_chunks
            if T < H + N * K:
                raise ValueError(
                    f"chunk_rollout_chunks={N} requires T >= H + N*K = {H + N * K}, got T={T}"
                )
            cur_latent = {
                "hidden": out["hidden"],
                "history": out["history"],
                "actions": out["actions"],
                "lang": lang_emb,
            }
            if isinstance(out.get("proprio"), torch.Tensor):
                cur_latent["proprio"] = out["proprio"]
            rollout_preds: list[torch.Tensor] = []
            for c in range(1, N):
                cca = actions[:, H - 1 + c * K : H - 1 + (c + 1) * K]
                out_c = self.predict_next_chunk(cur_latent, cca)
                rollout_preds.append(out_c["hidden_seq"])
                cur_latent = {
                    "hidden": out_c["hidden"],
                    "history": out_c["history"],
                    "actions": out_c["actions"],
                    "lang": lang_emb,
                }
                if isinstance(out_c.get("proprio"), torch.Tensor):
                    cur_latent["proprio"] = out_c["proprio"]
            rollout_pred = torch.cat(rollout_preds, dim=1)
            rollout_target = obs_tokens[:, H + K : H + N * K].detach()
            rollout_loss_total, rollout_mse, rollout_cosine = self._hidden_loss_terms(
                rollout_pred, rollout_target
            )
            loss = loss + self.chunk_rollout_loss_scale * rollout_loss_total
            rollout_out = {
                "rollout_loss": rollout_loss_total.detach(),
                "rollout_mse": rollout_mse.detach(),
                "rollout_cosine_loss": rollout_cosine.detach(),
                "rollout_chunks": loss.new_tensor(float(N)),
            }

        reward_out: dict[str, torch.Tensor] = {}
        if self.reward_loss_scale > 0.0:
            rewards = batch.get("rewards")
            if rewards is None:
                raise KeyError("reward_loss_scale > 0 requires batch['rewards']")
            rewards_chunk = self._slice_per_frame_signal(rewards, T)
            reward_out = self._reward_loss_terms(hidden_target, rewards_chunk)
            loss = loss + self.reward_loss_scale * reward_out["reward_loss"]

        success_return_out: dict[str, torch.Tensor] = {}
        if self.success_return_loss_scale > 0.0:
            success_to_go = (
                batch.get("success_to_go")
                if batch.get("success_to_go") is not None
                else batch.get("return_to_go", batch.get("return_targets"))
            )
            if success_to_go is None:
                raise KeyError(
                    "success_return_loss_scale > 0 requires batch['success_to_go']"
                )
            success_chunk = self._slice_per_frame_signal(success_to_go, T)
            success_return_out = self._success_return_loss_terms(
                hidden_target, success_chunk
            )
            loss = (
                loss
                + self.success_return_loss_scale
                * success_return_out["success_return_loss"]
            )

        zero = loss.new_zeros(())
        out_dict: dict[str, torch.Tensor] = {
            "_loss": loss,
            "loss": loss.detach(),
            "next_latent_loss": hidden_mse.detach(),
            "next_latent_mse": hidden_mse.detach(),
            "next_latent_cosine_loss": hidden_cosine.detach(),
            "hidden_loss": hidden_mse.detach(),
            "hidden_mse": hidden_mse.detach(),
            "hidden_cosine_loss": hidden_cosine.detach(),
            "hidden_pred_norm": hidden_pred.detach().float().norm(dim=-1).mean(),
            "hidden_target_norm": hidden_target.detach().float().norm(dim=-1).mean(),
            "chunk_size": loss.new_tensor(float(self.chunk_size)),
            "rec_loss": zero.detach(),
            "dyn_loss": zero.detach(),
            "rep_loss": zero.detach(),
            "image_mse": zero.detach(),
            "image_psnr": zero.detach(),
        }
        if rollout_out:
            out_dict.update(rollout_out)
        if proprio_out:
            out_dict.update(proprio_out)
        if reward_out:
            out_dict.update(
                {
                    "reward_loss": reward_out["reward_loss"].detach(),
                    "reward_pred_mean": reward_out["reward_pred_mean"],
                    "reward_target_mean": reward_out["reward_target_mean"],
                }
            )
            if "reward_binary_acc" in reward_out:
                out_dict["reward_binary_acc"] = reward_out["reward_binary_acc"]
            if "reward_mae" in reward_out:
                out_dict["reward_mae"] = reward_out["reward_mae"]
        if success_return_out:
            out_dict.update(
                {
                    "success_return_loss": success_return_out[
                        "success_return_loss"
                    ].detach(),
                    "success_return_pred_mean": success_return_out[
                        "success_return_pred_mean"
                    ],
                    "success_return_target_mean": success_return_out[
                        "success_return_target_mean"
                    ],
                    "success_return_mse": success_return_out["success_return_mse"],
                }
            )
        return out_dict

    def reward_logits(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        self._require_reward_head()
        tokens = self._obs_tokens_from_obs(obs_embedding)
        pooled = self.reward_norm(tokens).mean(dim=2)
        return self.reward_head(pooled).squeeze(-1)

    def success_return_logits(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        self._require_success_return_head()
        tokens = self._obs_tokens_from_obs(obs_embedding)
        pooled = self.success_return_norm(tokens).mean(dim=2)
        return self.success_return_head(pooled).squeeze(-1)

    def _slice_per_frame_signal(self, signal: torch.Tensor, T: int) -> torch.Tensor:
        """Slice a per-frame signal (rewards / success_to_go) to the K target frames h_1..h_K.

        Accepts [B], [B,T], [B,T,1], [B,K], or [B,K+1] layouts. Returns [B,K].
        """
        H = self.num_hist
        K = self.chunk_size
        if signal.ndim == 1:
            signal = signal[:, None]
        if signal.ndim == 3 and signal.shape[-1] == 1:
            signal = signal.squeeze(-1)
        if signal.ndim != 2:
            raise ValueError(
                f"per-frame signal must be [B,T] or [B,T,1], got {tuple(signal.shape)}"
            )
        if signal.shape[1] == T:
            return signal[:, H : H + K]
        if signal.shape[1] == K:
            return signal
        if signal.shape[1] == K + 1:
            return signal[:, 1:]
        raise ValueError(
            f"per-frame signal length {signal.shape[1]} not aligned with T={T}, K={K}, K+1={K + 1}"
        )

    # ------------------------------------------------------------------ #
    # Routing                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _latent_with_batch_sidecars(
        latent: dict[str, torch.Tensor] | torch.Tensor,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if "lang_emb" not in batch and "proprio" not in batch:
            return latent
        if isinstance(latent, dict):
            enriched = dict(latent)
        else:
            enriched = {"hidden": latent}
        if "lang_emb" in batch and "lang" not in enriched:
            enriched["lang"] = batch["lang_emb"]
        if "proprio" in batch and "proprio" not in enriched:
            enriched["proprio"] = batch["proprio"]
        return enriched

    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return self.chunk_loss(batch)

    def forward(self, batch: dict[str, Any]) -> Any:  # type: ignore[override]
        if isinstance(batch, dict) and batch.get("mode") == "predict_next_chunk":
            latent = self._latent_with_batch_sidecars(batch["latent"], batch)
            return self.predict_next_chunk(latent, batch["actions"])
        if isinstance(batch, dict) and batch.get("mode") == "predict_next":
            latent = self._latent_with_batch_sidecars(batch["latent"], batch)
            return self.predict_next(latent, batch["actions"])
        if isinstance(batch, dict) and batch.get("mode") == "classifier_input":
            return self.actor_input(batch["latent"])
        if isinstance(batch, dict) and batch.get("mode") == "chunk_loss":
            return self.chunk_loss(batch)
        return super().forward(batch)
