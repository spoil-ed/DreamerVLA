from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamervla.models.world_model.dino_wm import DinoWMWorldModel


class _DinoStyleFeedForward(nn.Module):
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


class _DinoStyleAttention(nn.Module):
    """DINO-WM-style attention with residual dim independent of QKV inner dim."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if dim_head < 1:
            raise ValueError(f"dim_head must be >= 1, got {dim_head}")
        self.heads = int(heads)
        self.dim_head = int(dim_head)
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
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            dots = dots + mask.to(device=dots.device, dtype=dots.dtype)[None, None]
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.to_out(out)


class _DinoStyleTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _DinoStyleAttention(
                            dim,
                            heads=int(heads),
                            dim_head=int(dim_head),
                            dropout=float(dropout),
                        ),
                        _DinoStyleFeedForward(
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


class ChunkAwareDinoWMWorldModel(DinoWMWorldModel):
    """Chunk WM over original VLA hidden tokens with DINO-WM-style conditioning.

    The transition model keeps each observation token in source token space and
    concatenates an encoded action to every observation token channel, matching
    the default DINO-WM ``concat_dim=1`` pattern.  A chunk is rolled out
    autoregressively: every step predicts ``e_{t+1}`` from the latest
    ``num_hist`` latent frames conditioned on the current action, then slides
    the predicted observation tokens into the next history.
    """

    def __init__(
        self,
        *args: Any,
        chunk_size: int = 5,
        mask_init_scale: float = 0.02,
        chunk_rollout_chunks: int = 1,
        chunk_rollout_loss_scale: float = 0.0,
        action_emb_dim: int = 10,
        num_action_repeat: int = 1,
        dim_head: int = 64,
        **kwargs: Any,
    ) -> None:
        args_list = list(args)
        requested_model_dim = (
            args_list[5] if len(args_list) > 5 else kwargs.get("model_dim")
        )
        token_dim_hint = int(args_list[3] if len(args_list) > 3 else kwargs.get("token_dim", 1024))
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
        del mask_init_scale  # Kept only for old config/checkpoint compatibility.
        self.action_emb_dim = int(action_emb_dim)
        self.num_action_repeat = int(num_action_repeat)
        self.action_condition_dim = self.action_emb_dim * self.num_action_repeat
        if self.action_emb_dim < 1:
            raise ValueError(f"action_emb_dim must be >= 1, got {action_emb_dim}")
        if self.num_action_repeat < 1:
            raise ValueError(
                f"num_action_repeat must be >= 1, got {num_action_repeat}"
            )
        expected_model_dim = self.token_dim + self.action_condition_dim
        if requested_model_dim is None:
            requested_model_dim = expected_model_dim
        self.model_dim = int(requested_model_dim)
        if self.model_dim != expected_model_dim:
            raise ValueError(
                "ChunkAwareDinoWMWorldModel uses DINO-WM concat conditioning; "
                "set model_dim == token_dim + action_emb_dim * num_action_repeat, "
                f"got model_dim={self.model_dim}, token_dim={self.token_dim}, "
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
        self.dim_head = int(dim_head)
        self.slots_per_step = self.token_count
        self.pos_context_len = self.num_hist
        self.obs_norm = nn.Identity()
        self.obs_proj = nn.Identity()
        self.action_proj = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.action_emb_dim),
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.pos_context_len * self.slots_per_step, self.model_dim)
            * 0.02
        )
        self.predictor = _DinoStyleTransformer(
            dim=self.model_dim,
            depth=int(kwargs.get("depth", 6) if len(args_list) <= 6 else args_list[6]),
            heads=int(kwargs.get("heads", 8) if len(args_list) <= 7 else args_list[7]),
            dim_head=self.dim_head,
            mlp_dim=int(kwargs.get("mlp_dim", 2048) if len(args_list) <= 8 else args_list[8]),
            dropout=float(kwargs.get("dropout", 0.1) if len(args_list) <= 9 else args_list[9]),
        )
        self.out_norm = nn.Identity()
        self.out_proj = nn.Identity()
        if self.freeze_input_embeddings_requested:
            self.freeze_input_embeddings()

    # ------------------------------------------------------------------ #
    # DINO-WM-style action concat transition                             #
    # ------------------------------------------------------------------ #
    def _module_dtype(self) -> torch.dtype:
        return self.action_proj[-1].weight.dtype

    def _module_device(self) -> torch.device:
        return self.action_proj[-1].weight.device

    def _condition_tokens(
        self,
        obs_tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        obs_tokens = self.obs_to_tokens(obs_tokens)
        actions = self._validate_actions(actions, int(obs_tokens.shape[1]))
        action_emb = self.action_proj(actions)
        if self.num_action_repeat > 1:
            action_emb = action_emb.repeat(1, 1, self.num_action_repeat)
        action_tokens = action_emb[:, :, None, :].expand(
            -1, -1, self.token_count, -1
        )
        return torch.cat([obs_tokens, action_tokens], dim=-1)

    def encode(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor:
        obs_tokens = self.obs_to_tokens(self._obs_embedding_from_obs(obs))
        z = self._condition_tokens(obs_tokens, act)
        bsz, steps, slots, dim = z.shape
        flat = z.reshape(bsz, steps * slots, dim)
        if flat.shape[1] > self.pos_embedding.shape[1]:
            raise ValueError(
                "ChunkAwareDinoWMWorldModel DINO-WM predictor is configured for "
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
        act_emb = z[..., self.token_dim :].mean(dim=2)
        proprio = visual.new_zeros(visual.shape[0], visual.shape[1], 0)
        return {"visual": visual, "proprio": proprio}, act_emb

    def replace_actions_from_z(
        self,
        z: torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor:
        obs_tokens = z[..., : self.token_dim]
        return self._condition_tokens(obs_tokens, act)

    def predict_next(
        self,
        latent: dict[str, torch.Tensor] | torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history = self._latent_history(latent)
        bsz = int(history.shape[0])
        action = actions[:, 0] if actions.ndim == 3 else actions
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Dreamer action must be [B,{self.action_dim}], got {tuple(actions.shape)}"
            )

        action_history = self._latent_actions(latent, bsz).clone()
        action_history[:, -1] = action.to(
            device=action_history.device, dtype=action_history.dtype
        )
        z = self.encode(history, action_history)
        pred_z = self.predict(z)
        pred_obs, _ = self.separate_emb(pred_z[:, -1:])
        next_hidden = pred_obs["visual"][:, -1]

        if self.num_hist > 1:
            next_history = torch.cat([history[:, 1:], next_hidden[:, None]], dim=1)
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
        return {
            "hidden": next_hidden,
            "history": next_history,
            "actions": next_action_history,
        }

    def predict_next_chunk(
        self,
        latent: dict[str, torch.Tensor] | torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Advance the WM by ``chunk_size`` env-steps autoregressively.

        Args:
            latent: dict with ``history`` [B,H,N,token_dim], ``actions`` [B,H,A],
                ``hidden`` [B,N,token_dim]; flat legacy hidden tensors are also
                accepted and normalized at the boundary.
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

        device = self._module_device()
        dtype = self._module_dtype()
        action_chunk_v = action_chunk.to(device=device, dtype=dtype)
        cur: dict[str, torch.Tensor] = {
            "hidden": history[:, -1],
            "history": history,
            "actions": action_history,
        }
        preds: list[torch.Tensor] = []
        for step in range(K):
            cur = self.predict_next(cur, action_chunk_v[:, step])
            preds.append(cur["hidden"])
        hidden_seq = torch.stack(preds, dim=1)

        return {
            "hidden": cur["hidden"],
            "hidden_seq": hidden_seq,
            "history": cur["history"],
            "actions": cur["actions"],
        }

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
        actions = batch["actions"]
        H = self.num_hist
        K = self.chunk_size
        obs_tokens = self.obs_to_tokens(obs)
        T = int(obs_tokens.shape[1])
        if T < H + K:
            raise ValueError(f"chunk_loss requires T >= H+K = {H + K}, got T={T}")
        if actions.shape[1] < H - 1 + K:
            raise ValueError(
                f"actions length {actions.shape[1]} too short for chunk loss (need >= {H - 1 + K})"
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
        }
        out = self.predict_next_chunk(latent, chunk_actions)
        hidden_pred = out["hidden_seq"]

        loss, hidden_mse, hidden_cosine = self._hidden_loss_terms(
            hidden_pred, hidden_target
        )

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
            }
            rollout_preds: list[torch.Tensor] = []
            for c in range(1, N):
                cca = actions[:, H - 1 + c * K : H - 1 + (c + 1) * K]
                out_c = self.predict_next_chunk(cur_latent, cca)
                rollout_preds.append(out_c["hidden_seq"])
                cur_latent = {
                    "hidden": out_c["hidden"],
                    "history": out_c["history"],
                    "actions": out_c["actions"],
                }
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
    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return self.chunk_loss(batch)

    def forward(self, batch: dict[str, Any]) -> Any:  # type: ignore[override]
        if isinstance(batch, dict) and batch.get("mode") == "predict_next_chunk":
            return self.predict_next_chunk(batch["latent"], batch["actions"])
        if isinstance(batch, dict) and batch.get("mode") == "chunk_loss":
            return self.chunk_loss(batch)
        return super().forward(batch)

    # ------------------------------------------------------------------ #
    # Ckpt loading                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_rynn_dino_wm_ckpt(
        cls,
        ckpt_path: str | Path,
        chunk_size: int = 5,
        device: str | torch.device = "cpu",
        strict: bool = False,
    ) -> ChunkAwareDinoWMWorldModel:
        """Load a chunk WM checkpoint using config stored in the checkpoint.

        This helper expects a checkpoint whose config matches the current
        DINO-WM-style concat-action architecture.  Older projection or
        mask-token chunk checkpoints are not shape-compatible.
        """
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if not isinstance(sd, dict) or "model" not in sd:
            raise ValueError(
                f"ckpt {ckpt_path} does not have the expected 'model' key; "
                f"top-level keys = {list(sd) if isinstance(sd, dict) else type(sd)}"
            )
        cfg_blob = sd.get("cfg", {})
        wm_cfg = (
            cfg_blob.get("world_model", cfg_blob.get("model", {}))
            if hasattr(cfg_blob, "get")
            else {}
        )
        if not wm_cfg:
            raise ValueError(
                f"ckpt {ckpt_path} cfg blob missing 'world_model' section; "
                f"cannot reconstruct architecture"
            )
        kwargs: dict[str, Any] = {}
        for key in (
            "obs_dim",
            "action_dim",
            "token_count",
            "token_dim",
            "model_dim",
            "latent_stage",
            "latent_source",
            "action_emb_dim",
            "num_action_repeat",
            "dim_head",
            "depth",
            "heads",
            "mlp_dim",
            "dropout",
            "num_hist",
            "num_pred",
            "max_seq_len",
            "hidden_loss_scale",
            "cosine_loss_scale",
            "rollout_loss_scale",
            "rollout_horizon",
            "rollout_context",
            "reward_head_type",
            "reward_loss_scale",
            "reward_hidden_dim",
            "reward_init_logit",
            "reward_pos_weight",
            "return_predictions",
        ):
            if key in wm_cfg:
                kwargs[key] = wm_cfg[key]
        wm = cls(chunk_size=chunk_size, **kwargs)
        missing, unexpected = wm.load_state_dict(sd["model"], strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"strict load_state_dict failed: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        return wm.to(device)
