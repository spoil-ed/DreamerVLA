from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dreamer_vla.models.world_model.rynn_dino_wm import RynnDinoWMWorldModel


class ChunkAwareRynnDinoWMWorldModel(RynnDinoWMWorldModel):
    """RynnDinoWM variant: a K-step action chunk is consumed in ONE forward pass.

    True chunk-as-one-input dynamics (Plan B): given starting hidden h_0 (with
    optional H-step history) and a K-step action chunk [a_0..a_{K-1}], predict
    [h_1..h_K] in a SINGLE predictor call.  Future obs slots inside the chunk
    are filled by a learned ``mask_obs_token`` that lives in model-space, so
    the model is trained to treat them as queries rather than as real frames.

    This matches the chunk-level latent dynamics used by diffusion-policy
    control horizons and VLA chunk action generation; it is NOT K stacked
    autoregressive ``predict_next`` calls (the previous loop-based variant).

    The K dimension is locked to the host pi0 actor's ``time_horizon`` and is
    validated at construction and at every call.
    """

    def __init__(
        self,
        *args: Any,
        chunk_size: int = 5,
        mask_init_scale: float = 0.02,
        chunk_rollout_chunks: int = 1,
        chunk_rollout_loss_scale: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
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
        # Lives in model-space → bypasses obs_norm / obs_proj so it stays a clean query token.
        self.mask_obs_token = nn.Parameter(
            torch.randn(self.token_count, self.model_dim) * float(mask_init_scale)
        )

    # ------------------------------------------------------------------ #
    # Chunk-as-one-input forward                                         #
    # ------------------------------------------------------------------ #
    def _chunk_forward_z(
        self,
        history: torch.Tensor,
        action_history: torch.Tensor,
        future_chunk_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Build the ``H+K-1``-step input tensor and run the predictor exactly once.

        Layout (per step = ``token_count`` obs tokens + 1 action token):
          step 0  ..  H-1 : real history obs tokens + (a_{-H+1}..a_{-1}, a_0)
          step H  ..  H+K-2: ``mask_obs_token`` placeholders + (a_1..a_{K-1})

        With block-causal mask, prediction at step ``t`` reads h_t and all earlier
        slots, so step ``H-1`` predicts h_1, step ``H+K-2`` predicts h_K.

        Returns ``pred_z`` of shape ``[B, H+K-1, slots_per_step, model_dim]``.
        """
        bsz = int(history.shape[0])
        H = self.num_hist
        K = self.chunk_size
        total_steps = H + K - 1
        device = self._module_device()
        dtype = self._module_dtype()

        history = history.to(device=device, dtype=dtype)
        action_history = action_history.to(device=device, dtype=dtype)

        history_tokens = self.obs_to_tokens(history)
        obs_emb_real = self.obs_proj(self.obs_norm(history_tokens))

        if K > 1:
            future_chunk_actions = future_chunk_actions.to(device=device, dtype=dtype)
            mask = self.mask_obs_token.to(device=device, dtype=dtype)
            mask = mask.view(1, 1, self.token_count, self.model_dim).expand(
                bsz, K - 1, -1, -1
            )
            obs_emb = torch.cat([obs_emb_real, mask], dim=1)
            all_actions = torch.cat([action_history, future_chunk_actions], dim=1)
        else:
            obs_emb = obs_emb_real
            all_actions = action_history

        act_emb = self.action_proj(all_actions)
        z = torch.cat([obs_emb, act_emb.unsqueeze(2)], dim=2)
        z_flat = z.reshape(bsz, total_steps * self.slots_per_step, self.model_dim)
        z_flat = z_flat + self.pos_embedding[:, : z_flat.shape[1]]
        z = z_flat.reshape(bsz, total_steps, self.slots_per_step, self.model_dim)
        return self.predict(z)

    def predict_next_chunk(
        self,
        latent: dict[str, torch.Tensor] | torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Advance the WM by ``chunk_size`` env-steps in ONE transformer call.

        Args:
            latent: dict with ``history`` [B,H,obs_dim], ``actions`` [B,H,A],
                ``hidden`` [B,obs_dim]; or a raw hidden tensor.
            action_chunk: [B, K, A] where K == ``self.chunk_size``.

        Returns dict with:
            ``hidden``     [B, obs_dim]    — last predicted frame h_K.
            ``hidden_seq`` [B, K, obs_dim] — all K predicted frames h_1..h_K.
            ``history``    [B, H, obs_dim] — rolled history after K steps.
            ``actions``    [B, H, A]       — rolled action history, last slot zero.
        """
        if action_chunk.ndim != 3 or action_chunk.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_chunk must be [B,K,{self.action_dim}], got {tuple(action_chunk.shape)}"
            )
        if action_chunk.shape[1] != self.chunk_size:
            raise ValueError(
                f"action_chunk time dim {action_chunk.shape[1]} != chunk_size {self.chunk_size}"
            )
        H = self.num_hist
        K = self.chunk_size
        bsz = int(action_chunk.shape[0])

        history = self._latent_history(latent)
        action_history = self._latent_actions(latent, bsz).clone()

        device = self._module_device()
        dtype = self._module_dtype()
        action_chunk_v = action_chunk.to(device=device, dtype=dtype)
        action_history[:, -1] = action_chunk_v[:, 0]

        if K > 1:
            future_chunk_actions = action_chunk_v[:, 1:]
        else:
            future_chunk_actions = action_chunk_v.new_zeros(bsz, 0, self.action_dim)

        pred_z = self._chunk_forward_z(history, action_history, future_chunk_actions)
        target_z = pred_z[:, H - 1 : H - 1 + K]
        pred_obs, _ = self.separate_emb(target_z)
        hidden_seq = pred_obs["visual"].reshape(bsz, K, self.obs_dim)
        next_hidden = hidden_seq[:, -1]

        all_hidden = torch.cat([history, hidden_seq], dim=1)
        new_history = (
            all_hidden[:, -H:].contiguous() if H >= 1 else next_hidden[:, None]
        )

        if K > 1:
            combined_actions = torch.cat([action_history, future_chunk_actions], dim=1)
        else:
            combined_actions = action_history
        if H > 1:
            actions_kept = combined_actions[:, -(H - 1) :]
            zero_slot = combined_actions.new_zeros(bsz, 1, self.action_dim)
            new_action_history = torch.cat([actions_kept, zero_slot], dim=1)
        else:
            new_action_history = combined_actions.new_zeros(bsz, 1, self.action_dim)

        return {
            "hidden": next_hidden,
            "hidden_seq": hidden_seq,
            "history": new_history,
            "actions": new_action_history,
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
        if obs.ndim != 3:
            raise ValueError(
                f"obs_embedding must be [B,T,obs_dim], got {tuple(obs.shape)}"
            )
        T = int(obs.shape[1])
        if T < H + K:
            raise ValueError(f"chunk_loss requires T >= H+K = {H + K}, got T={T}")
        if actions.shape[1] < H - 1 + K:
            raise ValueError(
                f"actions length {actions.shape[1]} too short for chunk loss (need >= {H - 1 + K})"
            )

        obs = self._validate_obs_embedding(obs)
        actions = self._validate_actions(actions, int(actions.shape[1]))
        bsz = int(obs.shape[0])

        history = obs[:, :H]
        chunk_actions = actions[:, H - 1 : H - 1 + K]
        hidden_target = obs[:, H : H + K].detach()

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
            rollout_pred = torch.cat(rollout_preds, dim=1)  # [B, (N-1)*K, obs_dim]
            rollout_target = obs[:, H + K : H + N * K].detach()  # [B, (N-1)*K, obs_dim]
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
    ) -> "ChunkAwareRynnDinoWMWorldModel":
        """Warm-start a chunk WM from a parent RynnDinoWM (or old chunk) ckpt.

        Old checkpoints lack the new ``mask_obs_token`` parameter and were
        trained with per-step teacher forcing rather than chunk objective.
        The predictor / projection weights are still a sensible initialization,
        but to get correct Plan-B inference you MUST fine-tune with
        ``chunk_loss`` so the mask token learns its role.
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
        allowed_missing = {"mask_obs_token"}
        missing_real = [k for k in missing if k not in allowed_missing]
        if strict and (missing_real or unexpected):
            raise RuntimeError(
                f"strict load_state_dict failed: missing={missing_real[:5]} unexpected={unexpected[:5]}"
            )
        return wm.to(device)
