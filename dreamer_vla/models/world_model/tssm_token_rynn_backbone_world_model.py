from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from dreamer_vla.models.world_model.base_world_model import DreamerV3ActorAdapterMixin
from dreamer_vla.models.world_model.dreamerv3_torch import (
    DreamerV3Loss,
    DreamerV3PixelDecoder,
    FullHiddenSequenceDecoder,
    MLPHead,
    _make_reward_head,
    _module_device,
    _module_dtype,
    _reward_loss,
    _reward_pred,
)
from dreamer_vla.models.world_model.tssm_torch import (
    TSSMTokenDynamic,
    TSSMTokenLatentState,
    _PerTokenMLPDecoder,
    _Transformer,
    _build_hidden_decoder,
    _onehot_st_sample,
)


class TSSMTokenRynnBackboneWorldModel(DreamerV3ActorAdapterMixin):
    """Token-based TSSM WM: action_hidden kept as N=35 tokens of dim D_tok=1024.

    Key differences from the flat ``TSSMRynnBackboneWorldModel``:
        - No ``_RynnBackboneObsEncoder`` compressing flattened hidden→embed_dim. Instead the flattened hidden
          is reshaped to [B, T, 35, 1024] and each token is passed individually through
          a small per-token Linear (or Identity if d_model==1024).
        - The Transformer sees (T * 35) tokens with spatio-temporal causal mask
          (causal at time, bidirectional within timestep) — matches TransDreamer's
          original H*W * T = N * T tokenization 1:1.
        - Posterior, prior, deter, stoch are ALL per-token; richer latent that
          preserves the 35-token semantic structure of pi0's action queries.
        - hidden_decoder reconstructs the 35×1024 token sequence (per-token MLP shared
          across tokens, or Pi0Style transformer).
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        latent_dim: int | None = None,
        action_dim: int = 7,
        image_channels: int = 6,
        image_size: int = 64,
        n_tokens: int | None = None,
        token_dim: int = 1024,
        time_horizon: int = 5,
        # TSSM
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 32,
        tssm_layers: int = 4,
        tssm_nhead: int = 8,
        tssm_d_model: int = 1024,
        tssm_d_inner: int = 128,
        tssm_d_ff_inner: int = 4096,
        tssm_dropout: float = 0.1,
        tssm_dropatt: float = 0.0,
        tssm_pre_lnorm: bool = True,
        tssm_gating: bool = False,
        tssm_deter_type: str = "concat_o",
        tssm_window: int = 8,
        # cfg-compat (unused)
        blocks: int = 1,
        unimix: float = 0.0,
        deter: int | None = None,
        embed_dim: int | None = None,
        encoder_hidden: int = 2048,
        encoder_layers: int = 2,
        free_nats: float = 1.0,
        # pixel decoder
        depth: int = 64,
        mults: tuple = (2, 3, 4, 4),
        kernel: int = 5,
        act: str = "silu",
        # losses
        contdisc: bool = True,
        horizon: int = 333,
        dyn_scale: float = 1.0,
        rep_scale: float = 0.1,
        rec_scale: float = 1.0,
        rew_scale: float = 1.0,
        con_scale: float = 1.0,
        hidden_rec_scale: float = 100.0,
        # hidden_decoder (operates per-token; we use a small per-token MLP)
        hidden_decoder_kind: str = "per_token_mlp",  # 'per_token_mlp' | 'mlp' | 'resnet' | 'pi0_transformer'
        hidden_decoder_layers: int = 2,
        hidden_decoder_units: int = 4096,
        hidden_decoder_d_model: int = 1024,
        hidden_decoder_nhead: int = 8,
        hidden_decoder_mem_tokens: int = 8,
        hidden_decoder_dropout: float = 0.0,
        full_hidden_rec_scale: float = 0.0,
        actor_sequence_length: int = 0,
        actor_input_kind: str = "hidden",
        sequence_decoder_query_dim: int = 1024,
        sequence_decoder_layers: int = 1,
        sequence_decoder_units: int = 2048,
        reward_bins: int = 255,
        reward_head_type: str = "binary",
        reward_init_logit: float = 0.0,
        reward_pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.image_size = int(image_size)
        self.actor_input_kind = str(actor_input_kind).lower()
        self.n_tokens = (
            int(n_tokens)
            if n_tokens is not None
            else int(time_horizon) * int(action_dim)
        )
        self.token_dim = int(token_dim)
        if obs_dim is None:
            obs_dim = (
                int(latent_dim)
                if latent_dim is not None
                else self.n_tokens * self.token_dim
            )
        self.obs_dim = int(obs_dim)
        if self.n_tokens * self.token_dim != self.obs_dim:
            raise ValueError(
                f"obs_dim={self.obs_dim} must equal n_tokens * token_dim "
                f"({self.n_tokens} * {self.token_dim} = {self.n_tokens * self.token_dim})"
            )

        # TSSM dynamics (token-based)
        self.tssm = TSSMTokenDynamic(
            n_tokens=self.n_tokens,
            token_dim=self.token_dim,
            action_dim=action_dim,
            hidden=hidden,
            stoch=stoch,
            classes=classes,
            n_layers=tssm_layers,
            n_head=tssm_nhead,
            d_model=tssm_d_model,
            d_inner=tssm_d_inner,
            d_ff_inner=tssm_d_ff_inner,
            dropout=tssm_dropout,
            dropatt=tssm_dropatt,
            pre_lnorm=tssm_pre_lnorm,
            gating=tssm_gating,
            deter_type=tssm_deter_type,
            tssm_window=tssm_window,
            free_nats=free_nats,
        )
        self.rssm = self.tssm  # alias for DreamerV3ActorAdapterMixin compatibility

        # Aggregate deter (over N tokens) for pixel decoder & reward head (they expect [B, T, deter_dim])
        # We concat all N tokens' deter into one big vec: deter_dim_total = N * deter_per_token
        self.deter_dim_total = self.n_tokens * self.tssm.deter_per_token
        self.flat_stoch_total = self.n_tokens * self.tssm.flat_stoch
        feat_dim = self.deter_dim_total + self.flat_stoch_total

        # Pixel decoder (operates on aggregated deter; we treat aggregated deter as 'deter' input)
        self.decoder = DreamerV3PixelDecoder(
            image_channels=image_channels,
            image_size=image_size,
            deter=self.deter_dim_total,
            stoch=self.n_tokens * stoch,
            classes=classes,
            depth=depth,
            mults=tuple(mults),
            kernel=kernel,
            act=act,
        )

        # Hidden decoder
        if hidden_decoder_kind == "per_token_mlp":
            # Per-token shared MLP: (deter_per_tok + flat_stoch) → token_dim
            per_tok_in = self.tssm.deter_per_token + self.tssm.flat_stoch
            self.hidden_decoder = _PerTokenMLPDecoder(
                in_dim_per_tok=per_tok_in,
                out_dim_per_tok=self.token_dim,
                n_tokens=self.n_tokens,
                layers=hidden_decoder_layers,
                units=hidden_decoder_units,
                act=act,
            )
        else:
            self.hidden_decoder = _build_hidden_decoder(
                hidden_decoder_kind,
                feat_dim,
                self.obs_dim,
                layers=int(hidden_decoder_layers),
                units=int(hidden_decoder_units),
                d_model=int(hidden_decoder_d_model),
                nhead=int(hidden_decoder_nhead),
                mem_tokens=int(hidden_decoder_mem_tokens),
                dropout=float(hidden_decoder_dropout),
                act=act,
            )
        self.hidden_decoder_kind = str(hidden_decoder_kind).lower()

        self.actor_sequence_length = int(actor_sequence_length)
        self.full_hidden_rec_scale = float(full_hidden_rec_scale)
        self.sequence_decoder: nn.Module | None = None
        if self.actor_sequence_length > 0:
            self.sequence_decoder = FullHiddenSequenceDecoder(
                feat_dim,
                sequence_length=self.actor_sequence_length,
                hidden_dim=self.obs_dim,
                query_dim=int(sequence_decoder_query_dim),
                layers=int(sequence_decoder_layers),
                units=int(sequence_decoder_units),
                act=act,
            )

        # reward & continue heads
        self.reward_head = _make_reward_head(
            feat_dim=feat_dim,
            reward_bins=reward_bins,
            hidden=hidden,
            act=act,
            reward_head_type=reward_head_type,
            reward_init_logit=reward_init_logit,
            reward_pos_weight=reward_pos_weight,
        )
        self.continue_head = MLPHead(feat_dim, 1, layers=1, units=hidden, act=act)

        # loss scales
        self.contdisc = bool(contdisc)
        self.horizon = int(horizon)
        self.dyn_scale = float(dyn_scale)
        self.rep_scale = float(rep_scale)
        self.rec_scale = float(rec_scale)
        self.rew_scale = float(rew_scale)
        self.con_scale = float(con_scale)
        self.hidden_rec_scale = float(hidden_rec_scale)

    # ---- features ----

    def feature(self, seq: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate per-token (stoch, deter) into a flat per-step feat [B, T, feat_dim]."""
        stoch_flat = seq["stoch"].reshape(
            *seq["stoch"].shape[:-3], self.flat_stoch_total
        )  # [B, T, N*stoch*classes]
        deter_flat = seq["deter"].reshape(
            *seq["deter"].shape[:-2], self.deter_dim_total
        )  # [B, T, N*deter_per_tok]
        return torch.cat([stoch_flat, deter_flat], dim=-1)

    def _feature_dim(self) -> int:
        return self.deter_dim_total + self.flat_stoch_total

    def _obs_to_tokens(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        """Reshape [..., obs_dim=N*D_tok] to [..., N, D_tok] and cast to module dtype.
        obs_embedding may come from disk as fp16; cast it to the model's compute dtype
        (typically bf16) so downstream Linear layers don't hit dtype mismatch.
        """
        dtype = _module_dtype(self, obs_embedding.dtype)
        obs_embedding = obs_embedding.to(dtype=dtype)
        return obs_embedding.reshape(
            *obs_embedding.shape[:-1], self.n_tokens, self.token_dim
        )

    # ---- adapter mixin interface ----

    def encode_latent(self, hidden: torch.Tensor) -> TSSMTokenLatentState:
        device = _module_device(self, hidden.device)
        hidden = hidden.to(device=device)
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        # hidden: [B, obs_dim]
        obs_tokens = self._obs_to_tokens(hidden)  # [B, N, D_tok]
        tok_emb = self.tssm.token_embed(obs_tokens)
        post_logits = self.tssm._logit_view(self.tssm.post_stoch_mlp(tok_emb).float())
        stoch = _onehot_st_sample(post_logits).to(dtype=obs_tokens.dtype)
        deter = obs_tokens.new_zeros(
            obs_tokens.shape[0], self.n_tokens, self.tssm.deter_per_token
        )
        return TSSMTokenLatentState(stoch=stoch, deter=deter, logits=post_logits)

    def observe_next(
        self,
        latent: TSSMTokenLatentState,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor | bool | None = None,
    ) -> TSSMTokenLatentState:
        device = _module_device(self, hidden.device)
        hidden = hidden.to(device=device)
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        obs_tokens = self._obs_to_tokens(hidden)
        return self.tssm.observe_next(latent, obs_tokens, actions, is_first=is_first)

    def predict_next(
        self, latent: TSSMTokenLatentState, actions: torch.Tensor
    ) -> TSSMTokenLatentState:
        action = actions if actions.ndim == 2 else actions[:, 0]
        prev_stoch_step = latent.stoch.unsqueeze(1)
        action_step = action.unsqueeze(1)
        if latent.history_stoch is None:
            new_h_stoch = prev_stoch_step
            new_h_action = action_step
        else:
            new_h_stoch = torch.cat([latent.history_stoch, prev_stoch_step], dim=1)
            new_h_action = torch.cat([latent.history_action, action_step], dim=1)
        if new_h_stoch.shape[1] > self.tssm.tssm_window:
            new_h_stoch = new_h_stoch[:, -self.tssm.tssm_window :]
            new_h_action = new_h_action[:, -self.tssm.tssm_window :]
        T = new_h_stoch.shape[1]
        B = new_h_stoch.shape[0]
        N = self.n_tokens
        prev_stoch_flat = new_h_stoch.reshape(B, T, N, self.tssm.flat_stoch)
        tx_in = self.tssm._build_tx_input(prev_stoch_flat, new_h_action)
        mask = _Transformer.spatio_temporal_mask(T, N, action.device)
        o_t = self.tssm.cell(tx_in, attn_mask=mask)
        new_deter = self.tssm._per_token_aggregated(o_t, T, N)[:, -1]
        prior_logits = self.tssm._logit_view(
            self.tssm.prior_stoch_mlp(new_deter).float()
        )
        new_stoch = _onehot_st_sample(prior_logits).to(dtype=new_deter.dtype)
        return TSSMTokenLatentState(
            stoch=new_stoch,
            deter=new_deter,
            logits=prior_logits,
            history_stoch=new_h_stoch,
            history_action=new_h_action,
        )

    def actor_input(self, latent: TSSMTokenLatentState) -> torch.Tensor:
        if self.actor_input_kind == "feature":
            return latent.feature()
        # hidden_decoder expects either per-token (per_token_mlp) or flat feat
        if isinstance(self.hidden_decoder, _PerTokenMLPDecoder):
            stoch_flat = latent.stoch.reshape(
                *latent.stoch.shape[:-2], -1
            )  # [B, N, flat_stoch]
            feat_per_tok = torch.cat(
                [stoch_flat, latent.deter], dim=-1
            )  # [B, N, deter_per_tok+flat_stoch]
            return self.hidden_decoder(feat_per_tok)  # [B, obs_dim]
        return self.hidden_decoder(latent.feature())

    def actor_input_sequence(self, latent: TSSMTokenLatentState) -> torch.Tensor:
        if self.sequence_decoder is None:
            raise ValueError("actor_input_sequence requires actor_sequence_length > 0")
        return self.sequence_decoder(latent.feature())

    def critic_input(self, latent: TSSMTokenLatentState) -> torch.Tensor:
        return latent.feature()

    def observe_sequence(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        obs_embedding = batch["obs_embedding"]
        device = _module_device(self, obs_embedding.device)
        obs_embedding = obs_embedding.to(device=device)
        obs_tokens = self._obs_to_tokens(obs_embedding)  # [B, T, N, D_tok]
        actions = batch["actions"].to(device=device, dtype=obs_tokens.dtype)
        is_first = batch["is_first"].to(device=device)
        seq = self.tssm.observe(obs_tokens, actions, is_first)
        latent = TSSMTokenLatentState(
            deter=seq["deter"], stoch=seq["stoch"], logits=seq["post_logits"]
        )
        return {"latent": latent, "feat": self.feature(seq)}

    def state_reward(self, latent: TSSMTokenLatentState) -> torch.Tensor:
        pred = self.reward_head(latent.feature())
        return _reward_pred(self.reward_head, pred).squeeze(-1)

    def continue_prob(self, latent: TSSMTokenLatentState) -> torch.Tensor:
        return torch.sigmoid(self.continue_head(latent.feature()).squeeze(-1))

    def reward(self, latent, actions, next_latent):
        del latent, actions
        return self.state_reward(next_latent)

    def _resize_target(self, images, dtype, device):
        if images.ndim != 5:
            raise ValueError(f"images must be [B,T,C,H,W], got {tuple(images.shape)}")
        bsz, steps, channels, height, width = images.shape
        target = images.to(device=device, dtype=dtype) / 255.0
        if (height, width) == (self.image_size, self.image_size):
            return target
        flat = target.reshape(bsz * steps, channels, height, width)
        flat = F.interpolate(
            flat,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return flat.reshape(bsz, steps, channels, self.image_size, self.image_size)

    def loss(self, batch: dict[str, torch.Tensor]) -> DreamerV3Loss:
        images = batch["images"]
        obs_embedding = batch["obs_embedding"]
        actions = batch["actions"]
        rewards = batch["rewards"].to(device=actions.device, dtype=actions.dtype)
        terminal = batch.get("is_terminal", batch["dones"])
        dones = terminal.to(device=actions.device, dtype=actions.dtype)
        is_first = batch["is_first"].to(device=actions.device)

        obs_tokens = self._obs_to_tokens(obs_embedding)  # [B, T, N, D_tok]
        seq = self.tssm.observe(
            obs_tokens, actions.to(dtype=obs_tokens.dtype), is_first
        )
        kls = self.tssm.kl_loss(seq["post_logits"], seq["prior_logits"])

        # For pixel decoder: aggregate (stoch, deter) across N tokens
        B, T, N, _, _ = seq["stoch"].shape
        agg_deter = seq["deter"].reshape(B, T, self.deter_dim_total)
        agg_stoch = seq["stoch"].reshape(
            B, T, self.n_tokens * self.tssm.stoch, self.tssm.classes
        )
        recon = self.decoder(agg_deter, agg_stoch)
        target = self._resize_target(
            images[:, 1:], dtype=recon.dtype, device=recon.device
        )
        rec_per = (recon - target).square().sum(dim=(-3, -2, -1))
        rec_loss = rec_per.mean()

        # hidden_decoder (per-token preferred)
        if isinstance(self.hidden_decoder, _PerTokenMLPDecoder):
            stoch_flat = seq["stoch"].reshape(
                *seq["stoch"].shape[:-2], -1
            )  # [B, T, N, stoch*classes]
            feat_per_tok = torch.cat(
                [stoch_flat, seq["deter"]], dim=-1
            )  # [B, T, N, per_tok_in]
            hidden_pred = self.hidden_decoder(feat_per_tok)  # [B, T, obs_dim]
        else:
            hidden_pred = self.hidden_decoder(self.feature(seq))
        hidden_target = (
            obs_embedding[:, 1:]
            .to(device=hidden_pred.device, dtype=hidden_pred.dtype)
            .detach()
        )
        hidden_mse = (hidden_pred.float() - hidden_target.float()).square().mean()
        hidden_pred_norm = F.normalize(hidden_pred.float(), dim=-1)
        hidden_target_norm = F.normalize(hidden_target.float(), dim=-1)
        hidden_cosine = 1.0 - (hidden_pred_norm * hidden_target_norm).sum(dim=-1).mean()
        full_hidden_loss = obs_tokens.new_zeros(())
        full_hidden_cosine = obs_tokens.new_zeros(())

        feat = self.feature(seq)
        reward_logits = self.reward_head(feat)
        cont_logits = self.continue_head(feat).squeeze(-1)
        reward_target = rewards[:, 1:].to(
            device=reward_logits.device, dtype=reward_logits.dtype
        )
        reward_loss = _reward_loss(self.reward_head, reward_logits, reward_target)
        cont_target = 1.0 - dones[:, 1:].to(
            device=cont_logits.device, dtype=cont_logits.dtype
        )
        if self.contdisc:
            cont_target = cont_target * (1.0 - 1.0 / float(self.horizon))
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

        loss = (
            self.rec_scale * rec_loss
            + self.dyn_scale * kls["dyn"]
            + self.rep_scale * kls["rep"]
            + self.rew_scale * reward_loss
            + self.con_scale * cont_loss
            + self.hidden_rec_scale * hidden_mse
            + self.full_hidden_rec_scale * full_hidden_loss
        )
        zero = kls["dyn"].new_zeros(())
        metrics = {
            "rec_loss": rec_loss.detach(),
            "dyn_kl": kls["dyn"].detach(),
            "rep_kl": kls["rep"].detach(),
            "kl_loss": (kls["dyn"] + kls["rep"]).detach(),
            "reward_loss": reward_loss.detach(),
            "cont_loss": cont_loss.detach(),
            "hidden_rec_loss": hidden_mse.detach(),
            "hidden_rec_scaled_loss": (self.hidden_rec_scale * hidden_mse).detach(),
            "hidden_cosine_loss": hidden_cosine.detach(),
            "full_hidden_rec_loss": full_hidden_loss.detach(),
            "full_hidden_rec_scaled_loss": (
                self.full_hidden_rec_scale * full_hidden_loss
            ).detach(),
            "full_hidden_cosine_loss": full_hidden_cosine.detach(),
            "hidden_pred_norm": hidden_pred.float().norm(dim=-1).mean().detach(),
            "hidden_target_norm": hidden_target.float().norm(dim=-1).mean().detach(),
            "image_decoder_loss": rec_per.mean().detach(),
            "image_recon_mse_loss": (recon - target).float().square().mean().detach(),
            "predicted_reward_mean": _reward_pred(self.reward_head, reward_logits)
            .mean()
            .detach(),
            "transition_loss": zero,
            "delta_latent_loss": zero,
            "action_margin_loss": zero,
            "image_recon_ce_loss": zero,
            "image_static_ce_loss": zero,
            "image_dynamic_ce_loss": zero,
            "image_recon_accuracy": zero,
            "image_static_accuracy": zero,
            "image_dynamic_accuracy": zero,
            "image_dynamic_fraction": zero,
            "pred_entropy": zero,
            "pred_unique_tokens": zero,
            "gt_unique_tokens": zero,
            "latent_norm": zero,
            "grad_norm": zero,
        }
        return DreamerV3Loss(loss=loss, metrics=metrics)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict) and "mode" in batch:
            return self._forward_actor_adapter(batch)
        out = self.loss(batch)
        return self._compat_forward_dict(out)


__all__ = ["TSSMTokenRynnBackboneWorldModel"]
