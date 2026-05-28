from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer_vla.models.world_model.base_world_model import DreamerV3ActorAdapterMixin
from dreamer_vla.models.world_model.dreamerv3_torch import (
    DreamerV3Loss,
    DreamerV3PixelDecoder,
    FullHiddenSequenceDecoder,
    MLPHead,
    _RynnBackboneObsEncoder,
    _make_reward_head,
    _module_device,
    _reward_loss,
    _reward_pred,
)
from dreamer_vla.models.world_model.tssm_torch import (
    TSSMDynamic,
    TSSMLatentState,
    _build_hidden_decoder,
    _onehot_st_sample,
)


class TSSMRynnBackboneWorldModel(DreamerV3ActorAdapterMixin):
    """DreamerVLA WM with TSSM (TransDreamer) replacing RSSM. Faithful TransDreamer port."""

    def __init__(
        self,
        obs_dim: int = 35840,
        latent_dim: int | None = None,
        action_dim: int = 7,
        image_channels: int = 6,
        image_size: int = 64,
        # encoder (shared)
        embed_dim: int | None = None,
        encoder_hidden: int = 2048,
        encoder_layers: int = 2,
        # TSSM
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 32,
        tssm_layers: int = 4,
        tssm_nhead: int = 4,
        tssm_d_model: int = 384,
        tssm_d_inner: int = 96,
        tssm_d_ff_inner: int = 1536,
        tssm_dropout: float = 0.1,
        tssm_dropatt: float = 0.0,
        tssm_pre_lnorm: bool = True,
        tssm_gating: bool = False,
        tssm_deter_type: str = "concat_o",
        tssm_window: int = 64,
        # legacy / unused but kept for cfg-compat
        blocks: int = 1,
        unimix: float = 0.0,
        deter: int | None = None,  # ignored; computed from tssm_layers * tssm_d_model
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
        # hidden_decoder
        hidden_decoder_kind: str = "mlp",
        hidden_decoder_layers: int = 1,
        hidden_decoder_units: int = 2048,
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
        if latent_dim is not None:
            obs_dim = int(obs_dim if obs_dim is not None else latent_dim)
        self.obs_dim = int(obs_dim)
        self.image_channels = int(image_channels)
        self.image_size = int(image_size)
        self.actor_input_kind = str(actor_input_kind).lower()

        # encoder
        self.encoder = _RynnBackboneObsEncoder(
            obs_dim=obs_dim,
            embed_dim=embed_dim,
            hidden=encoder_hidden,
            layers=encoder_layers,
            act=act,
        )

        # TSSM dynamics
        self.tssm = TSSMDynamic(
            obs_emb_dim=self.encoder.out_dim,
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
        self.rssm = (
            self.tssm
        )  # alias so DreamerV3ActorAdapterMixin paths that touch .rssm work

        # pixel decoder reused
        self.decoder = DreamerV3PixelDecoder(
            image_channels=image_channels,
            image_size=image_size,
            deter=self.tssm.deter,
            stoch=stoch,
            classes=classes,
            depth=depth,
            mults=tuple(mults),
            kernel=kernel,
            act=act,
        )

        feat_dim = self.tssm.deter + self.tssm.flat_stoch
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
        deter = seq["deter"]
        stoch_flat = seq["stoch"].reshape(*seq["stoch"].shape[:-2], -1)
        return torch.cat([stoch_flat, deter], dim=-1)

    def _feature_dim(self) -> int:
        return self.tssm.deter + self.tssm.flat_stoch

    def _single_observation_sequence(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 2:
            return hidden.unsqueeze(1)
        return hidden

    # ---- adapter mixin interface ----

    def encode_latent(self, hidden: torch.Tensor) -> TSSMLatentState:
        device = _module_device(self, hidden.device)
        obs = self._single_observation_sequence(hidden.to(device=device))
        enc = self.encoder(obs)
        post_logits = self.tssm._logit_view(self.tssm.post_stoch_mlp(enc[:, 0]).float())
        stoch = _onehot_st_sample(post_logits).to(dtype=enc.dtype)
        deter = enc.new_zeros(enc.shape[0], self.tssm.deter)
        return TSSMLatentState(stoch=stoch, deter=deter, logits=post_logits)

    def observe_next(
        self,
        latent: TSSMLatentState,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor | bool | None = None,
    ) -> TSSMLatentState:
        device = _module_device(self, hidden.device)
        obs = self._single_observation_sequence(hidden.to(device=device))
        enc = self.encoder(obs)
        return self.tssm.observe_next(latent, enc[:, 0], actions, is_first=is_first)

    def predict_next(
        self, latent: TSSMLatentState, actions: torch.Tensor
    ) -> TSSMLatentState:
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
        tx_in = self.tssm._build_tx_input(new_h_stoch, new_h_action)
        o_t = self.tssm.cell(tx_in)
        new_deter = self.tssm._deter_from_layers(o_t)[:, -1]
        prior_logits = self.tssm._logit_view(
            self.tssm.prior_stoch_mlp(new_deter).float()
        )
        new_stoch = _onehot_st_sample(prior_logits).to(dtype=new_deter.dtype)
        return TSSMLatentState(
            stoch=new_stoch,
            deter=new_deter,
            logits=prior_logits,
            history_stoch=new_h_stoch,
            history_action=new_h_action,
        )

    def actor_input(self, latent: TSSMLatentState) -> torch.Tensor:
        if self.actor_input_kind == "feature":
            return latent.feature()
        return self.hidden_decoder(latent.feature())

    def actor_input_sequence(self, latent: TSSMLatentState) -> torch.Tensor:
        if self.sequence_decoder is None:
            raise ValueError("actor_input_sequence requires actor_sequence_length > 0")
        return self.sequence_decoder(latent.feature())

    def critic_input(self, latent: TSSMLatentState) -> torch.Tensor:
        return latent.feature()

    def observe_sequence(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        obs_embedding = batch["obs_embedding"]
        device = _module_device(self, obs_embedding.device)
        enc = self.encoder(obs_embedding.to(device=device))
        actions = batch["actions"].to(device=device, dtype=enc.dtype)
        is_first = batch["is_first"].to(device=device)
        seq = self.tssm.observe(enc, actions, is_first)
        latent = TSSMLatentState(
            deter=seq["deter"], stoch=seq["stoch"], logits=seq["post_logits"]
        )
        return {"latent": latent, "feat": self.feature(seq)}

    def state_reward(self, latent: TSSMLatentState) -> torch.Tensor:
        pred = self.reward_head(latent.feature())
        return _reward_pred(self.reward_head, pred).squeeze(-1)

    def continue_prob(self, latent: TSSMLatentState) -> torch.Tensor:
        return torch.sigmoid(self.continue_head(latent.feature()).squeeze(-1))

    def reward(self, latent, actions, next_latent):
        del latent, actions
        return self.state_reward(next_latent)

    def _resize_target(
        self, images: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"images must be [B,T,C,H,W], got {tuple(images.shape)}")
        bsz, steps, channels, height, width = images.shape
        if channels != self.image_channels:
            raise ValueError(
                f"Expected {self.image_channels} image channels, got {channels}"
            )
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

        enc = self.encoder(obs_embedding)
        seq = self.tssm.observe(enc, actions.to(dtype=enc.dtype), is_first)
        kls = self.tssm.kl_loss(seq["post_logits"], seq["prior_logits"])

        recon = self.decoder(seq["deter"], seq["stoch"])
        target = self._resize_target(
            images[:, 1:], dtype=recon.dtype, device=recon.device
        )
        rec_per = (recon - target).square().sum(dim=(-3, -2, -1))
        rec_loss = rec_per.mean()

        feat = self.feature(seq)
        hidden_pred = self.hidden_decoder(feat)
        hidden_target = (
            obs_embedding[:, 1:]
            .to(device=hidden_pred.device, dtype=hidden_pred.dtype)
            .detach()
        )
        hidden_mse = (hidden_pred.float() - hidden_target.float()).square().mean()
        hidden_pred_norm = F.normalize(hidden_pred.float(), dim=-1)
        hidden_target_norm = F.normalize(hidden_target.float(), dim=-1)
        hidden_cosine = 1.0 - (hidden_pred_norm * hidden_target_norm).sum(dim=-1).mean()
        full_hidden_loss = feat.new_zeros(())
        full_hidden_cosine = feat.new_zeros(())

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


__all__ = ["TSSMRynnBackboneWorldModel"]
