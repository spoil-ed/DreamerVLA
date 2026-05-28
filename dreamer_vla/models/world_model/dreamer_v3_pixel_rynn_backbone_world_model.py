from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer_vla.models.world_model.base_world_model import (
    DreamerV3ActorAdapterMixin,
    DreamerV3LatentState,
    DreamerV3Loss,
)
from dreamer_vla.models.world_model.dreamerv3_torch import (
    DreamerV3PixelDecoder,
    DreamerV3RSSM,
    FullHiddenSequenceDecoder,
    MLPHead,
    PerTokenMLPHead,
    Pi0StyleHiddenDecoder,
    Pi0TimeBroadcastDecoder,
    ResMLPHead,
    _RynnBackboneObsEncoder,
    _make_reward_head,
    _module_device,
    _module_dtype,
    _reward_loss,
    _reward_pred,
)


class DreamerV3PixelRynnBackboneWorldModel(DreamerV3ActorAdapterMixin):
    """Pixel DreamerV3 with the observation encoder replaced by frozen RynnVLA.

    The workspace supplies:

      images: raw pixel observations, used as the reconstruction target.
      obs_embedding: frozen RynnVLA-002 hidden vectors, used as RSSM observations.

    Thus only the ``encode`` slot changes.  RSSM, pixel decoder, reward head and
    continue head stay aligned with ``DreamerV3PixelWorldModel``.
    """

    def __init__(
        self,
        obs_dim: int = 4096,
        latent_dim: int | None = None,
        action_dim: int = 7,
        image_channels: int = 6,
        image_size: int = 64,
        embed_dim: int | None = None,
        encoder_hidden: int = 2048,
        encoder_layers: int = 2,
        deter: int = 8192,
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 64,
        blocks: int = 8,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel: int = 5,
        act: str = "silu",
        unimix: float = 0.01,
        free_nats: float = 1.0,
        reward_bins: int = 255,
        reward_head_type: str = "twohot",
        reward_init_logit: float = -5.0,
        reward_pos_weight: float | None = None,
        contdisc: bool = True,
        horizon: int = 333,
        dyn_scale: float = 1.0,
        rep_scale: float = 0.1,
        rec_scale: float = 1.0,
        rew_scale: float = 1.0,
        con_scale: float = 1.0,
        hidden_rec_scale: float = 100.0,
        hidden_decoder_layers: int = 1,
        hidden_decoder_units: int = 2048,
        hidden_decoder_kind: str = "mlp",
        hidden_decoder_d_model: int = 1024,
        hidden_decoder_nhead: int = 8,
        hidden_decoder_mem_tokens: int = 8,
        hidden_decoder_dropout: float = 0.0,
        hidden_decoder_n_tokens: int = 35,
        hidden_decoder_token_dim: int = 1024,
        hidden_decoder_query_dim: int = 128,
        hidden_decoder_n_time_queries: int = 5,
        hidden_decoder_joint_broadcast: int = 7,
        hidden_decoder: nn.Module
        | None = None,  # ← LEGO slot: cfg can _target_ a module directly
        full_hidden_rec_scale: float = 0.0,
        actor_sequence_length: int = 0,
        actor_input_kind: str = "hidden",
        sequence_decoder_query_dim: int = 1024,
        sequence_decoder_layers: int = 1,
        sequence_decoder_units: int = 2048,
    ) -> None:
        super().__init__()
        if latent_dim is not None:
            obs_dim = int(obs_dim if obs_dim is not None else latent_dim)
        self.obs_dim = int(obs_dim)
        self.image_channels = int(image_channels)
        self.image_size = int(image_size)
        self.actor_input_kind = str(actor_input_kind).lower()
        if self.actor_input_kind not in {"hidden", "feature"}:
            raise ValueError("actor_input_kind must be one of: hidden, feature")
        self.encoder_is_identity = embed_dim is None or int(embed_dim) == self.obs_dim
        self.encoder = _RynnBackboneObsEncoder(
            obs_dim=obs_dim,
            embed_dim=embed_dim,
            hidden=encoder_hidden,
            layers=encoder_layers,
            act=act,
        )
        self.rssm = DreamerV3RSSM(
            action_dim=action_dim,
            deter=deter,
            hidden=hidden,
            stoch=stoch,
            classes=classes,
            blocks=blocks,
            unimix=unimix,
            free_nats=free_nats,
            act=act,
        )
        self.rssm.build_posterior(self.encoder.out_dim, act=act)
        self.decoder = DreamerV3PixelDecoder(
            image_channels=image_channels,
            image_size=image_size,
            deter=deter,
            stoch=stoch,
            classes=classes,
            depth=depth,
            mults=tuple(mults),
            kernel=kernel,
            act=act,
        )
        feat_dim = int(deter) + int(stoch) * int(classes)
        # LEGO slot path: if cfg supplies a pre-built nn.Module (via _target_), use it directly.
        # Otherwise fall back to the kind-string dispatch (legacy cfg path).
        if hidden_decoder is not None:
            if not isinstance(hidden_decoder, nn.Module):
                raise TypeError(
                    f"hidden_decoder must be an nn.Module if provided, got {type(hidden_decoder)}"
                )
            self.hidden_decoder = hidden_decoder
            self.hidden_decoder_kind = str(
                getattr(hidden_decoder, "__class__", type(hidden_decoder)).__name__
            )
            kind = "custom"  # marker so downstream code knows the dispatch was bypassed
        else:
            kind = str(hidden_decoder_kind).lower()
        if kind == "custom":
            pass  # already built above
        elif kind == "mlp":
            self.hidden_decoder = MLPHead(
                feat_dim,
                self.obs_dim,
                layers=int(hidden_decoder_layers),
                units=int(hidden_decoder_units),
                act=act,
            )
        elif kind == "resnet":
            self.hidden_decoder = ResMLPHead(
                feat_dim,
                self.obs_dim,
                layers=int(hidden_decoder_layers),
                units=int(hidden_decoder_units),
                act=act,
            )
        elif kind in {"pi0_transformer", "transformer", "pi0"}:
            self.hidden_decoder = Pi0StyleHiddenDecoder(
                feat_dim,
                self.obs_dim,
                layers=int(hidden_decoder_layers),
                d_model=int(hidden_decoder_d_model),
                nhead=int(hidden_decoder_nhead),
                mem_tokens=int(hidden_decoder_mem_tokens),
                token_dim=int(hidden_decoder_token_dim),
                dropout=float(hidden_decoder_dropout),
                act=act,
            )
        elif kind in {"pi0_time_broadcast", "time_broadcast", "pi0_time"}:
            self.hidden_decoder = Pi0TimeBroadcastDecoder(
                feat_dim,
                self.obs_dim,
                layers=int(hidden_decoder_layers),
                d_model=int(hidden_decoder_d_model),
                nhead=int(hidden_decoder_nhead),
                mem_tokens=int(hidden_decoder_mem_tokens),
                n_time_queries=int(hidden_decoder_n_time_queries),
                joint_broadcast=int(hidden_decoder_joint_broadcast),
                token_dim=int(hidden_decoder_token_dim),
                dropout=float(hidden_decoder_dropout),
                act=act,
            )
        elif kind in {"per_token_mlp", "per_token", "token_mlp"}:
            n_tokens = int(hidden_decoder_n_tokens)
            token_dim = int(hidden_decoder_token_dim)
            if n_tokens * token_dim != self.obs_dim:
                raise ValueError(
                    f"per_token_mlp: n_tokens × token_dim ({n_tokens}×{token_dim} = "
                    f"{n_tokens * token_dim}) must equal obs_dim ({self.obs_dim})"
                )
            self.hidden_decoder = PerTokenMLPHead(
                feat_dim,
                n_tokens=n_tokens,
                token_dim=token_dim,
                query_dim=int(hidden_decoder_query_dim),
                layers=int(hidden_decoder_layers),
                units=int(hidden_decoder_units),
                act=act,
            )
        else:
            raise ValueError(f"Unknown hidden_decoder_kind: {hidden_decoder_kind}")
        if kind != "custom":
            self.hidden_decoder_kind = kind
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
        self.reward_head = _make_reward_head(
            feat_dim,
            reward_bins,
            hidden,
            act,
            reward_head_type=reward_head_type,
            reward_init_logit=reward_init_logit,
            reward_pos_weight=reward_pos_weight,
        )
        self.continue_head = MLPHead(feat_dim, 1, layers=1, units=hidden, act=act)
        self.dyn_scale = float(dyn_scale)
        self.rep_scale = float(rep_scale)
        self.rec_scale = float(rec_scale)
        self.rew_scale = float(rew_scale)
        self.con_scale = float(con_scale)
        self.hidden_rec_scale = float(hidden_rec_scale)
        self.contdisc = bool(contdisc)
        self.horizon = int(horizon)

    def _feature_dim(self) -> int:
        return int(self.rssm.deter + self.rssm.stoch * self.rssm.classes)

    def _encode_obs_embedding(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        if not self.encoder_is_identity:
            return self.encoder(obs_embedding)
        if obs_embedding.ndim == 2:
            obs_embedding = obs_embedding[:, None]
        if obs_embedding.ndim != 3:
            raise ValueError(
                f"Rynn backbone obs_embedding must be [B,T,D] or [B,D], got {tuple(obs_embedding.shape)}"
            )
        if obs_embedding.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Rynn backbone obs dim mismatch: got {obs_embedding.shape[-1]}, expected {self.obs_dim}"
            )
        device = _module_device(self, obs_embedding.device)
        dtype = _module_dtype(self, obs_embedding.dtype)
        return obs_embedding.to(device=device, dtype=dtype)

    def feature(self, seq: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [seq["deter"], seq["stoch"].reshape(*seq["stoch"].shape[:2], -1)],
            dim=-1,
        )

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

    def encode_latent(self, hidden: torch.Tensor) -> DreamerV3LatentState:
        if (
            hidden.ndim == 2
            and hidden.shape[-1] == self._feature_dim()
            and torch.is_floating_point(hidden)
        ):
            return self._latent_from_feature(hidden)
        device = _module_device(self, hidden.device)
        obs = self._encode_obs_embedding(hidden.to(device=device))
        batch_size = obs.shape[0]
        dtype = obs.dtype
        actions = torch.zeros(
            batch_size, 1, self.rssm.action_dim, device=device, dtype=dtype
        )
        is_first = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
        seq = self.rssm.observe(obs, actions, is_first)
        return DreamerV3LatentState(
            deter=seq["deter"][:, 0],
            stoch=seq["stoch"][:, 0],
            logits=seq["post_logits"][:, 0],
        )

    def observe_next(
        self,
        latent: DreamerV3LatentState,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor | bool | None = None,
    ) -> DreamerV3LatentState:
        device = _module_device(self, hidden.device)
        enc = self._encode_obs_embedding(hidden.to(device=device))
        if enc.ndim != 3 or enc.shape[1] != 1:
            raise ValueError(
                f"Rynn observe_next expected one obs embedding, got encoder output {tuple(enc.shape)}"
            )
        return self.rssm.observe_next(latent, enc[:, 0], actions, is_first=is_first)

    def observe_sequence(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        obs_embedding = batch["obs_embedding"]
        device = _module_device(self, obs_embedding.device)
        enc = self._encode_obs_embedding(obs_embedding.to(device=device))
        actions = batch["actions"].to(device=device, dtype=enc.dtype)
        is_first = batch["is_first"].to(device=device)
        seq = self.rssm.observe(enc, actions, is_first)
        latent = DreamerV3LatentState(
            deter=seq["deter"],
            stoch=seq["stoch"],
            logits=seq["post_logits"],
        )
        return {"latent": latent, "feat": latent.feature()}

    def actor_input(self, latent: DreamerV3LatentState) -> torch.Tensor:
        if self.actor_input_kind == "feature":
            return latent.feature()
        return self.hidden_decoder(latent.feature())

    def actor_input_sequence(self, latent: DreamerV3LatentState) -> torch.Tensor:
        if self.sequence_decoder is None:
            raise ValueError("actor_input_sequence requires actor_sequence_length > 0")
        return self.sequence_decoder(latent.feature())

    def critic_input(self, latent: DreamerV3LatentState) -> torch.Tensor:
        return latent.feature()

    def loss(self, batch: dict[str, torch.Tensor]) -> DreamerV3Loss:
        images = batch["images"]
        obs_embedding = batch["obs_embedding"]
        actions = batch["actions"]
        rewards = batch["rewards"].to(device=actions.device, dtype=actions.dtype)
        terminal = batch.get("is_terminal", batch["dones"])
        dones = terminal.to(device=actions.device, dtype=actions.dtype)
        is_first = batch["is_first"].to(device=actions.device)

        enc = self._encode_obs_embedding(obs_embedding)
        seq = self.rssm.observe(enc, actions, is_first)
        kls = self.rssm.kl_loss(seq["post_logits"], seq["prior_logits"])
        recon = self.decoder(seq["deter"], seq["stoch"])
        target = self._resize_target(images, dtype=recon.dtype, device=recon.device)
        rec_per = (recon - target).square().sum(dim=(-3, -2, -1))
        rec_loss = rec_per.mean()

        feat = self.feature(seq)
        hidden_pred = self.hidden_decoder(feat)
        hidden_target = obs_embedding.to(
            device=hidden_pred.device, dtype=hidden_pred.dtype
        ).detach()
        hidden_mse = (hidden_pred.float() - hidden_target.float()).square().mean()
        hidden_pred_norm = F.normalize(hidden_pred.float(), dim=-1)
        hidden_target_norm = F.normalize(hidden_target.float(), dim=-1)
        hidden_cosine = 1.0 - (hidden_pred_norm * hidden_target_norm).sum(dim=-1).mean()
        full_hidden_loss = feat.new_zeros(())
        full_hidden_cosine = feat.new_zeros(())
        if self.sequence_decoder is not None and "actor_hidden_states" in batch:
            full_pred = self.sequence_decoder(feat)
            full_target = (
                batch["actor_hidden_states"]
                .to(device=full_pred.device, dtype=full_pred.dtype)
                .detach()
            )
            if full_target.shape[-2] != full_pred.shape[-2]:
                target_len = int(full_pred.shape[-2])
                if full_target.shape[-2] > target_len:
                    full_target = full_target[..., :target_len, :]
                else:
                    pad = target_len - int(full_target.shape[-2])
                    full_target = F.pad(full_target, (0, 0, 0, pad))
            mask = batch.get("actor_attention_mask")
            if isinstance(mask, torch.Tensor):
                mask = mask.to(device=full_pred.device).bool()[
                    ..., : full_pred.shape[-2]
                ]
            else:
                mask = torch.ones(
                    full_pred.shape[:-1], device=full_pred.device, dtype=torch.bool
                )
            mask_f = mask.to(dtype=full_pred.dtype).unsqueeze(-1)
            denom = mask_f.sum().clamp_min(1.0) * full_pred.shape[-1]
            full_hidden_loss = (
                (full_pred.float() - full_target.float()).square() * mask_f.float()
            ).sum() / denom
            pred_norm = F.normalize(full_pred.float(), dim=-1)
            target_norm = F.normalize(full_target.float(), dim=-1)
            full_hidden_cosine = (
                (1.0 - (pred_norm * target_norm).sum(dim=-1)) * mask.float()
            ).sum() / mask.float().sum().clamp_min(1.0)
        reward_logits = self.reward_head(feat)
        cont_logits = self.continue_head(feat).squeeze(-1)
        reward_loss = _reward_loss(self.reward_head, reward_logits, rewards)
        cont_target = 1.0 - dones.to(device=cont_logits.device, dtype=cont_logits.dtype)
        if self.contdisc:
            cont_target = cont_target * (1.0 - 1.0 / float(self.horizon))
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

        loss = (
            self.rec_scale * rec_loss
            + self.dyn_scale * kls["dyn"]
            + self.rep_scale * kls["rep"]
            + self.hidden_rec_scale * hidden_mse
            + self.full_hidden_rec_scale * full_hidden_loss
            + self.rew_scale * reward_loss
            + self.con_scale * cont_loss
        )
        mse = (recon.detach() - target).square().mean()
        metrics = {
            "loss": loss.detach(),
            "rec_loss": rec_loss.detach(),
            "dyn_loss": kls["dyn"].detach(),
            "rep_loss": kls["rep"].detach(),
            "reward_loss": reward_loss.detach(),
            "continue_loss": cont_loss.detach(),
            "hidden_rec_loss": hidden_mse.detach(),
            "hidden_rec_scaled_loss": (self.hidden_rec_scale * hidden_mse).detach(),
            "hidden_cosine_loss": hidden_cosine.detach(),
            "full_hidden_rec_loss": full_hidden_loss.detach(),
            "full_hidden_rec_scaled_loss": (
                self.full_hidden_rec_scale * full_hidden_loss
            ).detach(),
            "full_hidden_cosine_loss": full_hidden_cosine.detach(),
            "hidden_pred_norm": hidden_pred.detach()
            .float()
            .norm(dim=-1)
            .mean()
            .detach(),
            "hidden_target_norm": hidden_target.detach()
            .float()
            .norm(dim=-1)
            .mean()
            .detach(),
            "reward_pred_mean": _reward_pred(self.reward_head, reward_logits.detach())
            .mean()
            .detach(),
            "image_mse": mse.detach(),
            "image_psnr": (-10.0 * torch.log10(mse.clamp_min(1e-8))).detach(),
            "dyn_entropy": kls["dyn_entropy"].detach(),
            "rep_entropy": kls["rep_entropy"].detach(),
        }
        return DreamerV3Loss(loss=loss, metrics=metrics)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict) and batch.get("mode") is not None:
            return self._forward_actor_adapter(batch)
        out = self.loss(batch)
        return self._compat_forward_dict(out)


__all__ = ["DreamerV3PixelRynnBackboneWorldModel"]
