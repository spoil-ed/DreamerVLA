from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.world_model.base_world_model import (
    DreamerV3ActorAdapterMixin,
    DreamerV3Loss,
)
from src.models.world_model.dreamerv3_torch import (
    DreamerV3PixelDecoder,
    DreamerV3PixelEncoder,
    DreamerV3RSSM,
    MLPHead,
    _make_reward_head,
    _reward_loss,
    _reward_pred,
)


class DreamerV3PixelWorldModel(DreamerV3ActorAdapterMixin):
    """PyTorch DreamerV3 world model for pixel observations.

    This intentionally mirrors the public DreamerV3 architecture: CNN encoder,
    discrete RSSM with block-GRU deterministic state, decoder from (h, z),
    reward and continue heads, and split dyn/rep KL losses.
    """

    def __init__(
        self,
        action_dim: int = 7,
        image_channels: int = 6,
        image_size: int = 64,
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
    ) -> None:
        super().__init__()
        self.encoder = DreamerV3PixelEncoder(
            image_channels, image_size, depth, mults, kernel, act
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
            mults=mults,
            kernel=kernel,
            act=act,
        )
        feat_dim = int(deter) + int(stoch) * int(classes)
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
        self.contdisc = bool(contdisc)
        self.horizon = int(horizon)

    def feature(self, seq: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [seq["deter"], seq["stoch"].reshape(*seq["stoch"].shape[:2], -1)], dim=-1
        )

    def loss(self, batch: dict[str, torch.Tensor]) -> DreamerV3Loss:
        images = batch["images"]
        actions = batch["actions"]
        rewards = batch["rewards"].to(device=images.device, dtype=torch.float32)
        dones = batch["dones"].to(device=images.device, dtype=torch.float32)
        is_first = batch["is_first"].to(device=images.device)

        tokens = self.encoder(images)
        seq = self.rssm.observe(tokens, actions, is_first)
        kls = self.rssm.kl_loss(seq["post_logits"], seq["prior_logits"])
        recon = self.decoder(seq["deter"], seq["stoch"])
        target = images.to(device=recon.device, dtype=recon.dtype) / 255.0
        rec_per = (recon - target).square().sum(dim=(-3, -2, -1))
        rec_loss = rec_per.mean()

        feat = self.feature(seq)
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


__all__ = ["DreamerV3PixelWorldModel"]
