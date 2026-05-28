from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.world_model.base_world_model import (
    DreamerV3ActorAdapterMixin,
    DreamerV3Loss,
)
from src.models.world_model.dreamerv3_torch import (
    DreamerV3RSSM,
    DreamerV3TokenDecoder,
    DreamerV3TokenEncoder,
    MLPHead,
    _make_reward_head,
    _reward_loss,
    _reward_pred,
)


class DreamerV3TokenWorldModel(DreamerV3ActorAdapterMixin):
    """DreamerV3 RSSM with categorical image-token observations.

    This is the controlled token counterpart of ``DreamerV3PixelWorldModel``:
    same RSSM, same aggregate free-nats KL semantics, same reward and continue
    heads. The observation likelihood is the only intended change, replacing
    pixel MSE with categorical CE over image tokens.
    """

    def __init__(
        self,
        action_dim: int = 7,
        num_image_tokens_vocab: int = 8192,
        n_image_tokens: int = 512,
        num_views: int = 2,
        spatial_grid: tuple[int, int] = (16, 16),
        token_embed_dim: int = 512,
        deter: int = 8192,
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 64,
        blocks: int = 8,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3),
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
        rec_reduction: str = "sum",
    ) -> None:
        super().__init__()
        self.encoder = DreamerV3TokenEncoder(
            num_image_tokens_vocab=num_image_tokens_vocab,
            n_image_tokens=n_image_tokens,
            num_views=num_views,
            spatial_grid=tuple(spatial_grid),
            token_embed_dim=token_embed_dim,
            depth=depth,
            mults=tuple(mults),
            kernel=kernel,
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
        self.decoder = DreamerV3TokenDecoder(
            num_image_tokens_vocab=num_image_tokens_vocab,
            n_image_tokens=n_image_tokens,
            num_views=num_views,
            spatial_grid=tuple(spatial_grid),
            deter=deter,
            stoch=stoch,
            classes=classes,
            depth=depth,
            mults=tuple(mults),
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
        self.rec_reduction = str(rec_reduction).lower()
        if self.rec_reduction not in {"sum", "mean"}:
            raise ValueError("rec_reduction must be 'sum' or 'mean'")
        self.contdisc = bool(contdisc)
        self.horizon = int(horizon)

    def feature(self, seq: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [seq["deter"], seq["stoch"].reshape(*seq["stoch"].shape[:2], -1)],
            dim=-1,
        )

    def loss(self, batch: dict[str, torch.Tensor]) -> DreamerV3Loss:
        tokens = batch["tokens"].long()
        actions = batch["actions"]
        rewards = batch["rewards"].to(device=actions.device, dtype=actions.dtype)
        terminal = batch.get("is_terminal", batch["dones"])
        dones = terminal.to(device=actions.device, dtype=actions.dtype)
        is_first = batch["is_first"].to(device=actions.device)

        enc = self.encoder(tokens)
        seq = self.rssm.observe(enc, actions, is_first)
        kls = self.rssm.kl_loss(seq["post_logits"], seq["prior_logits"])
        logits = self.decoder(seq["deter"], seq["stoch"])  # [B,T,views,tokens,classes]

        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1).to(device=logits.device),
            reduction="none",
        ).reshape(tokens.shape)
        rec_per_step = ce.reshape(ce.shape[0], ce.shape[1], -1).sum(dim=-1)
        rec_loss = rec_per_step.mean() if self.rec_reduction == "sum" else ce.mean()

        feat = self.feature(seq)
        reward_logits = self.reward_head(feat)
        cont_logits = self.continue_head(feat).squeeze(-1)
        reward_loss = _reward_loss(self.reward_head, reward_logits, rewards)
        cont_target = 1.0 - dones.to(device=cont_logits.device, dtype=cont_logits.dtype)
        if self.contdisc:
            cont_target = cont_target * (1.0 - 1.0 / float(self.horizon))
        cont_loss = F.binary_cross_entropy_with_logits(
            cont_logits, cont_target, reduction="none"
        ).mean()

        loss = (
            self.rec_scale * rec_loss
            + self.dyn_scale * kls["dyn"]
            + self.rep_scale * kls["rep"]
            + self.rew_scale * reward_loss
            + self.con_scale * cont_loss
        )

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            reward_pred = _reward_pred(self.reward_head, reward_logits.detach())
            token_acc = (pred == tokens.to(device=pred.device)).float().mean()
            token_ce = ce.detach().float().mean()
            log_probs = F.log_softmax(logits.detach().float(), dim=-1)
            probs = log_probs.exp()
            pred_entropy = -(probs * log_probs).sum(dim=-1).mean()
            flat_pred = pred.reshape(pred.shape[0] * pred.shape[1], -1)
            flat_gt = tokens.to(device=pred.device).reshape(
                tokens.shape[0] * tokens.shape[1], -1
            )
            pred_unique = torch.tensor(
                [int(torch.unique(row).numel()) for row in flat_pred],
                dtype=logits.dtype,
                device=logits.device,
            ).mean()
            gt_unique = torch.tensor(
                [int(torch.unique(row).numel()) for row in flat_gt],
                dtype=logits.dtype,
                device=logits.device,
            ).mean()
        metrics = {
            "loss": loss.detach(),
            "rec_loss": rec_loss.detach(),
            "token_ce": token_ce,
            "token_acc": token_acc.detach(),
            "dyn_loss": kls["dyn"].detach(),
            "rep_loss": kls["rep"].detach(),
            "reward_loss": reward_loss.detach(),
            "reward_pred_mean": reward_pred.mean().detach(),
            "continue_loss": cont_loss.detach(),
            "pred_entropy": pred_entropy.detach(),
            "pred_unique_tokens": pred_unique.detach(),
            "gt_unique_tokens": gt_unique.detach(),
            "dyn_entropy": kls["dyn_entropy"].detach(),
            "rep_entropy": kls["rep_entropy"].detach(),
        }
        return DreamerV3Loss(loss=loss, metrics=metrics)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict) and batch.get("mode") is not None:
            return self._forward_actor_adapter(batch)
        out = self.loss(batch)
        return self._compat_forward_dict(out)


__all__ = ["DreamerV3TokenWorldModel"]
