from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from dreamervla.models.embodiment.world_model.common import (
    _module_device,
    _module_dtype,
)
from dreamervla.models.embodiment.world_model.reward_heads import _reward_pred


class BaseWorldModel(nn.Module, ABC):
    """Common base class for DreamerVLA world models."""

    @abstractmethod
    def loss(self, batch: dict[str, torch.Tensor]) -> Any:
        raise NotImplementedError


@dataclass
class DreamerV3Loss:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass
class DreamerV3LatentState:
    """Single-step DreamerV3 RSSM state used by DreamerVLA actor cotrain."""

    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor | None = None

    def feature(self) -> torch.Tensor:
        return torch.cat(
            [self.deter, self.stoch.reshape(*self.stoch.shape[:-2], -1)], dim=-1
        )


class DreamerV3ActorAdapterMixin(BaseWorldModel):
    """Adds DreamerVLA's single-step actor interface to DreamerV3-style WMs."""

    def _single_observation_sequence(self, hidden: torch.Tensor) -> torch.Tensor:
        encoder = getattr(self, "encoder", None)
        is_token_encoder = bool(hasattr(encoder, "num_image_tokens_vocab"))
        if is_token_encoder:
            if hidden.ndim in {2, 3}:
                return hidden[:, None]
            if hidden.ndim == 4 and hidden.shape[1] == 1:
                return hidden
            raise ValueError(
                f"Unsupported DreamerV3 token observation shape: {tuple(hidden.shape)}"
            )

        if hidden.ndim == 4:
            return hidden[:, None]
        if hidden.ndim == 5 and hidden.shape[1] == 1:
            return hidden
        raise ValueError(
            f"Unsupported DreamerV3 single observation shape: {tuple(hidden.shape)}"
        )

    def _feature_dim(self) -> int:
        return int(self.rssm.deter + self.rssm.stoch * self.rssm.classes)

    def _latent_from_feature(self, feature: torch.Tensor) -> DreamerV3LatentState:
        dtype = _module_dtype(self, feature.dtype)
        device = _module_device(self, feature.device)
        feature = feature.to(device=device, dtype=dtype)
        deter = feature[:, : self.rssm.deter]
        stoch_flat = feature[:, self.rssm.deter :]
        stoch = stoch_flat.reshape(feature.shape[0], self.rssm.stoch, self.rssm.classes)
        return DreamerV3LatentState(deter=deter, stoch=stoch)

    def encode_latent(self, hidden: torch.Tensor) -> DreamerV3LatentState:
        if (
            hidden.ndim == 2
            and hidden.shape[-1] == self._feature_dim()
            and torch.is_floating_point(hidden)
        ):
            return self._latent_from_feature(hidden)

        device = _module_device(self, hidden.device)
        obs = self._single_observation_sequence(hidden.to(device=device))
        enc = self.encoder(obs)
        batch_size = enc.shape[0]
        dtype = enc.dtype
        actions = torch.zeros(
            batch_size, 1, self.rssm.action_dim, device=device, dtype=dtype
        )
        is_first = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
        seq = self.rssm.observe(enc, actions, is_first)
        return DreamerV3LatentState(
            deter=seq["deter"][:, 0],
            stoch=seq["stoch"][:, 0],
            logits=seq["post_logits"][:, 0],
        )

    def predict_next(
        self, latent: DreamerV3LatentState, actions: torch.Tensor
    ) -> DreamerV3LatentState:
        device = _module_device(self, actions.device)
        dtype = latent.deter.dtype
        action = actions.to(device=device, dtype=dtype)
        if action.ndim == 3:
            action = action[:, 0, :]
        deter = self.rssm._core(
            latent.deter.to(device=device, dtype=dtype),
            latent.stoch.to(device=device, dtype=dtype),
            action,
        )
        logits = self.rssm._prior(deter)
        stoch = self.rssm._sample(logits)
        return DreamerV3LatentState(deter=deter, stoch=stoch, logits=logits)

    def observe_next(
        self,
        latent: DreamerV3LatentState,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor | bool | None = None,
    ) -> DreamerV3LatentState:
        device = _module_device(self, hidden.device)
        obs = self._single_observation_sequence(hidden.to(device=device))
        enc = self.encoder(obs)
        return self.rssm.observe_next(latent, enc[:, 0], actions, is_first=is_first)

    def actor_input(self, latent: DreamerV3LatentState) -> torch.Tensor:
        return latent.feature()

    def critic_input(self, latent: DreamerV3LatentState) -> torch.Tensor:
        return latent.feature()

    def observe_sequence(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        if "images" in batch:
            obs = batch["images"]
        elif "tokens" in batch:
            obs = batch["tokens"]
        else:
            raise KeyError("DreamerV3 observe_sequence expects `images` or `tokens`.")
        device = _module_device(self, obs.device)
        enc = self.encoder(obs.to(device=device))
        actions = batch["actions"].to(device=device, dtype=enc.dtype)
        is_first = batch["is_first"].to(device=device)
        seq = self.rssm.observe(enc, actions, is_first)
        latent = DreamerV3LatentState(
            deter=seq["deter"],
            stoch=seq["stoch"],
            logits=seq["post_logits"],
        )
        return {"latent": latent, "feat": latent.feature()}

    def state_reward(self, latent: DreamerV3LatentState) -> torch.Tensor:
        pred = self.reward_head(latent.feature())
        return _reward_pred(self.reward_head, pred).squeeze(-1)

    def continue_prob(self, latent: DreamerV3LatentState) -> torch.Tensor:
        return torch.sigmoid(self.continue_head(latent.feature()).squeeze(-1))

    def reward(
        self,
        latent: DreamerV3LatentState,
        actions: torch.Tensor,
        next_latent: DreamerV3LatentState,
    ) -> torch.Tensor:
        del latent, actions
        return self.state_reward(next_latent)

    def _forward_actor_adapter(self, batch: dict[str, Any]) -> Any:
        mode = batch.get("mode")
        if mode == "encode_latent":
            return self.encode_latent(batch["hidden"])
        if mode == "predict_next":
            return self.predict_next(batch["latent"], batch["actions"])
        if mode == "observe_next":
            return self.observe_next(
                batch["latent"],
                batch["hidden"],
                batch["actions"],
                is_first=batch.get("is_first"),
            )
        if mode == "reward":
            if "next_latent" in batch:
                return self.reward(
                    batch["latent"], batch.get("actions"), batch["next_latent"]
                )
            return self.state_reward(batch["latent"])
        if mode == "continue":
            return self.continue_prob(batch["latent"])
        if mode == "actor_input":
            return self.actor_input(batch["latent"])
        if mode == "actor_input_sequence":
            if not hasattr(self, "actor_input_sequence"):
                raise ValueError(
                    "actor_input_sequence is not implemented for this world model"
                )
            return self.actor_input_sequence(batch["latent"])
        if mode == "critic_input":
            return self.critic_input(batch["latent"])
        if mode == "observe_sequence":
            return self.observe_sequence(batch)
        raise ValueError(f"Unknown DreamerV3 actor-adapter mode: {mode!r}")

    def feature(self, seq: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [seq["deter"], seq["stoch"].reshape(*seq["stoch"].shape[:2], -1)],
            dim=-1,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict) and batch.get("mode") is not None:
            return self._forward_actor_adapter(batch)
        out = self.loss(batch)
        return self._compat_forward_dict(out)

    @staticmethod
    def _compat_forward_dict(out: DreamerV3Loss) -> dict[str, torch.Tensor]:
        result = {"_loss": out.loss, **out.metrics}
        result["loss"] = out.loss
        zero = out.loss.new_zeros(())
        if "dyn_loss" in result:
            result.setdefault("dyn_kl", result["dyn_loss"])
        if "rep_loss" in result:
            result.setdefault("rep_kl", result["rep_loss"])
        if "rec_loss" in result:
            result.setdefault("image_decoder_loss", result["rec_loss"])
        if "image_mse" in result:
            result.setdefault("image_recon_mse_loss", result["image_mse"])
        if "token_ce" in result:
            result.setdefault("image_recon_ce_loss", result["token_ce"])
        if "token_acc" in result:
            result.setdefault("image_recon_accuracy", result["token_acc"])
        if "reward_pred_mean" in result:
            result.setdefault("predicted_reward_mean", result["reward_pred_mean"])
        result.setdefault("transition_loss", zero)
        result.setdefault(
            "kl_loss", result.get("dyn_loss", zero) + result.get("rep_loss", zero)
        )
        result.setdefault("delta_latent_loss", zero)
        result.setdefault("action_margin_loss", zero)
        return result


__all__ = [
    "BaseWorldModel",
    "DreamerV3ActorAdapterMixin",
    "DreamerV3LatentState",
    "DreamerV3Loss",
    "_module_dtype",
    "_module_device",
]
