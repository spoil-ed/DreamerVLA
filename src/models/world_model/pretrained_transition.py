from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.chameleon_model.modeling_xllmx_chameleon_ck_action_head import (
    ChameleonXLLMXForConditionalGeneration_ck_action_head,
)


class PretrainedTransitionWorldModel(nn.Module):
    """
    Hybrid world model:
    1. Write our own mapping from observation hidden -> latent.
    2. Use a Chameleon backbone initialized from `starting_point` as the transition model
       over the latent/action sequence via `inputs_embeds`.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        latent_dim: int = 256,
        mapper_hidden_dim: int = 1024,
        transition_hidden_dim: int = 4096,
        state_token_count: int = 1,
        pretrained_model_path: str = "/home/yuxinglei/workspace/2026nips/Dreamer-VLA/data/ckpts/starting_point",
        reward_hidden_dim: int = 512,
        reward_loss_coef: float = 0.0,
        consistency_loss_coef: float = 0.1,
        freeze_transition_backbone: bool = False,
        training: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.mapper_hidden_dim = int(mapper_hidden_dim)
        self.transition_hidden_dim = int(transition_hidden_dim)
        self.state_token_count = int(state_token_count)
        self.reward_loss_coef = float(reward_loss_coef)
        self.consistency_loss_coef = float(consistency_loss_coef)
        self.freeze_transition_backbone = bool(freeze_transition_backbone)

        self.latent_mapper = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, self.latent_dim),
        )
        self.next_latent_target = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, self.latent_dim),
        )
        self.state_to_tokens = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.state_token_count * self.transition_hidden_dim),
        )
        self.action_to_tokens = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.transition_hidden_dim),
        )
        self.token_type_embeddings = nn.Embedding(2, self.transition_hidden_dim)
        self.transition_summary = nn.Sequential(
            nn.LayerNorm(self.transition_hidden_dim),
            nn.Linear(self.transition_hidden_dim, self.mapper_hidden_dim),
            nn.GELU(),
        )
        self.next_hidden_head = nn.Linear(self.mapper_hidden_dim, self.obs_dim)
        self.next_latent_head = nn.Linear(self.mapper_hidden_dim, self.latent_dim)
        self.reward_head = nn.Sequential(
            nn.LayerNorm(self.mapper_hidden_dim),
            nn.Linear(self.mapper_hidden_dim, reward_hidden_dim),
            nn.GELU(),
            nn.Linear(reward_hidden_dim, 1),
        )

        self.transition_backbone = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            pretrained_model_path,
            action_dim=self.action_dim,
            time_horizon=5,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        if hasattr(self.transition_backbone.model, "vqmodel"):
            del self.transition_backbone.model.vqmodel
        if self.freeze_transition_backbone:
            self.transition_backbone.eval()
            for parameter in self.transition_backbone.parameters():
                parameter.requires_grad = False

    def _reduce_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 2:
            return hidden
        if hidden.ndim == 3:
            return hidden.mean(dim=1)
        raise ValueError(f"Unsupported hidden shape: {tuple(hidden.shape)}")

    @staticmethod
    def _apply_action_mask(actions: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None or actions.ndim != 3:
            return actions
        mask = action_mask.to(device=actions.device, dtype=actions.dtype).unsqueeze(-1)
        return actions * mask

    def encode_latent(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.latent_mapper(self._reduce_hidden(hidden))

    def _build_transition_inputs(self, latent: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, horizon, _ = action.shape
        state_tokens = self.state_to_tokens(latent).view(batch_size, self.state_token_count, self.transition_hidden_dim)
        action_tokens = self.action_to_tokens(action)

        state_type_ids = torch.zeros(batch_size, self.state_token_count, dtype=torch.long, device=latent.device)
        action_type_ids = torch.ones(batch_size, horizon, dtype=torch.long, device=latent.device)
        state_tokens = state_tokens + self.token_type_embeddings(state_type_ids)
        action_tokens = action_tokens + self.token_type_embeddings(action_type_ids)

        input_embeds = torch.cat([state_tokens, action_tokens], dim=1)
        attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.bool, device=input_embeds.device)
        return input_embeds, attention_mask

    def predict_next_hidden(self, latent: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_embeds, attention_mask = self._build_transition_inputs(latent, action)
        backbone_dtype = next(self.transition_backbone.parameters()).dtype
        outputs = self.transition_backbone.model(
            inputs_embeds=input_embeds.to(dtype=backbone_dtype),
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        summary = self.transition_summary(outputs.last_hidden_state[:, -1].float())
        predicted_next_hidden = self.next_hidden_head(summary)
        predicted_next_latent = self.next_latent_head(summary)
        predicted_reward = self.reward_head(summary).squeeze(-1)
        return predicted_next_hidden, predicted_next_latent, predicted_reward

    def pretrain_loss(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor,
        next_hidden: torch.Tensor,
        reward_target: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        action = self._apply_action_mask(action, action_mask)
        current_hidden = self._reduce_hidden(hidden)
        next_hidden_target = self._reduce_hidden(next_hidden)
        latent = self.encode_latent(current_hidden)
        next_latent_target = self.next_latent_target(next_hidden_target)

        predicted_next_hidden, predicted_next_latent, predicted_reward = self.predict_next_hidden(latent, action)
        transition_loss = F.mse_loss(predicted_next_hidden, next_hidden_target)
        consistency_loss = F.mse_loss(predicted_next_latent, next_latent_target)

        if reward_target is None:
            reward_target = torch.zeros_like(predicted_reward).unsqueeze(-1)
        reward_target = reward_target.reshape_as(predicted_reward)
        reward_loss = F.mse_loss(predicted_reward, reward_target)

        loss = transition_loss + self.consistency_loss_coef * consistency_loss
        if self.reward_loss_coef > 0:
            loss = loss + self.reward_loss_coef * reward_loss

        return {
            "loss": loss,
            "transition_loss": transition_loss,
            # Keep the key for compatibility with the existing workspace logging.
            "kl_loss": consistency_loss,
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "predicted_reward_mean": predicted_reward.mean(),
            "latent_norm": latent.norm(dim=-1).mean(),
        }

    def compute_loss_dict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        hidden = batch.get("obs_embedding")
        next_hidden = batch.get("next_obs_embedding")
        action = batch["action"]
        action_mask = batch.get("action_mask")
        reward = batch.get("reward")
        if hidden is None or next_hidden is None:
            raise ValueError("World model expects `obs_embedding` and `next_obs_embedding` in the batch.")
        device = next(self.parameters()).device
        hidden = hidden.to(device)
        next_hidden = next_hidden.to(device)
        action = action.to(device)
        if action_mask is not None:
            action_mask = action_mask.to(device)
        if reward is not None:
            reward = reward.to(device)
        return self.pretrain_loss(
            hidden=hidden,
            action=action,
            next_hidden=next_hidden,
            reward_target=reward,
            action_mask=action_mask,
        )

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.compute_loss_dict(batch)["loss"]


__all__ = ["PretrainedTransitionWorldModel"]
