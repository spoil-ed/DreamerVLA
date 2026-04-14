from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.chameleon_model.modeling_xllmx_chameleon_ck_action_head import (
    ChameleonXLLMXForConditionalGeneration_ck_action_head,
)


@dataclass
class TSSMState:
    mean: torch.Tensor
    std: torch.Tensor
    stoch: torch.Tensor
    deter: torch.Tensor

    def feature(self) -> torch.Tensor:
        return torch.cat([self.stoch, self.deter], dim=-1)


class TSSMWorldModel(nn.Module):
    """
    A TSSM-style world model adapted to the current DreamerVLA interface.

    The transition backbone is loaded exactly like RynnVLA-002 loads its
    action/world-model checkpoint: instantiate the full HF
    `ChameleonXLLMXForConditionalGeneration_ck_action_head` checkpoint and then
    drop the VQ module. For TSSM we only consume the decoder transformer hidden
    states as a causal dynamics backbone; we intentionally do not add a second
    autoregressive token head on top because the checkpoint already contains the
    causal decoder stack and TSSM's supervision is carried by the latent
    prior/posterior + transition/reward heads defined below.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        latent_dim: int = 256,
        state_token_count: int = 1,
        mapper_hidden_dim: int = 1024,
        dynamics_hidden_dim: int = 4096,
        reward_hidden_dim: int = 512,
        reward_loss_coef: float = 0.0,
        kl_loss_coef: float = 0.1,
        min_std: float = 0.1,
        pretrained_model_path: str = "/home/user01/yuxinglei/workspace/DreamerVLA/data/ckpts/Action_World_model_512/libero_10",
        freeze_transition_backbone: bool = False,
        backbone_dtype: str = "bfloat16",
        transition_time_horizon: int | None = None,
        training: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.state_token_count = int(state_token_count)
        self.mapper_hidden_dim = int(mapper_hidden_dim)
        self.dynamics_hidden_dim = int(dynamics_hidden_dim)
        self.reward_loss_coef = float(reward_loss_coef)
        self.kl_loss_coef = float(kl_loss_coef)
        self.min_std = float(min_std)
        self.freeze_transition_backbone = bool(freeze_transition_backbone)
        self.pretrained_model_path = str(pretrained_model_path)
        self.transition_time_horizon = self._resolve_transition_time_horizon(
            pretrained_model_path=self.pretrained_model_path,
            fallback=transition_time_horizon,
        )

        self.obs_to_stoch = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, 2 * self.latent_dim),
        )
        self.obs_to_deter = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, self.dynamics_hidden_dim),
        )
        self.state_to_tokens = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.dynamics_hidden_dim),
            nn.Linear(self.latent_dim + self.dynamics_hidden_dim, self.state_token_count * self.dynamics_hidden_dim),
        )
        self.action_to_tokens = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.dynamics_hidden_dim),
        )
        self.token_type_embeddings = nn.Embedding(2, self.dynamics_hidden_dim)

        self.transition_summary = nn.Sequential(
            nn.LayerNorm(self.dynamics_hidden_dim),
            nn.Linear(self.dynamics_hidden_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, self.dynamics_hidden_dim),
        )
        self.prior_head = nn.Sequential(
            nn.LayerNorm(self.dynamics_hidden_dim),
            nn.Linear(self.dynamics_hidden_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, 2 * self.latent_dim),
        )
        self.posterior_head = nn.Sequential(
            nn.LayerNorm(self.dynamics_hidden_dim + self.obs_dim),
            nn.Linear(self.dynamics_hidden_dim + self.obs_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, 2 * self.latent_dim),
        )
        self.transition_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.dynamics_hidden_dim),
            nn.Linear(self.latent_dim + self.dynamics_hidden_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, self.obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.LayerNorm(2 * (self.latent_dim + self.dynamics_hidden_dim) + self.action_dim),
            nn.Linear(2 * (self.latent_dim + self.dynamics_hidden_dim) + self.action_dim, reward_hidden_dim),
            nn.GELU(),
            nn.Linear(reward_hidden_dim, 1),
        )

        torch_dtype = getattr(torch, backbone_dtype)
        self.transition_backbone = self._load_transition_backbone_like_rynnvla002(
            pretrained_model_path=self.pretrained_model_path,
            action_dim=self.action_dim,
            time_horizon=self.transition_time_horizon,
            torch_dtype=torch_dtype,
        )
        if self.freeze_transition_backbone:
            self.transition_backbone.eval()
            for parameter in self.transition_backbone.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _resolve_pretrained_model_dir(pretrained_model_path: str) -> Path:
        candidate = Path(pretrained_model_path).expanduser().resolve()
        if candidate.is_dir():
            if (candidate / "config.json").is_file():
                return candidate
            for subdir in sorted(path for path in candidate.iterdir() if path.is_dir()):
                if (subdir / "config.json").is_file():
                    return subdir.resolve()
        return candidate

    @classmethod
    def _load_transition_backbone_like_rynnvla002(
        cls,
        pretrained_model_path: str,
        action_dim: int,
        time_horizon: int,
        torch_dtype: torch.dtype,
    ) -> ChameleonXLLMXForConditionalGeneration_ck_action_head:
        model_dir = cls._resolve_pretrained_model_dir(pretrained_model_path)
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Transition backbone config.json not found under {model_dir}")

        config = json.loads(config_path.read_text())
        model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            str(model_dir),
            action_dim=int(action_dim),
            time_horizon=int(time_horizon),
            max_position_embeddings=int(config.get("max_position_embeddings", 8192)),
            mask_image_logits=bool(config.get("mask_image_logits", False)),
            dropout=float(config.get("dropout", 0.0)),
            z_loss_weight=float(config.get("z_loss_weight", 0.0)),
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
            device_map="cpu",
            ignore_mismatched_sizes=False,
            low_cpu_mem_usage=True,
        )
        if hasattr(model.model, "vqmodel"):
            del model.model.vqmodel
        return model

    @staticmethod
    def _resolve_transition_time_horizon(pretrained_model_path: str, fallback: int | None) -> int:
        if fallback is not None:
            return int(fallback)
        config_path = TSSMWorldModel._resolve_pretrained_model_dir(pretrained_model_path) / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            time_horizon = config.get("time_horizon")
            if time_horizon is not None:
                return int(time_horizon)
        return 5

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

    def _stats_to_stoch(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std_param = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std_param) + self.min_std
        stoch = mean
        return mean, std, stoch

    def _build_state(self, stats: torch.Tensor, deter: torch.Tensor) -> TSSMState:
        mean, std, stoch = self._stats_to_stoch(stats)
        return TSSMState(mean=mean, std=std, stoch=stoch, deter=deter)

    def encode_latent(self, hidden: torch.Tensor) -> TSSMState:
        reduced_hidden = self._reduce_hidden(hidden)
        deter = self.obs_to_deter(reduced_hidden)
        stats = self.obs_to_stoch(reduced_hidden)
        return self._build_state(stats, deter)

    def _build_transition_inputs(self, state: TSSMState, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, horizon, _ = action.shape
        state_feature = state.feature()
        state_tokens = self.state_to_tokens(state_feature).view(batch_size, self.state_token_count, self.dynamics_hidden_dim)
        action_tokens = self.action_to_tokens(action)

        state_type_ids = torch.zeros(batch_size, self.state_token_count, dtype=torch.long, device=action.device)
        action_type_ids = torch.ones(batch_size, horizon, dtype=torch.long, device=action.device)
        state_tokens = state_tokens + self.token_type_embeddings(state_type_ids)
        action_tokens = action_tokens + self.token_type_embeddings(action_type_ids)

        input_embeds = torch.cat([state_tokens, action_tokens], dim=1)
        attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.bool, device=input_embeds.device)
        return input_embeds, attention_mask

    def _predict_prior(self, state: TSSMState, action: torch.Tensor) -> TSSMState:
        input_embeds, attention_mask = self._build_transition_inputs(state, action)
        backbone_dtype = next(self.transition_backbone.parameters()).dtype
        outputs = self.transition_backbone.model(
            inputs_embeds=input_embeds.to(dtype=backbone_dtype),
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        deter = self.transition_summary(outputs.last_hidden_state[:, -1])
        stats = self.prior_head(deter)
        return self._build_state(stats, deter)

    def predict_next(self, latent: TSSMState, actions: torch.Tensor) -> TSSMState:
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        return self._predict_prior(latent, actions)

    @staticmethod
    def _gaussian_kl_divergence(post: TSSMState, prior: TSSMState) -> torch.Tensor:
        post_var = post.std.pow(2)
        prior_var = prior.std.pow(2)
        log_ratio = torch.log(prior.std) - torch.log(post.std)
        sq_term = (post_var + (post.mean - prior.mean).pow(2)) / (2.0 * prior_var.clamp_min(1e-6))
        kl = log_ratio + sq_term - 0.5
        return kl.sum(dim=-1).mean()

    def reward(
        self,
        latent: TSSMState,
        actions: torch.Tensor,
        next_latent: TSSMState,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if actions.ndim == 3:
            actions = actions.mean(dim=1)
        features = torch.cat([latent.feature(), actions, next_latent.feature()], dim=-1)
        return self.reward_head(features).squeeze(-1)

    def pretrain_loss(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor,
        next_hidden: torch.Tensor,
        reward_target: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        action = self._apply_action_mask(action, action_mask)
        latent = self.encode_latent(hidden)
        prior_next = self.predict_next(latent, action)

        next_hidden_target = self._reduce_hidden(next_hidden)
        posterior_stats = self.posterior_head(torch.cat([prior_next.deter, next_hidden_target], dim=-1))
        posterior_next = self._build_state(posterior_stats, prior_next.deter)

        predicted_next_hidden = self.transition_head(prior_next.feature())
        transition_loss = F.mse_loss(predicted_next_hidden, next_hidden_target)
        kl_loss = self._gaussian_kl_divergence(posterior_next, prior_next)

        predicted_reward = self.reward(latent, action, prior_next)
        if reward_target is None:
            reward_target = torch.zeros_like(predicted_reward)
        reward_target = reward_target.reshape_as(predicted_reward)
        reward_loss = F.mse_loss(predicted_reward, reward_target)

        loss = transition_loss + self.kl_loss_coef * kl_loss
        if self.reward_loss_coef > 0:
            loss = loss + self.reward_loss_coef * reward_loss

        return {
            "loss": loss,
            "transition_loss": transition_loss,
            "kl_loss": kl_loss,
            "reward_loss": reward_loss,
            "predicted_reward_mean": predicted_reward.mean(),
            "latent_norm": latent.feature().norm(dim=-1).mean(),
        }

    def compute_loss_dict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        hidden = batch.get("obs_embedding")
        next_hidden = batch.get("next_obs_embedding")
        action = batch["action"]
        action_mask = batch.get("action_mask")
        reward = batch.get("reward")
        if hidden is None or next_hidden is None:
            raise ValueError("World model expects `obs_embedding` and `next_obs_embedding` in the batch.")
        first_param = next(self.parameters())
        device = first_param.device
        model_dtype = first_param.dtype
        hidden = hidden.to(device=device, dtype=model_dtype)
        next_hidden = next_hidden.to(device=device, dtype=model_dtype)
        action = action.to(device=device, dtype=model_dtype)
        if action_mask is not None:
            action_mask = action_mask.to(device=device)
        if reward is not None:
            reward = reward.to(device=device, dtype=model_dtype)
        return self.pretrain_loss(
            hidden=hidden,
            action=action,
            next_hidden=next_hidden,
            reward_target=reward,
            action_mask=action_mask,
        )

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.compute_loss_dict(batch)["loss"]

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """FSDP-compatible entry point: delegates to compute_loss_dict.

        Calling compute_loss_dict directly bypasses FSDP's all-gather hook
        (which fires only on __call__ / forward).  Routing through forward
        ensures sharded parameters are fully materialized before use.
        """
        return self.compute_loss_dict(batch)


__all__ = ["TSSMState", "TSSMWorldModel"]
