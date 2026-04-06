from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RSSMState:
    mean: torch.Tensor
    std: torch.Tensor
    stoch: torch.Tensor
    deter: torch.Tensor

    def feature(self) -> torch.Tensor:
        return torch.cat([self.stoch, self.deter], dim=-1)


class RSSMCore(nn.Module):
    def __init__(self, stoch_dim: int, deter_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.stoch_dim = int(stoch_dim)
        self.deter_dim = int(deter_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)

        self.img_in = nn.Linear(self.stoch_dim + self.action_dim, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.deter_dim)
        self.img_hidden = nn.Linear(self.deter_dim, self.hidden_dim)
        self.img_stats = nn.Linear(self.hidden_dim, 2 * self.stoch_dim)

        self.obs_hidden = nn.Linear(self.deter_dim + self.hidden_dim, self.hidden_dim)
        self.obs_stats = nn.Linear(self.hidden_dim, 2 * self.stoch_dim)

    def initial(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> RSSMState:
        zeros_stoch = torch.zeros(batch_size, self.stoch_dim, device=device, dtype=dtype)
        zeros_deter = torch.zeros(batch_size, self.deter_dim, device=device, dtype=dtype)
        zeros_std = torch.ones(batch_size, self.stoch_dim, device=device, dtype=dtype)
        return RSSMState(mean=zeros_stoch, std=zeros_std, stoch=zeros_stoch, deter=zeros_deter)

    def _stats_to_state(self, stats: torch.Tensor, deter: torch.Tensor) -> RSSMState:
        mean, std_param = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std_param) + 0.1
        stoch = mean
        return RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)

    def img_step(self, prev_state: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        x = torch.cat([prev_state.stoch, prev_action], dim=-1)
        x = F.elu(self.img_in(x))
        deter = self.gru(x, prev_state.deter)
        h = F.elu(self.img_hidden(deter))
        stats = self.img_stats(h)
        return self._stats_to_state(stats, deter)

    def obs_step(self, prev_state: RSSMState, prev_action: torch.Tensor, embed: torch.Tensor) -> tuple[RSSMState, RSSMState]:
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior.deter, embed], dim=-1)
        h = F.elu(self.obs_hidden(x))
        stats = self.obs_stats(h)
        post = self._stats_to_state(stats, prior.deter)
        return post, prior

    def posterior_from_prior(self, prior: RSSMState, embed: torch.Tensor) -> RSSMState:
        x = torch.cat([prior.deter, embed], dim=-1)
        h = F.elu(self.obs_hidden(x))
        stats = self.obs_stats(h)
        return self._stats_to_state(stats, prior.deter)

    def observe(self, embed: torch.Tensor, action: torch.Tensor, state: RSSMState | None = None) -> tuple[RSSMState, RSSMState]:
        batch_size, horizon, _ = action.shape
        if state is None:
            state = self.initial(batch_size, device=embed.device, dtype=embed.dtype)

        post_states: list[RSSMState] = []
        prior_states: list[RSSMState] = []
        prev_state = state
        for t in range(horizon):
            post, prior = self.obs_step(prev_state, action[:, t], embed[:, t])
            post_states.append(post)
            prior_states.append(prior)
            prev_state = post
        return self._stack_states(post_states), self._stack_states(prior_states)

    def imagine(self, action: torch.Tensor, state: RSSMState | None = None) -> RSSMState:
        batch_size, horizon, _ = action.shape
        if state is None:
            state = self.initial(batch_size, device=action.device, dtype=action.dtype)
        states: list[RSSMState] = []
        prev_state = state
        for t in range(horizon):
            prev_state = self.img_step(prev_state, action[:, t])
            states.append(prev_state)
        return self._stack_states(states)

    @staticmethod
    def _stack_states(states: list[RSSMState]) -> RSSMState:
        return RSSMState(
            mean=torch.stack([state.mean for state in states], dim=1),
            std=torch.stack([state.std for state in states], dim=1),
            stoch=torch.stack([state.stoch for state in states], dim=1),
            deter=torch.stack([state.deter for state in states], dim=1),
        )


class RSSMWorldModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        action_dim: int = 7,
        latent_dim: int = 64,
        projection_hidden_dim: int = 128,
        dynamics_hidden_dim: int = 128,
        reward_hidden_dim: int = 128,
        reward_loss_coef: float = 0.1,
        kl_loss_coef: float = 0.1,
        training: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.embed_dim = int(projection_hidden_dim)
        self.deter_dim = int(dynamics_hidden_dim)
        self.reward_loss_coef = float(reward_loss_coef)
        self.use_reward_loss = self.reward_loss_coef > 0
        self.kl_loss_coef = float(kl_loss_coef)

        self.encoder_projection = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, self.deter_dim),
        )
        self.rssm = RSSMCore(
            stoch_dim=self.latent_dim,
            deter_dim=self.deter_dim,
            hidden_dim=self.deter_dim,
            action_dim=self.action_dim,
        )
        self.transition_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.deter_dim),
            nn.Linear(self.latent_dim + self.deter_dim, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, self.obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.deter_dim + self.action_dim + self.latent_dim + self.deter_dim),
            nn.Linear(self.latent_dim + self.deter_dim + self.action_dim + self.latent_dim + self.deter_dim, reward_hidden_dim),
            nn.ELU(),
            nn.Linear(reward_hidden_dim, 1),
        )

    def _reduce_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 2:
            return hidden
        if hidden.ndim == 3:
            return hidden.mean(dim=1)
        raise ValueError(f"Unsupported hidden shape: {tuple(hidden.shape)}")

    @staticmethod
    def _apply_action_mask(actions: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return actions
        if actions.ndim != 3:
            return actions
        mask = action_mask.to(device=actions.device, dtype=actions.dtype).unsqueeze(-1)
        return actions * mask

    def encode_latent(self, hidden: torch.Tensor) -> RSSMState:
        embed = self.encoder_projection(self._reduce_hidden(hidden))
        batch_size = embed.shape[0]
        state = self.rssm.initial(batch_size, device=embed.device, dtype=embed.dtype)
        state.deter = embed
        prior_h = F.elu(self.rssm.img_hidden(embed))
        stats = self.rssm.img_stats(prior_h)
        return self.rssm._stats_to_state(stats, embed)

    def predict_next(self, latent: RSSMState, actions: torch.Tensor) -> RSSMState:
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        imagined = self.rssm.imagine(actions, state=latent)
        return RSSMState(
            mean=imagined.mean[:, -1],
            std=imagined.std[:, -1],
            stoch=imagined.stoch[:, -1],
            deter=imagined.deter[:, -1],
        )

    @staticmethod
    def _gaussian_kl_divergence(post: RSSMState, prior: RSSMState) -> torch.Tensor:
        post_var = post.std.pow(2)
        prior_var = prior.std.pow(2)
        log_ratio = torch.log(prior.std) - torch.log(post.std)
        sq_term = (post_var + (post.mean - prior.mean).pow(2)) / (2.0 * prior_var.clamp_min(1e-6))
        kl = log_ratio + sq_term - 0.5
        return kl.sum(dim=-1).mean()

    def reward(
        self,
        latent: RSSMState,
        actions: torch.Tensor,
        next_latent: RSSMState,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if actions.ndim == 3:
            actions = actions.mean(dim=1)
        features = torch.cat(
            [
                latent.feature(),
                actions,
                next_latent.feature(),
            ],
            dim=-1,
        )
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
        next_hidden_target = self._reduce_hidden(next_hidden)
        next_hidden_embed = self.encoder_projection(next_hidden_target)
        next_latent = self.predict_next(latent, action)
        posterior_next_latent = self.rssm.posterior_from_prior(next_latent, next_hidden_embed)
        predicted_next_hidden = self.transition_head(next_latent.feature())
        transition_loss = F.mse_loss(predicted_next_hidden, next_hidden_target)
        kl_loss = self._gaussian_kl_divergence(posterior_next_latent, next_latent)

        predicted_reward = self.reward(latent, action, next_latent)
        if self.use_reward_loss:
            if reward_target is None:
                reward_target = torch.zeros_like(predicted_reward).unsqueeze(-1)
            reward_target = reward_target.reshape_as(predicted_reward)
            reward_loss = F.mse_loss(predicted_reward, reward_target)
            loss = transition_loss + self.kl_loss_coef * kl_loss + self.reward_loss_coef * reward_loss
        else:
            reward_loss = transition_loss.new_zeros(())
            loss = transition_loss + self.kl_loss_coef * kl_loss
        return {
            "loss": loss,
            "transition_loss": transition_loss,
            "kl_loss": kl_loss,
            "reward_loss": reward_loss,
            "predicted_reward_mean": predicted_reward.mean() if self.use_reward_loss else transition_loss.new_zeros(()),
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


__all__ = ["RSSMState", "RSSMWorldModel"]
