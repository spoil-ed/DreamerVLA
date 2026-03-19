from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class LatentState:
    latent: torch.Tensor

    def detach(self) -> "LatentState":
        return LatentState(latent=self.latent.detach())


@dataclass
class ObservationRollout:
    latents: torch.Tensor
    next_latent_pred: torch.Tensor
    reward_pred: torch.Tensor
    continue_logit: torch.Tensor


@dataclass
class ImagineRollout:
    latents: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor
    entropies: torch.Tensor


class WorldModelAdapter(nn.Module):
    """Simple deterministic world model.

    It uses:
    - a dynamics MLP: `[z_t, a_t] -> z_{t+1}`
    - a reward head: `z_t -> r_t`
    - a continue head: `z_t -> c_t`
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        action_dim: int = 6,
        reward_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.dynamics = _build_mlp(latent_dim + action_dim, hidden_dim, latent_dim)
        self.reward_head = _build_mlp(latent_dim, reward_hidden_dim, 1)
        self.continue_head = _build_mlp(latent_dim, reward_hidden_dim, 1)

    def initial(self, batch_size: int, device: torch.device | str) -> LatentState:
        return LatentState(latent=torch.zeros(batch_size, self.latent_dim, device=device))

    def feature(self, state: LatentState) -> torch.Tensor:
        return state.latent

    def transition(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.dynamics(torch.cat([latent, action], dim=-1))
        return latent + delta

    def observe(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        done: torch.Tensor,
    ) -> ObservationRollout:
        reward_pred = self.reward_head(latents).squeeze(-1)
        continue_logit = self.continue_head(latents).squeeze(-1)
        next_latent_pred = self.transition(latents[:, :-1], actions[:, :-1])
        return ObservationRollout(
            latents=latents,
            next_latent_pred=next_latent_pred,
            reward_pred=reward_pred,
            continue_logit=continue_logit,
        )

    def imagine_step(
        self,
        state: LatentState,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> LatentState:
        del deterministic
        next_latent = self.transition(state.latent, action)
        return LatentState(latent=next_latent)

    def imagine(
        self,
        start_state: LatentState,
        actor: nn.Module,
        horizon: int,
        deterministic: bool = False,
    ) -> ImagineRollout:
        state = start_state
        latents = []
        actions = []
        rewards = []
        continues = []
        entropies = []

        for _ in range(horizon):
            feature = self.feature(state)
            action, entropy = actor.sample(feature, deterministic=deterministic)
            state = self.imagine_step(state, action, deterministic=deterministic)
            next_latent = self.feature(state)

            latents.append(next_latent)
            actions.append(action)
            rewards.append(self.reward_head(next_latent).squeeze(-1))
            continues.append(torch.sigmoid(self.continue_head(next_latent).squeeze(-1)))
            entropies.append(entropy)

        return ImagineRollout(
            latents=torch.stack(latents, dim=1),
            actions=torch.stack(actions, dim=1),
            rewards=torch.stack(rewards, dim=1),
            continues=torch.stack(continues, dim=1),
            entropies=torch.stack(entropies, dim=1),
        )

    def continue_loss(self, continue_logit: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        continue_target = 1.0 - done.float()
        return F.binary_cross_entropy_with_logits(continue_logit, continue_target)
