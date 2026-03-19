from __future__ import annotations

import math

import torch
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


class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dim, 2 * action_dim)
        self.action_dim = action_dim

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.net(features)
        mean, log_std = stats.chunk(2, dim=-1)
        return mean, log_std.clamp(min=-5.0, max=1.0)

    def sample(self, features: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(features)
        std = log_std.exp()
        if deterministic:
            raw_action = mean
        else:
            raw_action = mean + torch.randn_like(std) * std
        action = torch.tanh(raw_action)
        entropy = 0.5 * torch.log(2.0 * math.pi * math.e * std.pow(2) + 1e-8).sum(dim=-1)
        return action, entropy


class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class ActorCriticPlanner(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, actor_hidden_dim: int, critic_hidden_dim: int) -> None:
        super().__init__()
        self.actor = ActorNetwork(feature_dim, action_dim, hidden_dim=actor_hidden_dim)
        self.critic = CriticNetwork(feature_dim, hidden_dim=critic_hidden_dim)

    def sample_action(self, features: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor.sample(features, deterministic=deterministic)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic(features)
