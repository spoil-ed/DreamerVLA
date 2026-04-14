from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
from torch import nn
from torch.distributions import Normal


class SharedObservationEmbedding:
    def __init__(self, encoder: nn.Module) -> None:
        self.encoder = encoder

    def embed_observation(self, obs: Mapping[str, Any]) -> torch.Tensor:
        return self.encoder.encode(obs)


class VLAPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 128,
        policy_head_hidden_dim: int = 128,
        initial_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.policy_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(policy_head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(policy_head_hidden_dim), self.action_dim),
        )
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(initial_log_std)))
        self.embedding: SharedObservationEmbedding | None = None

    def _reduce_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 2:
            return hidden
        if hidden.ndim == 3:
            return hidden.mean(dim=1)
        raise ValueError(f"Unsupported hidden shape: {tuple(hidden.shape)}")

    def encode(self, obs: Mapping[str, Any]) -> torch.Tensor:
        if self.embedding is not None:
            return self.embedding.embed_observation(obs)

        obs_embedding = obs.get("obs_embedding")
        if isinstance(obs_embedding, torch.Tensor) and obs_embedding.numel() > 0:
            return self._reduce_hidden(obs_embedding)

        proprio = obs.get("proprio")
        if isinstance(proprio, torch.Tensor) and proprio.numel() > 0:
            return self._reduce_hidden(proprio)

        state = obs.get("state")
        if isinstance(state, torch.Tensor) and state.numel() > 0:
            return self._reduce_hidden(state)

        image = obs.get("image")
        if isinstance(image, torch.Tensor) and image.numel() > 0:
            return image.flatten(start_dim=1)

        batch_size = 1
        task_id = obs.get("task_id")
        if isinstance(task_id, torch.Tensor) and task_id.ndim >= 1:
            batch_size = int(task_id.shape[0])
        return torch.zeros(batch_size, self.hidden_dim, device=self.log_std.device, dtype=torch.float32)

    def _distribution_from_hidden(self, hidden: torch.Tensor) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        hidden = self._reduce_hidden(hidden).float()
        mean = self.policy_head(hidden)
        log_std = self.log_std.clamp(min=self.min_log_std, max=self.max_log_std).unsqueeze(0).expand_as(mean)
        std = log_std.exp()
        return Normal(mean, std), mean, std

    def sample_action(
        self,
        obs: Mapping[str, Any],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.encode(obs)
        return self.sample_action_from_embedding(hidden, deterministic=deterministic)

    def sample_action_from_embedding(
        self,
        hidden: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        dist, mean, std = self._distribution_from_hidden(hidden)
        action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, {"mean": mean, "std": std}

    def evaluate_action(
        self,
        obs: Mapping[str, Any],
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.encode(obs)
        return self.evaluate_action_from_embedding(hidden, action)

    def evaluate_action_from_embedding(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        dist, mean, std = self._distribution_from_hidden(hidden)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, {"mean": mean, "std": std}

    def snapshot_state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.state_dict())

    def load_snapshot_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.load_state_dict(dict(state_dict))


__all__ = ["SharedObservationEmbedding", "VLAPolicy"]
