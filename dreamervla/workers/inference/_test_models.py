"""Tiny importable torch modules for InferenceWorker e2e tests."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TinyEncoder(nn.Module):
    def encode_obs_batch(self, obs_batch: list[dict[str, Any]]) -> torch.Tensor:
        rows = [
            [
                float(obs.get("step", 0)),
                float(obs.get("env_id", 0)),
                float(bool(obs.get("is_first", False))),
                1.0,
            ]
            for obs in obs_batch
        ]
        return torch.tensor(rows, dtype=torch.float32)


class TinyWorldModel(nn.Module):
    def __init__(self, hidden_dim: int = 4, action_dim: int = 7) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.bias = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        mode = str(batch["mode"])
        if mode == "encode_latent":
            return batch["hidden"].float() + self.bias
        if mode == "observe_next":
            hidden = batch["hidden"].float()
            latent = batch["latent"].float()
            actions = batch["actions"].float()
            action_signal = actions.mean(dim=-1, keepdim=True)
            return hidden + 0.1 * latent + 0.01 * action_signal + self.bias
        if mode == "actor_input":
            return batch["latent"].float()
        raise ValueError(f"unknown mode {mode!r}")


class TinyDictWorldModel(TinyWorldModel):
    """World model returning dict latents like production RSSM-style models."""

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor] | torch.Tensor:
        mode = str(batch["mode"])
        if mode == "encode_latent":
            hidden = batch["hidden"].float() + self.bias
            return {"hidden": hidden, "carry": hidden * 0.5}
        if mode == "observe_next":
            hidden = batch["hidden"].float()
            latent = batch["latent"]
            actions = batch["actions"].float()
            action_signal = actions.mean(dim=-1, keepdim=True)
            next_hidden = hidden + 0.1 * latent["hidden"] + 0.01 * action_signal + self.bias
            return {"hidden": next_hidden, "carry": latent["carry"] + 1.0}
        if mode == "actor_input":
            latent = batch["latent"]
            return latent["hidden"] + 0.01 * latent["carry"]
        raise ValueError(f"unknown mode {mode!r}")


class TinyPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 4, action_dim: int = 7) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.bias = nn.Parameter(torch.zeros(self.action_dim))

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, None, None]:
        hidden = batch["hidden"].float()
        base = hidden.sum(dim=-1, keepdim=True).repeat(1, self.action_dim)
        return base.unsqueeze(1) + self.bias.view(1, 1, -1), None, None
