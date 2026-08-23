from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

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
        num_layers: int = 1,
        act: str = "gelu",
        initial_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        policy_head_hidden_dim = int(policy_head_hidden_dim)
        num_layers = int(num_layers)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        initial_log_std = float(initial_log_std)

        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim!r}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim!r}")
        if policy_head_hidden_dim <= 0:
            raise ValueError(f"policy_head_hidden_dim must be > 0, got {policy_head_hidden_dim!r}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers!r}")
        for name, value in (
            ("initial_log_std", initial_log_std),
            ("min_log_std", self.min_log_std),
            ("max_log_std", self.max_log_std),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.min_log_std > self.max_log_std:
            raise ValueError(
                "min_log_std must be <= max_log_std, got "
                f"{self.min_log_std!r} > {self.max_log_std!r}"
            )

        activation_name = str(act).lower()
        activation_types: dict[str, type[nn.Module]] = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "swish": nn.SiLU,
        }
        if activation_name not in activation_types:
            raise ValueError(f"act must be one of {sorted(activation_types)}, got {act!r}")
        layers: list[nn.Module] = []
        cur = self.hidden_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.LayerNorm(cur),
                    nn.Linear(cur, policy_head_hidden_dim),
                    activation_types[activation_name](),
                ]
            )
            cur = policy_head_hidden_dim
        layers.append(nn.Linear(cur, self.action_dim))
        self.policy_head = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), initial_log_std))
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

        raise KeyError(
            "VLAPolicy observation must contain a non-empty Tensor under one of "
            "'obs_embedding', 'proprio', 'state', or 'image'."
        )

    def _distribution_from_hidden(
        self, hidden: torch.Tensor
    ) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        # Match the param dtype: under FSDP MixedPrecision the gathered weights
        # are cast to bf16 inside the FSDP forward, so an fp32 input to a
        # LayerNorm holding bf16 weights triggers `expected Float, got BFloat16`.
        param_dtype = self.policy_head[0].weight.dtype
        hidden = self._reduce_hidden(hidden)
        if int(hidden.shape[-1]) != self.hidden_dim:
            raise ValueError(
                f"hidden feature dim must be {self.hidden_dim}, got {int(hidden.shape[-1])}"
            )
        hidden = hidden.to(dtype=param_dtype)
        mean = self.policy_head(hidden)
        log_std = (
            self.log_std.clamp(min=self.min_log_std, max=self.max_log_std)
            .unsqueeze(0)
            .expand_as(mean)
        )
        std = log_std.exp()
        # Distribution math is sensitive to precision — promote outputs to fp32.
        mean = mean.float()
        std = std.float()
        return Normal(mean, std), mean, std

    def forward(self, batch: dict) -> Any:
        """FSDP-compatible dispatcher: routes through __call__ so FSDP's
        all-gather hook fires before policy_head touches sharded params.

            policy({'mode': 'sample', 'hidden': h, 'deterministic': bool})
                -> (action, log_prob, {'mean', 'std'})
            policy({'mode': 'evaluate', 'hidden': h, 'action': a})
                -> (log_prob, entropy, {'mean', 'std'})
        """
        mode = batch.get("mode")
        if mode == "sample":
            return self.sample_action_from_embedding(
                batch["hidden"],
                deterministic=bool(batch.get("deterministic", False)),
            )
        if mode == "evaluate":
            return self.evaluate_action_from_embedding(batch["hidden"], batch["action"])
        raise ValueError(f"Unknown VLAPolicy forward mode: {mode!r}")

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
        if action.shape != mean.shape:
            raise ValueError(
                f"action shape must match policy mean {tuple(mean.shape)}, got {tuple(action.shape)}"
            )
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, {"mean": mean, "std": std}

    def snapshot_state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.state_dict())

    def load_snapshot_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.load_state_dict(dict(state_dict))


__all__ = ["SharedObservationEmbedding", "VLAPolicy"]
