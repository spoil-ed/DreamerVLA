"""TD-MPC actor-critic helpers used as an optional terminal-value bootstrap
on top of LUMOS/PPO over imagined trajectories.

The state-action variant feeds (z, a) into a critic to predict Q(z, a); the
plain ``state`` variant predicts V(z). Both share the same underlying critic
hidden derived from the world model's ``critic_input`` mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from dreamervla.algorithms.dreamervla import _world_model_critic_input


def _sequence_field(
    obs: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Pull a 2D ``[B, T]`` sequence from ``obs`` by trying each candidate key."""
    for key in keys:
        value = obs.get(key)
        if isinstance(value, torch.Tensor):
            tensor = value
            if tensor.ndim == 3 and tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
            if tensor.ndim == 1:
                tensor = tensor[:, None]
            if tensor.ndim != 2:
                raise ValueError(f"obs.{key} must be [B,T] or [B,T,1], got {tuple(value.shape)}")
            return tensor.to(device=device, dtype=dtype)
    return None


def _tdmpc_value_mode(tdmpc_ac_cfg: Mapping[str, Any]) -> str:
    value_mode = str(tdmpc_ac_cfg.get("value_mode", "state")).lower()
    if value_mode in {"state", "v", "v_z"}:
        return "state"
    if value_mode in {"state_action", "q", "q_za", "q(z,a)"}:
        return "state_action"
    raise ValueError(f"Unsupported algorithm.tdmpc_ac.value_mode: {value_mode!r}")


def _tdmpc_action_dim(tdmpc_ac_cfg: Mapping[str, Any], fallback: int) -> int:
    return int(tdmpc_ac_cfg.get("action_dim", fallback))


def _tdmpc_prepare_action(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    action = action.float()
    if action.ndim > 3:
        action = action.reshape(action.shape[0], -1)
    if action.ndim not in {2, 3}:
        raise ValueError(
            f"TD-MPC state-action critic expects action [B,A] or [B,T,A], got {tuple(action.shape)}"
        )
    if int(action.shape[-1]) < int(action_dim):
        raise ValueError(
            f"TD-MPC state-action critic action dim {action.shape[-1]} < configured {action_dim}"
        )
    return action[..., : int(action_dim)]


def _tdmpc_critic_hidden(
    world_model: nn.Module,
    latent: Any,
    action: torch.Tensor | None,
    *,
    value_mode: str,
    action_dim: int,
) -> torch.Tensor:
    feat = _world_model_critic_input(world_model, latent).detach().float()
    if value_mode == "state":
        return feat
    if action is None:
        raise ValueError("TD-MPC state-action critic requires an action tensor.")
    action_feat = _tdmpc_prepare_action(action, action_dim).to(device=feat.device, dtype=feat.dtype)
    if feat.ndim == 2:
        if action_feat.shape[0] != feat.shape[0]:
            raise ValueError(
                f"Action batch {action_feat.shape[0]} does not match critic feature batch {feat.shape[0]}"
            )
        return torch.cat([feat, action_feat], dim=-1)
    if feat.ndim == 3:
        if action_feat.ndim != 3:
            raise ValueError(
                "TD-MPC state-action replay critic expects sequence actions [B,T,A], "
                f"got {tuple(action.shape)}"
            )
        if action_feat.shape[:2] != feat.shape[:2]:
            raise ValueError(
                f"Action sequence shape {tuple(action_feat.shape[:2])} does not match "
                f"critic feature shape {tuple(feat.shape[:2])}"
            )
        return torch.cat([feat, action_feat], dim=-1)
    raise ValueError(f"TD-MPC critic feature must be [B,D] or [B,T,D], got {tuple(feat.shape)}")


__all__ = [
    "_sequence_field",
    "_tdmpc_value_mode",
    "_tdmpc_action_dim",
    "_tdmpc_prepare_action",
    "_tdmpc_critic_hidden",
]
