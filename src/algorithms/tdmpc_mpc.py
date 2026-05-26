"""TD-MPC/TD-MPC2-style latent MPC planner for DreamerVLA eval.

This module is intentionally eval-only and standalone.  It does not replace the
existing Dreamer/VLA rollout path; callers opt in through config and pass the
loaded policy, world model, and optional target critic.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Callable

import torch
from torch import nn


ActionTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class TDMPCMPCConfig:
    horizon: int = 3
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    action_dim: int = 7
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5
    gamma: float = 0.995
    terminal_value_scale: float = 1.0
    reward_scale: float = 1.0
    value_mode: str = "state"
    execute_steps: int = 1
    eval_mode: bool = True
    warm_start: bool = True
    seed: int = 0


@dataclass
class TDMPCMPCResult:
    raw_actions: torch.Tensor
    best_value: torch.Tensor
    elite_value_mean: torch.Tensor
    elite_value_max: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


def _repeat_latent(value: Any, repeats: int) -> Any:
    repeats = int(repeats)
    if repeats <= 1:
        return value
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    if isinstance(value, dict):
        return {key: _repeat_latent(item, repeats) for key, item in value.items()}
    if is_dataclass(value):
        return replace(value, **{field.name: _repeat_latent(getattr(value, field.name), repeats) for field in fields(value)})
    raise TypeError(f"Unsupported latent type for repeat: {type(value).__name__}")


def _world_model_actor_input(world_model: nn.Module, latent: Any) -> torch.Tensor:
    try:
        return world_model({"mode": "actor_input", "latent": latent})
    except ValueError as exc:
        message = str(exc)
        if "actor_input" not in message and "Unknown" not in message:
            raise
    return latent.feature()


def _world_model_critic_input(world_model: nn.Module, latent: Any) -> torch.Tensor:
    try:
        return world_model({"mode": "critic_input", "latent": latent})
    except ValueError as exc:
        message = str(exc)
        if "critic_input" not in message and "Unknown" not in message:
            raise
    return _world_model_actor_input(world_model, latent)


def _world_model_reward(world_model: nn.Module, latent: Any) -> torch.Tensor:
    reward = world_model({"mode": "reward", "latent": latent})
    return reward.reshape(reward.shape[0], -1).mean(dim=-1)


def _critic_hidden(
    world_model: nn.Module,
    latent: Any,
    action: torch.Tensor | None,
    *,
    value_mode: str,
    action_dim: int,
) -> torch.Tensor:
    feat = _world_model_critic_input(world_model, latent).float()
    mode = str(value_mode).lower()
    if mode in {"state", "v", "v_z"}:
        return feat
    if mode not in {"state_action", "q", "q_za", "q(z,a)"}:
        raise ValueError(f"Unsupported TD-MPC MPC value_mode: {value_mode!r}")
    if action is None:
        raise ValueError("TD-MPC MPC state_action value mode requires an action.")
    action = action.float()
    if action.ndim != 2:
        action = action.reshape(action.shape[0], -1)
    action = action[..., : int(action_dim)].to(device=feat.device, dtype=feat.dtype)
    return torch.cat([feat, action], dim=-1)


class TDMPCMPCPlanner:
    """Latent-space MPPI/CEM planner following the TD-MPC2 evaluation loop."""

    def __init__(self, cfg: TDMPCMPCConfig) -> None:
        self.cfg = cfg
        self._prev_mean: torch.Tensor | None = None
        self._generator: torch.Generator | None = None

    def reset(self) -> None:
        self._prev_mean = None

    def _generator_for(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(int(self.cfg.seed))
        return self._generator

    def _sample_random_actions(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        count: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        noise = torch.randn(
            int(self.cfg.horizon),
            int(count),
            int(self.cfg.action_dim),
            device=mean.device,
            dtype=mean.dtype,
            generator=generator,
        )
        return (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1.0, 1.0)

    @torch.no_grad()
    def _policy_trajectories(
        self,
        policy: nn.Module,
        world_model: nn.Module,
        latent: Any,
        count: int,
        action_transform: ActionTransform,
    ) -> torch.Tensor:
        z = _repeat_latent(latent, int(count))
        actions = []
        for _ in range(int(self.cfg.horizon)):
            feat = _world_model_actor_input(world_model, z).float()
            action, _log_prob, extra = policy({
                "mode": "sample",
                "hidden": feat,
                "deterministic": bool(self.cfg.eval_mode),
                "return_chunk": False,
            })
            if isinstance(extra, dict) and isinstance(extra.get("mean"), torch.Tensor) and bool(self.cfg.eval_mode):
                action = extra["mean"]
            action = action[:, : int(self.cfg.action_dim)].float().clamp(-1.0, 1.0)
            actions.append(action)
            wm_action = action_transform(action)
            z = world_model({"mode": "predict_next", "latent": z, "actions": wm_action})
        return torch.stack(actions, dim=0)

    @torch.no_grad()
    def _estimate_value(
        self,
        policy: nn.Module,
        world_model: nn.Module,
        latent: Any,
        actions: torch.Tensor,
        action_transform: ActionTransform,
        target_critic: nn.Module | None,
    ) -> torch.Tensor:
        num_samples = int(actions.shape[1])
        z = _repeat_latent(latent, num_samples)
        total = torch.zeros(num_samples, device=actions.device, dtype=actions.dtype)
        discount = torch.ones_like(total)
        for step in range(int(self.cfg.horizon)):
            wm_action = action_transform(actions[step])
            z = world_model({"mode": "predict_next", "latent": z, "actions": wm_action})
            reward = _world_model_reward(world_model, z).to(dtype=total.dtype)
            total = total + discount * float(self.cfg.reward_scale) * reward
            discount = discount * float(self.cfg.gamma)
        if target_critic is not None and float(self.cfg.terminal_value_scale) != 0.0:
            terminal_action = None
            if str(self.cfg.value_mode).lower() in {"state_action", "q", "q_za", "q(z,a)"}:
                feat = _world_model_actor_input(world_model, z).float()
                terminal_action, _log_prob, extra = policy({
                    "mode": "sample",
                    "hidden": feat,
                    "deterministic": bool(self.cfg.eval_mode),
                    "return_chunk": False,
                })
                if isinstance(extra, dict) and isinstance(extra.get("mean"), torch.Tensor) and bool(self.cfg.eval_mode):
                    terminal_action = extra["mean"]
                terminal_action = action_transform(
                    terminal_action[:, : int(self.cfg.action_dim)].float().clamp(-1.0, 1.0)
                )
            critic_feat = _critic_hidden(
                world_model,
                z,
                terminal_action,
                value_mode=self.cfg.value_mode,
                action_dim=int(self.cfg.action_dim),
            )
            terminal = target_critic({"mode": "value", "hidden": critic_feat}).reshape(num_samples).to(dtype=total.dtype)
            total = total + discount * float(self.cfg.terminal_value_scale) * terminal
        return total

    @torch.no_grad()
    def plan(
        self,
        *,
        policy: nn.Module,
        world_model: nn.Module,
        latent: Any,
        device: torch.device,
        target_critic: nn.Module | None = None,
        action_transform: ActionTransform | None = None,
    ) -> TDMPCMPCResult:
        cfg = self.cfg
        horizon = max(1, int(cfg.horizon))
        action_dim = max(1, int(cfg.action_dim))
        num_samples = max(1, int(cfg.num_samples))
        num_pi_trajs = min(max(0, int(cfg.num_pi_trajs)), num_samples)
        num_random = num_samples - num_pi_trajs
        num_elites = min(max(1, int(cfg.num_elites)), num_samples)
        action_transform = action_transform or (lambda x: x)
        generator = self._generator_for(device)

        mean = torch.zeros(horizon, action_dim, device=device)
        if bool(cfg.warm_start) and self._prev_mean is not None and tuple(self._prev_mean.shape) == tuple(mean.shape):
            mean[:-1] = self._prev_mean[1:]
        std = torch.full_like(mean, float(cfg.max_std))
        actions = torch.empty(horizon, num_samples, action_dim, device=device)
        if num_pi_trajs > 0:
            actions[:, :num_pi_trajs] = self._policy_trajectories(
                policy=policy,
                world_model=world_model,
                latent=latent,
                count=num_pi_trajs,
                action_transform=action_transform,
            )

        value = torch.zeros(num_samples, device=device)
        elite_values = value
        elite_actions = actions[:, :num_elites]
        for _ in range(max(1, int(cfg.iterations))):
            if num_random > 0:
                actions[:, num_pi_trajs:] = self._sample_random_actions(mean, std, num_random, generator)
            value = self._estimate_value(
                policy=policy,
                world_model=world_model,
                latent=latent,
                actions=actions,
                action_transform=action_transform,
                target_critic=target_critic,
            ).nan_to_num(0.0)
            elite_idxs = torch.topk(value, k=num_elites, dim=0).indices
            elite_values = value[elite_idxs]
            elite_actions = actions[:, elite_idxs]
            max_value = elite_values.max()
            score = torch.exp(float(cfg.temperature) * (elite_values - max_value))
            score = score / score.sum().clamp_min(1.0e-9)
            mean = (score.view(1, num_elites, 1) * elite_actions).sum(dim=1)
            var = (score.view(1, num_elites, 1) * (elite_actions - mean.unsqueeze(1)).square()).sum(dim=1)
            std = var.sqrt().clamp(float(cfg.min_std), float(cfg.max_std))

        best_idx = torch.argmax(elite_values)
        best_actions = elite_actions[:, best_idx]
        execute_steps = min(max(1, int(cfg.execute_steps)), horizon)
        if bool(cfg.warm_start):
            self._prev_mean = mean.detach()
        return TDMPCMPCResult(
            raw_actions=best_actions[:execute_steps].detach(),
            best_value=elite_values[best_idx].reshape(1).detach(),
            elite_value_mean=elite_values.mean().reshape(1).detach(),
            elite_value_max=elite_values.max().reshape(1).detach(),
            mean=mean.detach(),
            std=std.detach(),
        )


__all__ = ["TDMPCMPCConfig", "TDMPCMPCPlanner", "TDMPCMPCResult"]
