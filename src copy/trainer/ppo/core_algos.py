"""Minimal optimization utilities for Dreamer-VLA."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FixedKLController:
    kl_coef: float = 0.0

    @property
    def value(self) -> float:
        return self.kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        del current_kl, n_steps


@dataclass
class AdaptiveKLController:
    init_kl_coef: float = 0.0
    target_kl: float = 1.0
    horizon: int = 1000

    def __post_init__(self) -> None:
        self.value = self.init_kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        proportional_error = (current_kl - self.target_kl) / max(self.target_kl, 1e-6)
        multiplier = 1.0 + proportional_error * n_steps / max(self.horizon, 1)
        self.value = max(0.0, self.value * multiplier)


def kl_penalty(old_log_probs: torch.Tensor, ref_log_prob: torch.Tensor, kl_penalty: str = "kl") -> torch.Tensor:
    delta = old_log_probs - ref_log_prob
    if kl_penalty == "abs":
        return delta.abs()
    if kl_penalty == "mse":
        return delta.pow(2)
    return delta


def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    horizon = rewards.shape[1]
    returns = []
    running = bootstrap
    for step in reversed(range(horizon)):
        next_value = bootstrap if step == horizon - 1 else values[:, step + 1]
        target = rewards[:, step] + gamma * continues[:, step] * ((1.0 - lam) * next_value + lam * running)
        returns.append(target)
        running = target
    return torch.stack(list(reversed(returns)), dim=1)
