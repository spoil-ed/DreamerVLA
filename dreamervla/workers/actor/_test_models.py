"""Tiny trainable actor models for Ray learner e2e tests."""

from __future__ import annotations

from types import SimpleNamespace

import ray
import torch
from torch import nn


class TinyTrainablePolicy(nn.Linear):
    """Linear policy with a stable constructor signature for tests."""

    def __init__(self, hidden_dim: int = 4, action_dim: int = 7) -> None:
        super().__init__(int(hidden_dim), int(action_dim))
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        return self(hidden.float())


class TinySharedPolicy(TinyTrainablePolicy):
    """Policy usable by both LearnerWorker and InferenceWorker."""

    def forward(self, batch):  # type: ignore[override]
        if isinstance(batch, dict):
            hidden = batch["hidden"].float()
            action = super().forward(hidden).unsqueeze(1)
            return action, None, None
        return super().forward(batch.float())


class TinyCheckpointPolicy(TinyTrainablePolicy):
    """Policy exposing the gradient-checkpointing hook used by FSDP tests."""

    def __init__(self, hidden_dim: int = 4, action_dim: int = 7) -> None:
        super().__init__(hidden_dim=hidden_dim, action_dim=action_dim)
        self.register_buffer("checkpoint_flag", torch.zeros((), dtype=torch.long))

    def gradient_checkpointing_enable(self) -> None:
        self.checkpoint_flag.fill_(1)


class TinyScalarModel(nn.Module):
    """Small trainable component for phase-updater tests."""

    def __init__(self, hidden_dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(int(hidden_dim), 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden.float()).squeeze(-1)


class TinyTrainableWorldModel(TinyScalarModel):
    """Small trainable world-model stand-in for learner routing tests."""


class TinyValueCritic(TinyScalarModel):
    """Small trainable critic stand-in for learner routing tests."""


class TinySuccessClassifier(nn.Module):
    """Tiny classifier with the cfg attributes used by online classifier updates."""

    def __init__(self, hidden_dim: int = 4, window: int = 3) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(
            window=int(window),
            chunk_size=1,
            chunk_pool="last",
        )
        self.linear = nn.Linear(int(hidden_dim), 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        hidden = windows.float().mean(dim=1)
        return self.linear(hidden)


class TinyWorldModelPhaseUpdater:
    """Configurable phase updater matching LearnerWorker's real-update boundary."""

    def update(
        self,
        *,
        phase: str,
        num_steps: int,
        modules: dict[str, nn.Module],
        optimizers: dict[str, torch.optim.Optimizer],
        replay,
        device: torch.device,
        train_cfg: dict,
        precision,
    ) -> dict[str, float]:
        if phase != "wm":
            return {f"train/{phase}_loss": 0.0}
        world_model = modules["world_model"]
        optimizer = optimizers["world_model"]
        batch_size = int(train_cfg.get("batch_size", 2))
        last_loss = 0.0
        for _ in range(int(num_steps)):
            batch = ray.get(replay.sample.remote(batch_size))
            hidden = batch["obs_embedding"].to(device).float().mean(dim=1)
            target = batch["current_actions"].to(device).float().mean(dim=(1, 2))
            optimizer.zero_grad(set_to_none=True)
            with precision.context():
                pred = world_model(hidden)
                loss = torch.mean((pred - target) ** 2)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
        return {"train/wm_loss": last_loss}
