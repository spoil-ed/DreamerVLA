"""Tiny trainable actor models for Ray learner e2e tests."""

from __future__ import annotations

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
