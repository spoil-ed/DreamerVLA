"""Default sparse outcome reward: float(complete) at finish_step, else 0."""

from __future__ import annotations

import torch

from dreamervla.algorithms.ppo.outcome import _build_reward_tensor
from dreamervla.algorithms.reward.registry import register_reward_model


class SparseOutcomeReward:
    """Wraps the canonical ``_build_reward_tensor`` so the default WMPO numerics are
    bit-for-bit unchanged; exists so the reward DEFINITION is selectable via
    ``algorithm.wmpo.reward_model`` alongside future dense / verifier-shaped forms.
    """

    name = "sparse_outcome"

    def build_reward(
        self,
        *,
        batch: int,
        max_steps: int,
        chunk_size: int,
        finish_step: torch.Tensor,
        complete: torch.Tensor,
        device: torch.device,
        score: torch.Tensor | None = None,
        score_step: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del score, score_step
        return _build_reward_tensor(
            batch=batch,
            max_steps=max_steps,
            chunk_size=chunk_size,
            finish_step=finish_step,
            complete=complete,
        ).to(device)


register_reward_model(SparseOutcomeReward(), aliases=("outcome", "sparse"))
