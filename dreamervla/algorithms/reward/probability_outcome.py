"""Continuous verifier-probability outcome reward."""

from __future__ import annotations

import torch

from dreamervla.algorithms.ppo.outcome import _validate_reward_inputs
from dreamervla.algorithms.reward.registry import register_reward_model


class ProbabilityOutcomeReward:
    """Place verifier ``p(success)`` at its best-scoring window end.

    Sparse threshold outcomes are useful for final success accounting, but during
    cold-start PPO a GRPO group can easily be all below or all above threshold.
    In that case the sparse return has zero within-group variance and the actor
    update is correctly skipped. This reward keeps the same verifier and LUMOS
    rollout path while using the continuous probability score as the return.
    """

    name = "probability_outcome"

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
        _validate_reward_inputs(
            batch=batch,
            max_steps=max_steps,
            chunk_size=chunk_size,
            finish_step=finish_step,
            complete=complete,
            score=score,
            score_step=score_step,
        )
        del chunk_size, finish_step
        reward = torch.zeros((batch, max_steps), dtype=torch.float32, device=device)

        if score is None:
            values = complete.detach().float().to(device)
        else:
            values = score.detach().float().to(device).clamp_(0.0, 1.0)
        if score_step is None:
            steps = torch.full((batch,), max_steps - 1, dtype=torch.long, device=device)
        else:
            steps = score_step.detach().long().to(device).clamp_(min=0, max=max_steps - 1)
        reward.scatter_(1, steps.unsqueeze(1), values.unsqueeze(1))
        return reward


register_reward_model(
    ProbabilityOutcomeReward(),
    aliases=("prob_outcome", "probability", "success_probability"),
)
