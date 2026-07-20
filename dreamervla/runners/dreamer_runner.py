"""Frozen-imagination specialization of the shared Ray cotrain runner."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from omegaconf import DictConfig, OmegaConf

from dreamervla.runners.cotrain_runner import CotrainRunner
from dreamervla.workers.cotrain.placement import ManualCotrainPlacementPlan

_REAL_ROLLOUT_REQUEST_KEY = "real_env"


class DreamerRunner(CotrainRunner):
    """Train the actor from imagined rollouts with frozen WM and classifier."""

    runner_name = "dreamer"
    runner_status = "current"
    runner_family = "dreamer"

    def __init__(self, cfg: dict[str, Any] | DictConfig) -> None:
        config = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
        super().__init__(config)
        allowed_modes = {"failure_imagined_rl", "imagined_success_sft"}
        if self._training_mode() not in allowed_modes:
            raise ValueError(
                "DreamerRunner requires manual_cotrain.training_mode to be one of "
                f"{sorted(allowed_modes)}"
            )

    def _placement_plan(self) -> ManualCotrainPlacementPlan:
        plan = super()._placement_plan()
        if plan.learner_spec is None:
            raise ValueError("DreamerRunner requires a checkpoint-owning LearnerGroup")
        return replace(
            plan,
            learner_spec=replace(plan.learner_spec, gpu_ids=[]),
        )

    def _real_rollout_request_key(self) -> str:
        return _REAL_ROLLOUT_REQUEST_KEY

    @staticmethod
    def _wm_env_rank_offset(real_env_workers: int) -> int:
        del real_env_workers
        return 0

    def _real_rollout_total_chunks(self) -> int:
        worker_epochs = self._real_rollout_epochs_by_worker(self._real_env_workers())
        return (
            sum(worker_epochs)
            * self._real_envs_per_worker()
            * (self._real_max_steps_per_rollout_epoch() // self._num_action_chunks())
        )


__all__ = ["DreamerRunner"]
