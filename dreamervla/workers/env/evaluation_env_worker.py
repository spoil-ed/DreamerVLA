"""Resident CPU LIBERO evaluation environment worker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dreamervla.workers.env.trajectory_env_worker import BaseTrajectoryEnvWorker


class EvaluationEnvironmentWorker(BaseTrajectoryEnvWorker):
    """Evaluate the resident rollout policy without replay or optimizer writes."""

    role_name = "eval_env"

    def __init__(
        self,
        env_cfg: Mapping[str, Any],
        num_slots: int,
        rollout_epoch: int,
        max_steps_per_rollout_epoch: int,
        num_action_chunks: int,
        *,
        task_ids: Sequence[int],
    ) -> None:
        cfg = dict(env_cfg)
        cfg["one_trajectory_per_rollout_epoch"] = True
        cfg["emit_actor_trajectories"] = False
        super().__init__(
            self.role_name,
            cfg,
            num_slots,
            rollout_epoch,
            max_steps_per_rollout_epoch,
            num_action_chunks,
            task_id=int(task_ids[0]),
            task_ids=task_ids,
            replay=None,
            dump=None,
            rank_offset=0,
            request_final_bootstrap=False,
            replay_write_enabled=False,
        )


__all__ = ["EvaluationEnvironmentWorker"]
