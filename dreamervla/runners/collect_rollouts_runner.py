from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.collect_parallel_rollouts import (
    _get_dist_info,
    collect_rollouts,
)
from dreamervla.runners.oft_collect_common import (
    resolve_num_images_in_input,
    select_vla_image_keys,
)


class CollectRolloutsRunner(BaseRunner):
    """Pure-Hydra cold-start rollout collector.

    Reads the single source of truth from ``task.openvla_oft.*`` (ckpt, dirs,
    expected_*) plus ``collect.*`` knobs and drives collect_rollouts.  Uses
    torchrun RANK/WORLD_SIZE/LOCAL_RANK for Layer-1 work sharding only — it does
    NOT initialize a torch process group (no DDP).
    """

    runner_name = "collect_rollouts"
    runner_status = "current"
    runner_family = "rollout"

    def setup(self) -> None:
        # Only rank 0 writes resolved_config.yaml / run_manifest.json (no DDP =>
        # is_main_process is True on every rank, so guard on RANK explicitly).
        if int(os.environ.get("RANK", 0)) == 0:
            self.write_run_artifacts()

    def _build_collect_cfg(self) -> dict[str, Any]:
        cfg = self.cfg
        oft = cfg.task.openvla_oft
        expected_history = int(oft.expected_history)
        num_images_in_input = resolve_num_images_in_input(cfg.collect)
        image_keys = select_vla_image_keys(
            list(cfg.task.image_keys),
            history=expected_history,
            num_images_in_input=num_images_in_input,
        )
        task_ids = cfg.collect.task_ids
        if OmegaConf.is_config(task_ids):
            task_ids = OmegaConf.to_container(task_ids, resolve=True)
        return {
            "model_path": str(oft.ckpt_path),
            "policy_mode": str(OmegaConf.select(cfg, "collect.policy_mode", default="auto")),
            "unnorm_key": str(oft.dataset_statistics_key),
            "task_suite_name": str(cfg.task.suite),
            "task_ids": task_ids,
            "episodes_per_task": int(cfg.collect.episodes_per_task),
            "num_tasks": OmegaConf.select(cfg, "collect.num_tasks", default=None),
            "episode_horizon": int(cfg.collect.episode_horizon),
            "envs_per_gpu": int(cfg.collect.envs_per_gpu),
            "demos_per_shard": int(OmegaConf.select(cfg, "collect.demos_per_shard", default=0)),
            "memory_fraction": float(cfg.collect.memory_fraction),
            "reward_dir": str(oft.hdf5_reward_dir),
            "hidden_dir": str(oft.action_hidden_dir),
            "image_keys": image_keys,
            "expected_history": expected_history,
            "num_images_in_input": num_images_in_input,
            "expected_action_head_type": str(oft.expected_action_head_type),
            "expected_include_state": bool(oft.expected_include_state),
            "expected_obs_hidden_source": str(oft.expected_obs_hidden_source),
            "expected_prompt_style": str(oft.expected_prompt_style),
            "expected_rotate_images_180": bool(oft.expected_rotate_images_180),
            "time_horizon": int(oft.time_horizon),
            "token_dim": int(oft.token_dim),
            "action_dim": int(cfg.task.action_dim),
            "chunk_size": int(oft.chunk_size),
            "resolution": int(cfg.task.image_resolution),
            "gpu_id": int(OmegaConf.select(cfg, "collect.gpu_id", default=0)),
            "min_free_gpu_gb": float(OmegaConf.select(cfg, "collect.min_free_gpu_gb", default=18.0)),
            # Shared (across ranks) dir for cross-process progress aggregation: rank 0
            # renders ONE global bar by summing per-rank files here. training.out_dir is
            # identical on every torchrun rank when launched via the e2e (fixed timestamp),
            # so the ranks agree on this path; a direct run with a ${now} out_dir degrades
            # to a per-rank bar.
            "progress_dir": str(
                Path(str(OmegaConf.select(cfg, "training.out_dir", default="."))) / ".progress"
            ),
        }

    def run(self) -> object:
        rank, world_size, local_rank = _get_dist_info()
        cfg = self._build_collect_cfg()
        task_suite = cfg.get("task_suite_name", "")
        self.console_banner("COLLECT ROLLOUTS", subtitle=f"suite={task_suite} rank={rank}/{world_size}")

        successes: list[bool] = []

        def _on_episode(task_id: int, episode_id: int, n_steps: int, success: bool) -> None:
            successes.append(success)
            self.console_record_success(success)

        demos_written = collect_rollouts(cfg, rank, world_size, local_rank, on_episode=_on_episode)

        succ_rate = sum(successes) / len(successes) if successes else 0.0
        self.console_metrics(
            "collect",
            {
                "collect/episodes": len(successes),
                "collect/success_rate": succ_rate,
            },
            force=True,
        )
        self.console_banner(
            "COLLECT ROLLOUTS",
            done=True,
            subtitle=f"{demos_written} episodes · succ {succ_rate:.3f}",
        )
        return demos_written
