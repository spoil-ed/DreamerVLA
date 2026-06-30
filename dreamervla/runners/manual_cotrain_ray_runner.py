"""Target manual-cotrain Ray runner.

This route follows ``spec/99_manual_notes.md``: LearnerGroup owns WM/classifier,
ActorGroup owns VLA PPO, RolloutGroup owns no-grad policy inference, and
EnvGroup owns real/WM env interaction.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.runners.base_runner import BaseRunner
from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import (
    NodePlacementStrategy,
    ResourceMapPlacementStrategy,
)
from dreamervla.scheduler.worker_group import WorkerGroup
from dreamervla.workers.actor.embodied_fsdp_actor import EmbodiedFSDPActor
from dreamervla.workers.actor.learner_worker import LearnerWorker
from dreamervla.workers.cotrain.handshake_trace import trace as _hs_trace
from dreamervla.workers.cotrain.messages import StopMsg
from dreamervla.workers.cotrain.placement import (
    ManualCotrainPlacementPlan,
    build_manual_cotrain_placement,
)
from dreamervla.workers.env.trajectory_env_worker import RealEnvWorker, WMEnvWorker
from dreamervla.workers.replay.replay_worker import ReplayWorker
from dreamervla.workers.rollout.multistep_rollout_worker import MultiStepRolloutWorker


class ManualCotrainRayRunner(BaseRunner):
    """Ray runner for the manual-notes cotrain target topology."""

    runner_name = "manual_cotrain_ray"
    runner_status = "current"
    runner_family = "actor"

    def __init__(self, cfg: dict[str, Any] | DictConfig) -> None:
        config = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
        super().__init__(config)
        self.history: dict[str, float | int] | None = None

    def setup(self) -> None:
        super().setup()

    def execute(self) -> dict[str, float | int]:
        self.history = self.run()
        return self.history

    def teardown(self) -> None:
        super().teardown()

    def run(self) -> dict[str, float | int]:
        cluster_cfg = OmegaConf.select(self.cfg, "cluster", default=None)
        cluster = Cluster(cluster_cfg)
        try:
            cluster.require_single_node()
            groups = self._build_groups(cluster)
            resume_payload = self._manual_resume_payload()
            resume_step = 0
            if resume_payload is not None:
                resume_step = int(resume_payload.get("global_step", 0))
                self._restore_manual_resume_state(groups, resume_payload)
            print(
                "[manual-cotrain] groups="
                + ",".join(self._target_group_names()),
                flush=True,
            )
            metrics: dict[str, float | int] = {}
            target_step = self._global_steps()
            last_step = resume_step
            for global_step in range(resume_step + 1, target_step + 1):
                step_metrics = self._run_global_step(groups, global_step)
                metrics.update(step_metrics)
                self.log_metrics(step_metrics, step=global_step)
                last_step = int(global_step)
            metrics["global_step"] = int(last_step)
            return metrics
        finally:
            cluster.shutdown()

    def _placement_plan(self) -> ManualCotrainPlacementPlan:
        return build_manual_cotrain_placement(self._ngpu())

    def _target_group_names(self) -> list[str]:
        return ["LearnerGroup", "ActorGroup", "RolloutGroup", "EnvGroup"]

    def _global_step_operation_names(self) -> list[str]:
        return [
            "set_global_step",
            "actor_to_rollout_sync",
            "env_interact_and_rollout_generate",
            "actor_recv_trajectories",
            "actor_compute_advantages_and_returns",
            "actor_run_training",
            "learner_update_wm_classifier",
            "learner_to_wm_env_sync",
            "checkpoint_and_metrics",
        ]

    def _build_groups(self, cluster: Cluster) -> dict[str, Any]:
        plan = self._placement_plan()
        env_channel_name = f"manual-cotrain-env-{uuid.uuid4().hex}"
        rollout_channel_name = f"manual-cotrain-rollout-{uuid.uuid4().hex}"
        actor_channel_name = f"manual-cotrain-actor-{uuid.uuid4().hex}"
        env_channel = Channel.create(env_channel_name)
        rollout_channel = Channel.create(rollout_channel_name)
        actor_channel = Channel.create(actor_channel_name)
        replay_group = None
        replay = None
        replay_cfg = OmegaConf.select(self.cfg, "replay.cfg", default=None)
        if replay_cfg is not None:
            replay_group = WorkerGroup(
                ReplayWorker,
                self._cfg_dict("replay.cfg"),
            ).launch(cluster, NodePlacementStrategy(1), name="ReplayWorker")
            replay = replay_group.workers[0]

        real_env_group = WorkerGroup(
            RealEnvWorker,
            self._real_env_cfg(),
            self._envs_per_worker(),
            self._rollout_epoch(),
            self._max_steps_per_rollout_epoch(),
            self._num_action_chunks(),
            task_id=self._task_id(),
            replay=replay,
            dump=None,
            rank_offset=0,
            request_final_bootstrap=self._requires_bootstrap_value(),
        ).launch(
            cluster,
            self._placement_for(plan.env_specs[0].gpu_ids),
            name="RealEnvWorker",
        )
        wm_gpus = [spec.gpu_ids[0] for spec in plan.env_specs if spec.role == "wm_env"]
        wm_env_group = None
        if wm_gpus:
            wm_env_group = WorkerGroup(
                WMEnvWorker,
                self._cfg_dict("env.wm.cfg"),
                self._envs_per_worker(),
                self._rollout_epoch(),
                self._wm_max_steps_per_rollout_epoch(),
                self._num_action_chunks(),
                task_id=self._task_id(),
                replay=replay,
                dump=None,
                rank_offset=1,
                request_final_bootstrap=self._requires_bootstrap_value(),
                replay_write_enabled=self._wm_env_write_replay(),
            ).launch(
                cluster,
                ResourceMapPlacementStrategy(",".join(str(gpu) for gpu in wm_gpus)),
                name="WMEnvWorker",
            )

        rollout_group = WorkerGroup(
            MultiStepRolloutWorker,
            self._cfg_dict("rollout.policy_cfg", "actor.policy_cfg"),
            self._optional_cfg_dict("rollout.encoder_cfg"),
            self._load_init_ckpt("rollout.init_ckpt"),
            self._cfg_dict("rollout.train_cfg"),
        ).launch(
            cluster,
            self._resource_map_for_specs(plan.rollout_specs),
            name="MultiStepRolloutWorker",
        )

        actor_group = WorkerGroup(
            EmbodiedFSDPActor,
            self._cfg_dict("actor.policy_cfg"),
            self._load_init_ckpt("actor.init_ckpt"),
            self._cfg_dict("actor.train_cfg"),
        ).launch(
            cluster,
            self._resource_map_for_specs(plan.actor_specs),
            name="EmbodiedFSDPActor",
        )

        learner_group = WorkerGroup(
            LearnerWorker,
            self._cfg_dict("learner.model_cfg"),
            self._load_init_ckpt("learner.init_ckpt"),
            self._cfg_dict("learner.train_cfg"),
            replay,
        ).launch(
            cluster,
            self._placement_for(plan.learner_spec.gpu_ids),
            name="LearnerWorker",
        )

        return {
            "LearnerGroup": learner_group,
            "ActorGroup": actor_group,
            "RolloutGroup": rollout_group,
            "ReplayGroup": replay_group,
            "replay": replay,
            "RealEnvGroup": real_env_group,
            "WMEnvGroup": wm_env_group,
            "env_channel": env_channel,
            "rollout_channel": rollout_channel,
            "actor_channel": actor_channel,
            "env_channel_name": env_channel_name,
            "rollout_channel_name": rollout_channel_name,
            "actor_channel_name": actor_channel_name,
            "placement": plan,
        }

    def _run_global_step(self, groups: dict[str, Any], global_step: int) -> dict[str, float]:
        actor = groups["ActorGroup"]
        rollout = groups["RolloutGroup"]
        learner = groups["LearnerGroup"]
        real_env = groups["RealEnvGroup"]
        wm_env = groups.get("WMEnvGroup")
        env_channel_name = str(groups["env_channel_name"])
        rollout_channel_name = str(groups["rollout_channel_name"])
        actor_channel_name = str(groups["actor_channel_name"])

        stage_times: dict[str, float] = {}
        sync_metrics: dict[str, float] = {}

        def mark_stage(name: str, started_at: float) -> float:
            stage_times[f"time/manual_cotrain/{name}_s"] = float(
                time.perf_counter() - started_at
            )
            return time.perf_counter()

        stage_start = time.perf_counter()
        actor.set_global_step(global_step).wait()
        rollout.set_global_step(global_step).wait()
        real_env.set_global_step(global_step).wait()
        if wm_env is not None:
            wm_env.set_global_step(global_step).wait()
        stage_start = mark_stage("set_global_step", stage_start)
        if global_step % self._sync_every() == 0:
            actor_sync = actor.sync_model_to_rollout("policy", global_step).wait()
            sync_metrics.update(_aggregate_sync_metric_lists([actor_sync]))
            rollout_sync = rollout.sync_model_from_actor("policy").wait()
            sync_metrics.update(_aggregate_sync_metric_lists([rollout_sync]))
            replay_group = groups.get("ReplayGroup")
            if replay_group is not None:
                replay_group.set_policy_version(global_step).wait()
        stage_start = mark_stage("actor_to_rollout_sync", stage_start)

        _hs_trace(f"[global_step={global_step}] EnvGroup.interact start")
        env_results = [real_env.interact(env_channel_name, rollout_channel_name, actor_channel_name)]
        if wm_env is not None:
            env_results.append(wm_env.interact(env_channel_name, rollout_channel_name, actor_channel_name))
        _hs_trace(f"[global_step={global_step}] RolloutGroup.generate start")
        rollout_result = rollout.generate(
            env_channel_name,
            rollout_channel_name,
            self._envs_per_worker(),
        )
        actor_recv_started = self._start_actor_trajectory_receivers(
            groups,
            expected_shards=self._configured_expected_trajectory_shards(groups),
            actor_channel_name=actor_channel_name,
        )

        env_metrics = _wait_env_metrics_with_rollout_guard(
            env_results,
            rollout_result,
            timeout_s=self._env_rollout_timeout_s(),
        )
        _hs_trace(f"[global_step={global_step}] EnvGroup.interact done")
        stage_start = mark_stage("env_interact_and_rollout_generate", stage_start)
        for rollout_rank, _ in enumerate(groups["RolloutGroup"].workers):
            for slot_id in range(self._envs_per_worker()):
                key = f"{int(rollout_rank)}:{int(slot_id)}"
                _hs_trace(
                    f"[env rank={int(rollout_rank)}] send StopMsg key={key}"
                )
                groups["env_channel"].put(
                    StopMsg(reason="global_step_complete"),
                    key=key,
                )
        rollout_metrics = _sum_metric_lists([rollout_result.wait()])
        _hs_trace(f"[global_step={global_step}] RolloutGroup.generate done")
        expected_shards = int(env_metrics.get("env/trajectory_shards", 0.0))
        actor_recv_metrics = self._receive_actor_trajectories(
            groups,
            expected_shards=expected_shards,
            actor_channel_name=actor_channel_name,
            stage_times=stage_times,
            started_receivers=actor_recv_started,
        )
        stage_start = mark_stage("actor_recv_trajectories", stage_start)
        advantage_metrics = _aggregate_actor_metric_lists(
            [actor.compute_advantages_and_returns().wait()]
        )
        stage_start = mark_stage("actor_compute_advantages_and_returns", stage_start)
        train_metrics = _aggregate_actor_metric_lists([actor.run_training().wait()])
        stage_start = mark_stage("actor_run_training", stage_start)

        learner_metrics: dict[str, float] = {}
        if global_step % self._learner_update_step() == 0:
            learner_metrics = _with_train_learner_aliases(
                _merge_metric_lists([learner.update("cotrain", 1).wait()])
            )
            if self._publish_learner_weights():
                learner.sync_weights("world_model", global_step).wait()
                learner.sync_weights("classifier", global_step).wait()
            stage_start = mark_stage("learner_update_wm_classifier", stage_start)
            if wm_env is not None:
                state_start = time.perf_counter()
                state_dict_results = learner.state_dicts().wait()
                sync_metrics["sync/learner_state_dicts_s"] = float(
                    time.perf_counter() - state_start
                )
                state_dicts = _first_result(state_dict_results)
                if not isinstance(state_dicts, dict):
                    raise TypeError("LearnerGroup.state_dicts() must return a mapping")
                load_start = time.perf_counter()
                load_metrics = wm_env.load_component_states(
                    {
                        "world_model": dict(state_dicts.get("world_model", {})),
                        "classifier": dict(state_dicts.get("classifier", {})),
                    },
                    global_step,
                ).wait()
                sync_metrics["sync/wm_env_load_component_states_s"] = float(
                    time.perf_counter() - load_start
                )
                sync_metrics.update(_aggregate_sync_metric_lists([load_metrics]))
            stage_start = mark_stage("learner_to_wm_env_sync", stage_start)
        else:
            stage_start = mark_stage("learner_update_wm_classifier", stage_start)
            stage_start = mark_stage("learner_to_wm_env_sync", stage_start)

        replay_group = groups.get("ReplayGroup")
        replay_metrics: dict[str, float] = {}
        if replay_group is not None:
            replay_metrics["replay_buffer/size"] = float(
                replay_group.size().wait()[0]
            )
            replay_metrics["replay_buffer/transitions"] = float(
                replay_group.num_transitions().wait()[0]
            )

        metrics = {
            "global_step": float(global_step),
            **env_metrics,
            **rollout_metrics,
            **actor_recv_metrics,
            **replay_metrics,
            **advantage_metrics,
            **train_metrics,
            **learner_metrics,
            **sync_metrics,
            **stage_times,
        }
        checkpoint_start = time.perf_counter()
        self._maybe_save_manual_checkpoint(groups, global_step, metrics)
        metrics["time/manual_cotrain/checkpoint_and_metrics_s"] = float(
            time.perf_counter() - checkpoint_start
        )
        return metrics

    def _ngpu(self) -> int:
        return int(OmegaConf.select(self.cfg, "manual_cotrain.ngpu", default=1))

    def _global_steps(self) -> int:
        return self._positive_manual_int("global_steps", default=1)

    def _sync_every(self) -> int:
        return self._positive_manual_int("sync_every", default=1)

    def _learner_update_step(self) -> int:
        return self._positive_manual_int("learner_update_step", default=1)

    def _rollout_epoch(self) -> int:
        return self._positive_manual_int("rollout_epoch", default=1)

    def _max_steps_per_rollout_epoch(self) -> int:
        return self._positive_manual_int("max_steps_per_rollout_epoch", default=1)

    def _wm_rollout_multiplier(self) -> int:
        return self._positive_manual_int("wm_rollout_multiplier", default=1)

    def _wm_max_steps_per_rollout_epoch(self) -> int:
        return self._max_steps_per_rollout_epoch() * self._wm_rollout_multiplier()

    def _num_action_chunks(self) -> int:
        return self._positive_manual_int("num_action_chunks", default=1)

    def _envs_per_worker(self) -> int:
        return self._positive_manual_int("envs_per_worker", default=1)

    def _task_id(self) -> int:
        return int(OmegaConf.select(self.cfg, "manual_cotrain.task_id", default=0))

    def _positive_manual_int(self, field: str, *, default: int) -> int:
        path = f"manual_cotrain.{field}"
        value = int(OmegaConf.select(self.cfg, path, default=default))
        if value <= 0:
            raise ValueError(f"{path} must be positive, got {value}")
        return value

    def _env_rollout_timeout_s(self) -> float:
        return float(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.env_rollout_timeout_s",
                default=600.0,
            )
            or 0.0
        )

    def _publish_learner_weights(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.publish_learner_weights",
                default=False,
            )
        )

    def _requires_bootstrap_value(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.requires_bootstrap_value",
                default=False,
            )
        )

    def _wm_env_write_replay(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.wm_env_write_replay",
                default=False,
            )
        )

    def _actor_group_size(self) -> int:
        return max(
            1,
            int(
                OmegaConf.select(
                    self.cfg,
                    "actor.train_cfg.algorithm_cfg.group_size",
                    default=OmegaConf.select(
                        self.cfg,
                        "algorithm.group_size",
                        default=1,
                    ),
                )
            ),
        )

    def _receive_actor_trajectories(
        self,
        groups: dict[str, Any],
        *,
        expected_shards: int,
        actor_channel_name: str,
        stage_times: dict[str, float],
        started_receivers: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        if started_receivers is not None:
            wait_start = time.perf_counter()
            metrics = _sum_metric_lists(
                [result.wait() for result in started_receivers["results"]]
            )
            finished_at = time.perf_counter()
            stage_times["time/manual_cotrain/actor_recv_rollout_trajectories_s"] = float(
                finished_at - wait_start
            )
            stage_times["time/manual_cotrain/actor_recv_overlap_total_s"] = float(
                finished_at - float(started_receivers["started_at"])
            )
            return metrics

        actor = groups["ActorGroup"]
        expected = max(0, int(expected_shards))
        counts = self._actor_direct_receive_counts(groups, expected)
        if counts is not None:
            recv_start = time.perf_counter()
            recv_results = [
                actor.execute_on(rank).recv_rollout_trajectories(
                    actor_channel_name,
                    count,
                )
                for rank, count in enumerate(counts)
            ]
            metrics = _sum_metric_lists([result.wait() for result in recv_results])
            stage_times["time/manual_cotrain/actor_recv_rollout_trajectories_s"] = float(
                time.perf_counter() - recv_start
            )
            return metrics

        channel_get_start = time.perf_counter()
        shards = groups["actor_channel"].get_batch(expected)
        stage_times["time/manual_cotrain/actor_channel_get_batch_s"] = float(
            time.perf_counter() - channel_get_start
        )
        load_start = time.perf_counter()
        actor.load_trajectory_shards(shards).wait()
        stage_times["time/manual_cotrain/actor_load_trajectory_shards_s"] = float(
            time.perf_counter() - load_start
        )
        return {"actor/received_shards": float(len(shards))}

    def _start_actor_trajectory_receivers(
        self,
        groups: dict[str, Any],
        *,
        expected_shards: int,
        actor_channel_name: str,
    ) -> dict[str, Any] | None:
        counts = self._actor_direct_receive_counts(groups, int(expected_shards))
        if counts is None:
            return None
        actor = groups["ActorGroup"]
        started_at = time.perf_counter()
        recv_results = [
            actor.execute_on(rank).recv_rollout_trajectories(
                actor_channel_name,
                count,
            )
            for rank, count in enumerate(counts)
        ]
        return {
            "started_at": float(started_at),
            "expected_shards": int(expected_shards),
            "results": recv_results,
        }

    def _actor_direct_receive_counts(
        self,
        groups: dict[str, Any],
        expected_shards: int,
    ) -> list[int] | None:
        actor = groups["ActorGroup"]
        expected = max(0, int(expected_shards))
        actor_ranks = len(getattr(actor, "workers", []) or [])
        group_size = self._actor_group_size()
        if (
            expected <= 0
            or actor_ranks <= 0
            or expected % group_size != 0
            or expected // group_size < actor_ranks
        ):
            return None
        return _split_actor_shard_counts(
            expected,
            actor_ranks=actor_ranks,
            group_size=group_size,
        )

    def _configured_expected_trajectory_shards(self, groups: dict[str, Any]) -> int:
        real_workers = _worker_count(groups.get("RealEnvGroup"))
        wm_workers = _worker_count(groups.get("WMEnvGroup"))
        return int(
            self._envs_per_worker()
            * self._rollout_epoch()
            * (real_workers + wm_workers)
        )

    def _render_backend(self) -> str:
        return str(
            OmegaConf.select(
                self.cfg,
                "render_backend",
                default=OmegaConf.select(self.cfg, "env.render_backend", default="osmesa"),
            )
        ).strip().lower()

    def _real_env_cfg(self) -> dict[str, Any]:
        cfg = self._cfg_dict("env.real.cfg")
        cfg.setdefault("render_backend", self._render_backend())
        cfg.setdefault("num_envs_per_worker", self._envs_per_worker())
        return cfg

    def _cfg_dict(self, path: str, fallback: str | None = None) -> dict[str, Any]:
        value = OmegaConf.select(self.cfg, path, default=None)
        if value is None and fallback is not None:
            value = OmegaConf.select(self.cfg, fallback, default=None)
        if value is None:
            raise ValueError(f"missing required config: {path}")
        plain = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
        return dict(plain or {})

    def _optional_cfg_dict(self, path: str) -> dict[str, Any]:
        value = OmegaConf.select(self.cfg, path, default=None)
        if value is None:
            return {}
        plain = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
        return dict(plain or {})

    def _load_init_ckpt(self, path: str) -> dict[str, Any]:
        cfg = OmegaConf.select(self.cfg, path, default=None)
        if cfg is None:
            return {}
        plain = _plain(cfg)
        if plain in ({}, "", None):
            return {}
        if isinstance(plain, str):
            return _load_runner_state_dicts(plain, components=None)
        if not isinstance(plain, dict):
            raise TypeError(f"{path} must be a path string or mapping")

        ckpt_path = plain.get("path")
        if ckpt_path in (None, ""):
            component_paths = {
                str(name): value
                for name, value in plain.items()
                if name not in {"path", "components"} and value not in (None, "")
            }
            return {
                name: _load_component_state_dict(str(value), name)
                for name, value in component_paths.items()
            }

        components = plain.get("components")
        if components is not None:
            components = [str(item) for item in components]
        return _load_runner_state_dicts(str(ckpt_path), components=components)

    def _manual_resume_payload(self) -> dict[str, Any] | None:
        explicit_path = OmegaConf.select(
            self.cfg,
            "manual_cotrain.resume_ckpt",
            default=None,
        )
        if explicit_path not in (None, ""):
            return _load_manual_resume_payload(str(_plain(explicit_path)), required=True)

        actor_path = self._init_ckpt_path("actor.init_ckpt")
        learner_path = self._init_ckpt_path("learner.init_ckpt")
        if actor_path is None or learner_path is None:
            return None
        if Path(actor_path).expanduser() != Path(learner_path).expanduser():
            return None
        return _load_manual_resume_payload(actor_path, required=False)

    def _init_ckpt_path(self, path: str) -> str | None:
        cfg = OmegaConf.select(self.cfg, path, default=None)
        if cfg is None:
            return None
        plain = _plain(cfg)
        if plain in ({}, "", None):
            return None
        if isinstance(plain, str):
            return str(plain)
        if not isinstance(plain, dict):
            return None
        ckpt_path = plain.get("path")
        if ckpt_path in (None, ""):
            return None
        return str(ckpt_path)

    def _restore_manual_resume_state(
        self,
        groups: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        replay_state = payload.get("replay")
        if replay_state is None:
            return
        replay_group = groups.get("ReplayGroup")
        if replay_group is None:
            raise ValueError(
                "manual checkpoint contains replay state but active config has no ReplayGroup"
            )
        replay_group.load_state_dict(dict(replay_state)).wait()

    def _maybe_save_manual_checkpoint(
        self,
        groups: dict[str, Any],
        global_step: int,
        metrics: dict[str, float],
    ) -> None:
        interval = int(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.checkpoint_every",
                default=0,
            )
            or 0
        )
        if interval <= 0 or int(global_step) % interval != 0:
            return
        ckpt_dir = (
            self.get_checkpoint_dir() / f"manual_cotrain_step_{int(global_step)}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        actor_state = _first_nonempty_mapping(groups["ActorGroup"].state_dict().wait())
        learner_states = groups["LearnerGroup"].state_dicts().wait()[0]
        if not isinstance(learner_states, dict):
            raise TypeError("LearnerGroup.state_dicts() must return a mapping")
        replay_group = groups.get("ReplayGroup")
        replay_state = None
        if replay_group is not None:
            replay_state = replay_group.state_dict().wait()[0]
        ckpt_path = ckpt_dir / "manual_cotrain.ckpt"
        state_dicts = {
            "policy": dict(actor_state),
            "world_model": dict(learner_states.get("world_model", {})),
            "classifier": dict(learner_states.get("classifier", {})),
        }
        torch.save(
            {
                "global_step": int(global_step),
                "metrics": dict(metrics),
                "state_dicts": state_dicts,
                "replay": replay_state,
            },
            ckpt_path,
        )
        run_metadata = self._manual_checkpoint_run_metadata(ckpt_dir)
        manifest = _manual_checkpoint_manifest(
            global_step=int(global_step),
            metrics=metrics,
            ckpt_name=ckpt_path.name,
            state_dicts=state_dicts,
            has_replay=replay_state is not None,
            run=run_metadata,
        )
        _atomic_write_json(ckpt_dir / "manual_cotrain_manifest.json", manifest)
        canonical_dir = self.get_global_step_checkpoint_dir(int(global_step))
        if canonical_dir != ckpt_dir:
            canonical_dir.mkdir(parents=True, exist_ok=True)
            alias_payload = (Path("..") / ckpt_dir.name / ckpt_path.name).as_posix()
            alias_manifest = _manual_checkpoint_manifest(
                global_step=int(global_step),
                metrics=metrics,
                ckpt_name=alias_payload,
                state_dicts=state_dicts,
                has_replay=replay_state is not None,
                run=self._manual_checkpoint_run_metadata(canonical_dir),
            )
            _atomic_write_json(
                canonical_dir / "manual_cotrain_manifest.json",
                alias_manifest,
            )

    def _manual_checkpoint_run_metadata(self, manifest_dir: Path) -> dict[str, str]:
        return {
            "root": _relative_path(self.get_run_dir(), manifest_dir),
            "resolved_config": _relative_path(
                self.get_resolved_config_path(),
                manifest_dir,
            ),
            "run_manifest": _relative_path(
                self.get_run_manifest_path(),
                manifest_dir,
            ),
        }

    @staticmethod
    def _placement_for(gpu_ids: list[int]) -> NodePlacementStrategy | ResourceMapPlacementStrategy:
        if not gpu_ids:
            return NodePlacementStrategy(1)
        return ResourceMapPlacementStrategy(",".join(str(gpu) for gpu in gpu_ids))

    @staticmethod
    def _resource_map_for_specs(specs: list[Any]) -> NodePlacementStrategy | ResourceMapPlacementStrategy:
        if not specs or not specs[0].gpu_ids:
            return NodePlacementStrategy(max(1, len(specs)))
        return ResourceMapPlacementStrategy(",".join(str(spec.gpu_ids[0]) for spec in specs))


def _merge_metric_lists(items: list[Any]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for item in items:
        values = item
        if isinstance(values, list):
            for nested in values:
                merged.update(_float_metrics(nested))
        else:
            merged.update(_float_metrics(values))
    return merged


def _with_train_learner_aliases(metrics: dict[str, float]) -> dict[str, float]:
    out = dict(metrics)
    if "learner/updates" in out and "train/learner_updates" not in out:
        out["train/learner_updates"] = float(out["learner/updates"])
    return out


def _sum_metric_lists(items: list[Any]) -> dict[str, float]:
    values_by_key: dict[str, list[float]] = {}
    for item in items:
        values = item
        if not isinstance(values, list):
            values = [values]
        for nested in values:
            for key, value in _float_metrics(nested).items():
                values_by_key.setdefault(key, []).append(float(value))

    summed: dict[str, float] = {}
    for key, values in values_by_key.items():
        if key.endswith("/batch_size_avg"):
            continue
        if key.endswith("/batch_size_min"):
            summed[key] = float(min(values))
        elif key.endswith("/batch_size_max"):
            summed[key] = float(max(values))
        else:
            summed[key] = float(sum(values))

    for key, value in list(summed.items()):
        if not key.endswith("/batch_size_sum"):
            continue
        prefix = key[: -len("batch_size_sum")]
        denominator = summed.get(
            f"{prefix}model_forwards",
            summed.get(f"{prefix}wm_forward_calls", 0.0),
        )
        if denominator > 0:
            summed[f"{prefix}batch_size_avg"] = float(value / denominator)
    return summed


def _aggregate_sync_metric_lists(items: list[Any]) -> dict[str, float]:
    """Aggregate synchronization metrics as barrier-oriented values.

    Sync times are wall-clock waits across workers, so max is more informative
    than sum. Counters such as number of updated rollout replicas still sum.
    """

    values_by_key: dict[str, list[float]] = {}
    for item in items:
        values = item if isinstance(item, list) else [item]
        for nested in values:
            for key, value in _float_metrics(nested).items():
                values_by_key.setdefault(key, []).append(float(value))

    aggregated: dict[str, float] = {}
    for key, values in values_by_key.items():
        if key.endswith("_s"):
            aggregated[key] = float(max(values))
        elif key.endswith("_updated"):
            aggregated[key] = float(sum(values))
        elif key.endswith("_version"):
            aggregated[key] = float(max(values))
        elif key.endswith("_bytes") or key.endswith("_tensors"):
            aggregated[key] = float(max(values))
        else:
            aggregated[key] = float(sum(values))
    return aggregated


def _aggregate_actor_metric_lists(items: list[Any]) -> dict[str, float]:
    values_by_key: dict[str, list[float]] = {}
    for item in items:
        values = item if isinstance(item, list) else [item]
        for nested in values:
            for key, value in _float_metrics(nested).items():
                values_by_key.setdefault(key, []).append(float(value))

    aggregated: dict[str, float] = {}
    for key, values in values_by_key.items():
        if key in {"actor/trajectory_count", "actor/received_shards"}:
            aggregated[key] = float(sum(values))
        elif key in {"actor/ppo_updates"}:
            aggregated[key] = float(max(values))
        else:
            aggregated[key] = float(sum(values) / len(values))
    return aggregated


def _split_actor_shard_counts(
    expected_shards: int,
    *,
    actor_ranks: int,
    group_size: int,
) -> list[int]:
    if actor_ranks <= 0:
        raise ValueError("actor_ranks must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    total = max(0, int(expected_shards))
    if total % int(group_size) != 0:
        raise ValueError(
            "actor trajectory shard count must be divisible by group_size; "
            f"got {total} and {group_size}"
        )
    groups = total // int(group_size)
    base_groups, extra_groups = divmod(groups, int(actor_ranks))
    return [
        int(group_size) * (base_groups + (1 if rank < extra_groups else 0))
        for rank in range(int(actor_ranks))
    ]


def _worker_count(group: Any | None) -> int:
    if group is None:
        return 0
    workers = getattr(group, "workers", None)
    if workers is None:
        return 1
    return len(workers)


def _chunk_steps(max_steps_per_rollout_epoch: int, num_action_chunks: int) -> int:
    max_steps = int(max_steps_per_rollout_epoch)
    chunks = int(num_action_chunks)
    if chunks <= 0:
        raise ValueError("num_action_chunks must be positive")
    if max_steps % chunks != 0:
        raise ValueError(
            "max_steps_per_rollout_epoch must be divisible by num_action_chunks "
            f"({max_steps} % {chunks})"
        )
    return max_steps // chunks


def _wait_env_metrics_with_rollout_guard(
    env_results: list[Any],
    rollout_result: Any,
    *,
    timeout_s: float,
    poll_s: float = 1.0,
) -> dict[str, float]:
    """Wait for EnvGroup while surfacing RolloutGroup failures immediately."""

    start = time.monotonic()
    while not all(result.done() for result in env_results):
        ready = rollout_result.ready()
        if ready:
            values = rollout_result.wait_refs(ready)
            raise RuntimeError(
                "RolloutGroup.generate completed before EnvGroup.interact; "
                f"ready_result={values!r}"
            )
        if timeout_s > 0 and (time.monotonic() - start) > float(timeout_s):
            raise TimeoutError(
                "EnvGroup.interact did not finish before "
                f"manual_cotrain.env_rollout_timeout_s={float(timeout_s):.1f}s; "
                "RolloutGroup.generate is still running or waiting for StopMsg. "
                "Set DVLA_COTRAIN_HANDSHAKE_TRACE=1 before launching to log "
                "EnvGroup/RolloutGroup action handshakes."
            )
        time.sleep(max(0.0, float(poll_s)))
    return _sum_metric_lists([result.wait() for result in env_results])


def _float_metrics(values: Any) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}
    return {str(key): float(value) for key, value in values.items()}


def _first_result(values: Any) -> Any:
    if isinstance(values, list):
        return values[0] if values else None
    return values


def _first_nonempty_mapping(values: Any) -> dict[str, Any]:
    if isinstance(values, dict):
        return dict(values)
    if isinstance(values, list):
        for value in values:
            if isinstance(value, dict) and value:
                return dict(value)
        return {}
    return {}


def _plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _manual_checkpoint_manifest(
    *,
    global_step: int,
    metrics: dict[str, float],
    ckpt_name: str,
    state_dicts: dict[str, dict[str, Any]],
    has_replay: bool,
    run: dict[str, str] | None = None,
) -> dict[str, Any]:
    policy_version = int(metrics.get("sync/policy_version", global_step))
    rollout_policy_version = int(
        metrics.get("sync/rollout_policy_version", policy_version)
    )
    world_model_version = int(
        metrics.get("sync/world_model_version", global_step)
    )
    classifier_version = int(
        metrics.get("sync/classifier_version", global_step)
    )
    versions = {
        "global_step": int(global_step),
        "policy_version": policy_version,
        "world_model_version": world_model_version,
        "classifier_version": classifier_version,
        "actor_policy_version": policy_version,
        "rollout_policy_version": rollout_policy_version,
        "wm_version": world_model_version,
    }
    return {
        "schema_version": 1,
        "global_step": int(global_step),
        "versions": versions,
        "components": {
            name: {
                "path": str(ckpt_name),
                "state_dict_key": name,
                "tensors": len(state),
            }
            for name, state in sorted(state_dicts.items())
        },
        "replay": {
            "path": str(ckpt_name),
            "state_dict_key": "replay",
            "present": bool(has_replay),
            "size": float(metrics.get("replay_buffer/size", 0.0)),
            "transitions": float(metrics.get("replay_buffer/transitions", 0.0)),
        },
        "run": dict(run or {}),
        "metrics_keys": sorted(str(key) for key in metrics),
    }


def _relative_path(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start)).as_posix()


def _checkpoint_payload_path(path: Path) -> Path:
    if path.is_dir():
        manifest_path = path / "manual_cotrain_manifest.json"
        if manifest_path.is_file():
            return _checkpoint_payload_path(manifest_path)
        return path / "manual_cotrain.ckpt"
    if path.name == "manual_cotrain_manifest.json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        components = manifest.get("components", {})
        policy = components.get("policy", {}) if isinstance(components, dict) else {}
        payload_name = policy.get("path", "manual_cotrain.ckpt")
        return path.parent / str(payload_name)
    return path


def _load_manual_resume_payload(
    ckpt_path: str,
    *,
    required: bool,
) -> dict[str, Any] | None:
    path = _checkpoint_payload_path(Path(ckpt_path).expanduser())
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"manual cotrain resume checkpoint not found: {path}")
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        if required:
            raise TypeError("manual cotrain resume checkpoint must contain a mapping")
        return None
    if "global_step" not in payload:
        if required:
            raise ValueError(
                "manual cotrain resume checkpoint must include global_step"
            )
        return None
    if "state_dicts" not in payload:
        if required:
            raise ValueError(
                "manual cotrain resume checkpoint must include state_dicts"
            )
        return None
    return payload


def _load_runner_state_dicts(
    ckpt_path: str,
    *,
    components: list[str] | None,
) -> dict[str, Any]:
    path = _checkpoint_payload_path(Path(ckpt_path).expanduser())
    if not path.is_file():
        raise FileNotFoundError(f"Ray init checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dicts = payload.get("state_dicts") if isinstance(payload, dict) else None
    if not isinstance(state_dicts, dict):
        raise RuntimeError(f"{path} has no runner-format state_dicts mapping")
    names = list(state_dicts) if components is None else list(components)
    missing = [name for name in names if name not in state_dicts]
    if missing:
        raise RuntimeError(
            f"{path} missing state_dicts for requested component(s): {missing}"
        )
    return {name: state_dicts[name] for name in names}


def _load_component_state_dict(ckpt_path: str, component: str) -> dict[str, Any]:
    path = Path(ckpt_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Ray init checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        state_dicts = payload.get("state_dicts")
        if isinstance(state_dicts, dict) and component in state_dicts:
            return state_dicts[component]
        for key in ("model", "state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        component_sd = payload.get(component)
        if isinstance(component_sd, dict):
            return component_sd
        if all(isinstance(key, str) for key in payload):
            return payload
    raise RuntimeError(f"{path} does not contain a usable state dict")


__all__ = ["ManualCotrainRayRunner"]
