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
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ray
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.render_device_config import (
    cuda_visible_devices_from_env,
    parse_device_ids,
)
from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import (
    ComponentPlacement,
    NodePlacementStrategy,
    ResourceMapPlacementStrategy,
)
from dreamervla.scheduler.worker_group import WorkerGroup
from dreamervla.utils.egl_device import _ZERO_GPU_EGL_ERROR
from dreamervla.utils.frozen_components import (
    load_frozen_component,
    require_component_config_match,
    resolve_classifier_threshold,
    state_dict_sha256,
)
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
        self._real_rollout_episodes = 0
        self._real_rollout_successes = 0
        self._overlap_bootstrap_offload_warned = False
        self._frozen_state_hashes: dict[str, str] = {}
        self._frozen_source_checkpoints: dict[str, str] = {}
        self._frozen_classifier_threshold: float | None = None
        self._pending_manual_resume_payload: dict[str, Any] | None = None
        self._policy_initial_hash = ""
        self._policy_final_hash = ""
        self._applied_policy_steps = 0

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
            _hs_trace("[manual-cotrain] resume payload load start")
            resume_payload = self._manual_resume_payload()
            self._pending_manual_resume_payload = resume_payload
            groups = self._build_groups(cluster)
            resume_step = 0
            if resume_payload is not None:
                resume_step = int(resume_payload.get("global_step", 0))
                _hs_trace(
                    f"[manual-cotrain] restore resume state start step={resume_step}"
                )
                self._restore_manual_resume_state(groups, resume_payload)
                _hs_trace("[manual-cotrain] restore resume state done")
            else:
                _hs_trace("[manual-cotrain] no resume payload")
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
                self._report_global_step_progress(
                    global_step=global_step,
                    total_steps=target_step,
                    metrics=step_metrics,
                )
                last_step = int(global_step)
            metrics["global_step"] = int(last_step)
            if not self._learner_updates_enabled():
                self._finalize_frozen_policy_run(
                    groups,
                    global_step=int(last_step),
                    metrics=metrics,
                )
            return metrics
        finally:
            self._pending_manual_resume_payload = None
            cluster.shutdown()

    def _placement_plan(self) -> ManualCotrainPlacementPlan:
        return build_manual_cotrain_placement(
            self._ngpu(),
            real_env_workers=(
                self._real_env_workers() if self._real_env_enabled() else 0
            ),
            include_learner=self._learner_updates_enabled(),
            component_gpu_groups=self._component_gpu_groups(),
        )

    def _component_gpu_groups(self) -> dict[str, list[list[int]]] | None:
        component_cfg = OmegaConf.select(
            self.cfg,
            "cluster.component_placement",
            default=None,
        )
        if component_cfg is None:
            return None
        placement = ComponentPlacement(self.cfg)
        cluster = SimpleNamespace(num_gpus=self._ngpu())
        groups: dict[str, list[list[int]]] = {}
        for component in ("env", "real_env", "wm_env", "rollout", "actor", "learner"):
            if not placement.has_component(component):
                continue
            resolved = placement.get_strategy(component).get_placement(cluster)
            groups[component] = [
                [int(gpu) for gpu in item.visible_accelerators]
                for item in resolved
            ]
        return groups or None

    def _target_group_names(self) -> list[str]:
        if self._learner_updates_enabled() and self._real_env_enabled():
            return ["LearnerGroup", "ActorGroup", "RolloutGroup", "EnvGroup"]
        names: list[str] = []
        if self._learner_updates_enabled():
            names.append("LearnerGroup")
        names.extend(["ActorGroup", "RolloutGroup"])
        if self._real_env_enabled():
            names.append("RealEnvGroup")
        names.append("WMEnvGroup")
        return names

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

    def _manual_cotrain_progress_dir(self, global_step: int) -> Path:
        return (
            self.get_diagnostics_dir()
            / "manual_cotrain_progress"
            / f"global_step_{int(global_step):08d}"
        )

    def _prepare_manual_cotrain_progress_dir(self, global_step: int) -> Path:
        progress_dir = self._manual_cotrain_progress_dir(global_step)
        progress_dir.mkdir(parents=True, exist_ok=True)
        for file in progress_dir.glob("*.json"):
            try:
                file.unlink()
            except FileNotFoundError:
                pass
        return progress_dir

    def _configure_env_progress(
        self,
        group: Any | None,
        progress_dir: Path,
    ) -> None:
        if group is None:
            return
        progress_dir.mkdir(parents=True, exist_ok=True)
        interval = float(
            OmegaConf.select(self.cfg, "console.progress_every_s", default=5.0)
        )
        group.configure_progress(str(progress_dir), min_interval_s=interval).wait()

    def _build_groups(self, cluster: Cluster) -> dict[str, Any]:
        plan = self._placement_plan()
        _hs_trace(f"[build_groups] start ngpu={int(plan.ngpu)}")
        env_channel_name = f"manual-cotrain-env-{uuid.uuid4().hex}"
        rollout_channel_name = f"manual-cotrain-rollout-{uuid.uuid4().hex}"
        actor_channel_name = f"manual-cotrain-actor-{uuid.uuid4().hex}"
        env_channel = Channel.create(env_channel_name)
        rollout_channel = Channel.create(rollout_channel_name)
        actor_channel = Channel.create(actor_channel_name)
        replay_group = None
        replay = None
        replay_seed_metrics: dict[str, float] = {}
        replay_resume_restored = False
        replay_cfg = OmegaConf.select(self.cfg, "replay.cfg", default=None)
        if replay_cfg is not None:
            _hs_trace("[build_groups] launch ReplayWorker start")
            replay_group = WorkerGroup(
                ReplayWorker,
                self._cfg_dict("replay.cfg"),
            ).launch(cluster, NodePlacementStrategy(1), name="ReplayWorker")
            replay = replay_group.workers[0]
            _hs_trace("[build_groups] launch ReplayWorker done")
            replay_seed_cfg = OmegaConf.select(
                self.cfg,
                "replay.seed",
                default=None,
            )
            resume_replay_state = (
                self._pending_manual_resume_payload.get("replay")
                if self._pending_manual_resume_payload is not None
                else None
            )
            if resume_replay_state is not None:
                replay_group.load_state_dict(dict(resume_replay_state)).wait()
                replay_resume_restored = True
            elif replay_seed_cfg is not None:
                _hs_trace("[build_groups] seed ReplayWorker start")
                replay_seed_metrics = _merge_metric_lists(
                    [
                        replay_group.seed_from_offline(
                            self._cfg_dict("replay.seed")
                        ).wait()
                    ]
                )
                _hs_trace("[build_groups] seed ReplayWorker done")
            resume_sampling_state = (
                self._pending_manual_resume_payload.get("replay_sampling_state")
                if self._pending_manual_resume_payload is not None
                else None
            )
            if resume_sampling_state is not None:
                replay_group.load_sampling_state_dict(
                    dict(resume_sampling_state)
                ).wait()
                replay_resume_restored = True

        real_specs = [spec for spec in plan.env_specs if spec.role == "real_env"]
        real_env_group = None
        if self._real_env_enabled():
            if not real_specs:
                raise ValueError(
                    "manual cotrain placement must include a real_env worker when "
                    "manual_cotrain.real_env_enabled=true"
                )
            real_rollout_epochs = self._real_rollout_epochs_by_worker(len(real_specs))
            initial_real_rollout_epoch = real_rollout_epochs[0]
            _hs_trace("[build_groups] launch RealEnvWorker start")
            real_env_group = WorkerGroup(
                RealEnvWorker,
                self._real_env_cfg(),
                self._envs_per_worker(),
                initial_real_rollout_epoch,
                self._max_steps_per_rollout_epoch(),
                self._num_action_chunks(),
                task_id=self._task_id(),
                replay=replay,
                dump=None,
                rank_offset=0,
                request_final_bootstrap=self._requires_bootstrap_value(),
            ).launch(
                cluster,
                self._resource_map_for_specs(real_specs),
                name="RealEnvWorker",
            )
            for rank, rollout_epoch in enumerate(real_rollout_epochs):
                real_env_group.execute_on(rank).configure_rollout_epoch(
                    rollout_epoch
                ).wait()
            _hs_trace("[build_groups] launch RealEnvWorker done")
        wm_specs = [spec for spec in plan.env_specs if spec.role == "wm_env"]
        wm_env_group = None
        frozen_component_load_metrics: dict[str, float] = {}
        frozen_hashes_verified = False
        if wm_specs:
            wm_rollout_epochs = self._wm_rollout_epochs_by_worker(len(wm_specs))
            initial_wm_rollout_epoch = wm_rollout_epochs[0]
            _hs_trace("[build_groups] launch WMEnvWorker start")
            wm_env_group = WorkerGroup(
                WMEnvWorker,
                self._cfg_dict("env.wm.cfg"),
                self._wm_envs_per_worker(),
                initial_wm_rollout_epoch,
                self._wm_max_steps_per_rollout_epoch(),
                self._num_action_chunks(),
                task_id=self._task_id(),
                replay=replay,
                dump=None,
                rank_offset=len(real_specs),
                request_final_bootstrap=self._requires_bootstrap_value(),
                replay_write_enabled=self._wm_env_write_replay(),
            ).launch(
                cluster,
                self._resource_map_for_specs(wm_specs),
                name="WMEnvWorker",
            )
            for rank, rollout_epoch in enumerate(wm_rollout_epochs):
                wm_env_group.execute_on(rank).configure_rollout_epoch(rollout_epoch).wait()
            _hs_trace("[build_groups] launch WMEnvWorker done")
            if not self._learner_updates_enabled():
                frozen = self._load_frozen_component_states()
                shared_component_states = _share_ray_value(
                    frozen["component_states"],
                    cluster=cluster,
                )
                frozen_component_load_metrics = _merge_metric_lists(
                    [
                        wm_env_group.load_component_states(
                            shared_component_states,
                            0,
                        ).wait()
                    ]
                )
                self._frozen_state_hashes = dict(frozen["frozen_state_hashes"])
                self._frozen_source_checkpoints = dict(
                    frozen["source_checkpoints"]
                )
                self._frozen_classifier_threshold = float(
                    frozen["component_states"]["classifier_threshold"]
                )
                self._assert_frozen_component_hashes(
                    {"WMEnvGroup": wm_env_group}
                )
                frozen_hashes_verified = True
        elif not self._real_env_enabled():
            raise ValueError("frozen manual cotrain requires at least one WMEnvWorker")

        _hs_trace("[build_groups] launch MultiStepRolloutWorker start")
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
        _hs_trace("[build_groups] launch MultiStepRolloutWorker done")

        _hs_trace("[build_groups] launch EmbodiedFSDPActor start")
        actor_init_checkpoint = _share_ray_value(
            self._actor_init_checkpoint(),
            cluster=cluster,
        )
        actor_group = WorkerGroup(
            EmbodiedFSDPActor,
            self._cfg_dict("actor.policy_cfg"),
            actor_init_checkpoint,
            self._cfg_dict("actor.train_cfg"),
        ).launch(
            cluster,
            self._resource_map_for_specs(plan.actor_specs),
            name="EmbodiedFSDPActor",
        )
        _hs_trace("[build_groups] launch EmbodiedFSDPActor done")
        if not self._learner_updates_enabled():
            self._initialize_policy_hashes(actor_group)

        learner_group = None
        if self._learner_updates_enabled():
            if plan.learner_spec is None:
                raise ValueError(
                    "manual cotrain learner updates require a learner placement"
                )
            _hs_trace("[build_groups] launch LearnerWorker start")
            learner_group = WorkerGroup(
                LearnerWorker,
                self._cfg_dict("learner.model_cfg"),
                self._learner_init_checkpoint(),
                self._cfg_dict("learner.train_cfg"),
                replay,
            ).launch(
                cluster,
                self._placement_for(plan.learner_spec.gpu_ids),
                name="LearnerWorker",
            )
            _hs_trace("[build_groups] launch LearnerWorker done")
            if wm_env_group is not None:
                initial_states = learner_group.state_dicts().wait()[0]
                component_states = {
                    name: dict(initial_states[name])
                    for name in ("world_model", "classifier")
                    if isinstance(initial_states.get(name), dict)
                }
                if "classifier_threshold" in initial_states:
                    component_states["classifier_threshold"] = float(
                        initial_states["classifier_threshold"]
                    )
                if component_states:
                    shared_component_states = _share_ray_value(
                        component_states,
                        cluster=cluster,
                    )
                    wm_env_group.load_component_states(
                        shared_component_states,
                        0,
                    ).wait()
        _hs_trace("[build_groups] all groups launched")

        return {
            "LearnerGroup": learner_group,
            "ActorGroup": actor_group,
            "RolloutGroup": rollout_group,
            "ReplayGroup": replay_group,
            "replay_resume_restored": bool(replay_resume_restored),
            "replay_seed_metrics": replay_seed_metrics,
            "frozen_component_load_metrics": frozen_component_load_metrics,
            "frozen_state_hashes": dict(self._frozen_state_hashes),
            "frozen_source_checkpoints": dict(self._frozen_source_checkpoints),
            "frozen_hashes_verified": bool(frozen_hashes_verified),
            "cluster": cluster,
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

    def _load_frozen_component_states(self) -> dict[str, Any]:
        """Load and validate immutable WM/CLS sources for the policy-only route."""

        wm_path = self._init_ckpt_path("init.world_model_state_ckpt")
        classifier_path = self._init_ckpt_path("init.classifier_state_ckpt")
        if wm_path is None or classifier_path is None:
            raise ValueError(
                "frozen manual cotrain requires explicit world-model and classifier "
                "checkpoints"
            )
        loaded_wm = load_frozen_component(wm_path, "world_model")
        loaded_classifier = load_frozen_component(classifier_path, "classifier")
        require_component_config_match(
            loaded_wm.metadata,
            component="world_model",
            active_cfg=OmegaConf.select(
                self.cfg,
                "world_model",
                default=OmegaConf.select(
                    self.cfg,
                    "learner.model_cfg.world_model",
                ),
            ),
        )
        require_component_config_match(
            loaded_classifier.metadata,
            component="classifier",
            active_cfg=OmegaConf.select(
                self.cfg,
                "classifier",
                default=OmegaConf.select(
                    self.cfg,
                    "learner.model_cfg.classifier",
                ),
            ),
        )
        configured_threshold = OmegaConf.select(
            self.cfg,
            "algorithm.lumos.classifier_threshold",
            default=OmegaConf.select(
                self.cfg,
                "learner.train_cfg.classifier_threshold",
                default=None,
            ),
        )
        threshold = resolve_classifier_threshold(
            loaded_classifier.metadata,
            configured=(
                None
                if configured_threshold is None
                else float(configured_threshold)
            ),
        )
        return {
            "component_states": {
                "world_model": loaded_wm.state_dict,
                "classifier": loaded_classifier.state_dict,
                "classifier_threshold": float(threshold),
            },
            "frozen_state_hashes": {
                "world_model": state_dict_sha256(loaded_wm.state_dict),
                "classifier": state_dict_sha256(loaded_classifier.state_dict),
            },
            "source_checkpoints": {
                "world_model": str(Path(wm_path).expanduser().resolve()),
                "classifier": str(Path(classifier_path).expanduser().resolve()),
            },
        }

    def _learner_init_checkpoint(self) -> dict[str, Any]:
        """Resolve either a consolidated checkpoint or independent WM/CLS sources.

        Independent component files are shared by frozen and trainable recipes;
        trainability is owned solely by ``manual_cotrain.learner_updates_enabled``.
        """

        consolidated = self._load_init_ckpt("learner.init_ckpt")
        if consolidated:
            return consolidated
        wm_path = self._init_ckpt_path("init.world_model_state_ckpt")
        classifier_path = self._init_ckpt_path("init.classifier_state_ckpt")
        if wm_path is None and classifier_path is None:
            return {}
        if wm_path is None or classifier_path is None:
            raise ValueError(
                "trainable LearnerGroup requires both init.world_model_state_ckpt "
                "and init.classifier_state_ckpt when either is set"
            )
        loaded_wm = load_frozen_component(wm_path, "world_model")
        loaded_classifier = load_frozen_component(classifier_path, "classifier")
        require_component_config_match(
            loaded_wm.metadata,
            component="world_model",
            active_cfg=OmegaConf.select(
                self.cfg,
                "world_model",
                default=OmegaConf.select(
                    self.cfg,
                    "learner.model_cfg.world_model",
                ),
            ),
        )
        require_component_config_match(
            loaded_classifier.metadata,
            component="classifier",
            active_cfg=OmegaConf.select(
                self.cfg,
                "classifier",
                default=OmegaConf.select(
                    self.cfg,
                    "learner.model_cfg.classifier",
                ),
            ),
        )
        configured_threshold = OmegaConf.select(
            self.cfg,
            "algorithm.lumos.classifier_threshold",
            default=OmegaConf.select(
                self.cfg,
                "learner.train_cfg.classifier_threshold",
                default=None,
            ),
        )
        threshold = resolve_classifier_threshold(
            loaded_classifier.metadata,
            configured=(
                None
                if configured_threshold is None
                else float(configured_threshold)
            ),
        )
        return {
            "world_model": loaded_wm.state_dict,
            "classifier": loaded_classifier.state_dict,
            "classifier_threshold": float(threshold),
        }

    def _actor_init_checkpoint(self) -> dict[str, Any]:
        payload = self._pending_manual_resume_payload
        if payload is not None:
            state_dicts = payload.get("state_dicts", {})
            policy_state = (
                state_dicts.get("policy")
                if isinstance(state_dicts, dict)
                else None
            )
            if not isinstance(policy_state, dict) or not policy_state:
                raise RuntimeError(
                    "manual cotrain resume checkpoint has no non-empty policy state"
                )
            init_state = {"policy": dict(policy_state)}
            optimizer_state = state_dicts.get("policy_optimizer")
            if isinstance(optimizer_state, dict) and optimizer_state:
                init_state["policy_optimizer"] = dict(optimizer_state)
            return init_state
        return self._load_init_ckpt("actor.init_ckpt")

    def _initialize_policy_hashes(self, actor_group: Any) -> None:
        actor_state = _first_nonempty_mapping(actor_group.state_dict().wait())
        current_hash = state_dict_sha256(actor_state)
        payload = self._pending_manual_resume_payload
        if payload is None:
            self._policy_initial_hash = current_hash
            self._policy_final_hash = current_hash
            self._applied_policy_steps = 0
            return
        expected_current = str(payload.get("policy_final_hash", "") or "")
        if expected_current and expected_current != current_hash:
            raise RuntimeError(
                "resume policy state hash differs from checkpoint metadata"
            )
        self._policy_initial_hash = str(
            payload.get("policy_initial_hash", current_hash) or current_hash
        )
        self._policy_final_hash = current_hash
        self._applied_policy_steps = int(
            payload.get("applied_policy_steps", 0) or 0
        )

    def _assert_frozen_component_hashes(self, groups: dict[str, Any]) -> None:
        expected = dict(self._frozen_state_hashes)
        if not expected:
            raise RuntimeError("frozen WM/classifier source hashes are missing")
        wm_env = groups.get("WMEnvGroup")
        if wm_env is None:
            raise RuntimeError("frozen WM/classifier audit requires WMEnvGroup")
        raw_results = wm_env.component_state_hashes().wait()
        results = raw_results if isinstance(raw_results, list) else [raw_results]
        if not results:
            raise RuntimeError("WMEnvGroup returned no frozen component hashes")
        for rank, raw_hashes in enumerate(results):
            hashes = dict(raw_hashes or {})
            if hashes != expected:
                raise RuntimeError(
                    "frozen WM/classifier state drift detected on WMEnv rank "
                    f"{rank}: {hashes!r} != {expected!r}"
                )

    def _run_global_step(self, groups: dict[str, Any], global_step: int) -> dict[str, float]:
        global_step_start = time.perf_counter()
        actor = groups["ActorGroup"]
        rollout = groups["RolloutGroup"]
        learner = groups.get("LearnerGroup")
        real_env = groups.get("RealEnvGroup")
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
        if real_env is not None:
            real_env.set_global_step(global_step).wait()
        if wm_env is not None:
            wm_env.set_global_step(global_step).wait()
        progress_dir = self._manual_cotrain_progress_dir(global_step)
        self._configure_env_progress(real_env, progress_dir)
        self._configure_env_progress(wm_env, progress_dir)
        progress_monitor = _ManualCotrainEnvProgressMonitor(
            progress_dir,
            self.console_progress,
            desc=f"manual-cotrain-env/{int(global_step):08d}",
        )
        dynamic_wm_leases = (
            wm_env is not None
            and self._wm_rollout_target_trajectories() is not None
        )
        if not dynamic_wm_leases:
            progress_monitor.report(force=True)
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
        real_env_results = []
        if real_env is not None:
            real_env_results.append(
                real_env.interact(
                    env_channel_name,
                    rollout_channel_name,
                    actor_channel_name,
                )
            )
        env_results = list(real_env_results)
        if wm_env is not None and not dynamic_wm_leases:
            env_results.append(
                wm_env.interact(
                    env_channel_name,
                    rollout_channel_name,
                    actor_channel_name,
                )
            )
        _hs_trace(f"[global_step={global_step}] RolloutGroup.generate start")
        rollout_result = rollout.generate(
            env_channel_name,
            rollout_channel_name,
            self._rollout_num_slots(),
        )
        actor_recv_started = None
        if self._actor_receive_overlap_enabled():
            actor_recv_started = self._start_actor_trajectory_receivers(
                groups,
                expected_shards=self._configured_expected_trajectory_shards(groups),
                actor_channel_name=actor_channel_name,
            )

        if wm_env is not None and dynamic_wm_leases:
            env_metrics = self._wait_env_metrics_with_dynamic_wm_leases(
                real_env_results=real_env_results,
                wm_env=wm_env,
                rollout_result=rollout_result,
                env_channel_name=env_channel_name,
                rollout_channel_name=rollout_channel_name,
                actor_channel_name=actor_channel_name,
                timeout_s=self._env_rollout_timeout_s(),
                progress=progress_monitor,
            )
        else:
            env_metrics = _wait_env_metrics_with_rollout_guard(
                env_results,
                rollout_result,
                timeout_s=self._env_rollout_timeout_s(),
                progress=progress_monitor,
            )
        if not dynamic_wm_leases:
            progress_monitor.report(force=True)
        _hs_trace(f"[global_step={global_step}] EnvGroup.interact done")
        stage_start = mark_stage("env_interact_and_rollout_generate", stage_start)
        for rollout_rank, _ in enumerate(groups["RolloutGroup"].workers):
            key = str(int(rollout_rank))
            _hs_trace(f"[env rank={int(rollout_rank)}] send StopMsg key={key}")
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
            env_metrics=env_metrics,
            actor_channel_name=actor_channel_name,
            stage_times=stage_times,
            started_receivers=actor_recv_started,
        )
        stage_start = mark_stage("actor_recv_trajectories", stage_start)
        advantage_metrics = _aggregate_actor_metric_lists(
            [actor.compute_advantages_and_returns().wait()]
        )
        stage_start = mark_stage("actor_compute_advantages_and_returns", stage_start)
        train_handle = actor.run_training()
        prefetch_handle = self._maybe_prefetch_env_bootstrap(real_env)
        train_metrics = _aggregate_actor_metric_lists([train_handle.wait()])
        if not self._learner_updates_enabled():
            self._applied_policy_steps += int(
                max(0.0, float(train_metrics.get("actor/ppo_updates", 0.0)))
            )
        if prefetch_handle is not None:
            prefetch_handle.wait()
        stage_start = mark_stage("actor_run_training", stage_start)

        learner_metrics: dict[str, float] = {}
        if self._learner_updates_enabled() and learner is None:
            raise ValueError(
                "manual cotrain learner updates are enabled but LearnerGroup is absent"
            )
        if (
            self._learner_updates_enabled()
            and global_step % self._learner_update_step() == 0
        ):
            learner_metrics = _with_train_learner_aliases(
                _merge_metric_lists([learner.update(self._learner_update_phase(), 1).wait()])
            )
            sync_world_model_to_env = self._should_sync_world_model_after_learner_update(
                learner_metrics
            )
            if self._publish_learner_weights():
                if sync_world_model_to_env:
                    learner.sync_weights("world_model", global_step).wait()
                    sync_metrics[
                        "sync/world_model_publish_skipped_classifier_not_updated"
                    ] = 0.0
                else:
                    sync_metrics[
                        "sync/world_model_publish_skipped_classifier_not_updated"
                    ] = 1.0
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
                component_states = {
                    "classifier": dict(state_dicts.get("classifier", {})),
                }
                if "classifier_threshold" in state_dicts:
                    component_states["classifier_threshold"] = float(
                        state_dicts["classifier_threshold"]
                    )
                if sync_world_model_to_env:
                    component_states["world_model"] = dict(
                        state_dicts.get("world_model", {})
                    )
                    sync_metrics[
                        "sync/wm_env_world_model_skipped_classifier_not_updated"
                    ] = 0.0
                else:
                    sync_metrics[
                        "sync/wm_env_world_model_skipped_classifier_not_updated"
                    ] = 1.0
                share_start = time.perf_counter()
                shared_component_states = _share_ray_value(
                    component_states,
                    cluster=groups.get("cluster"),
                )
                sync_metrics["sync/learner_state_share_s"] = float(
                    time.perf_counter() - share_start
                )
                load_start = time.perf_counter()
                load_metrics = wm_env.load_component_states(
                    shared_component_states,
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
            **self._real_env_success_rate_metrics(env_metrics, global_step=global_step),
            **rollout_metrics,
            **actor_recv_metrics,
            **replay_metrics,
            **advantage_metrics,
            **train_metrics,
            **learner_metrics,
            **sync_metrics,
            **stage_times,
        }
        if global_step == 1:
            metrics.update(groups.get("replay_seed_metrics", {}))
            metrics.update(groups.get("frozen_component_load_metrics", {}))
        metrics = _with_train_learner_aliases(metrics)
        checkpoint_start = time.perf_counter()
        self._maybe_save_manual_checkpoint(groups, global_step, metrics)
        metrics["time/manual_cotrain/checkpoint_and_metrics_s"] = float(
            time.perf_counter() - checkpoint_start
        )
        metrics["time/manual_cotrain/global_step_s"] = float(
            time.perf_counter() - global_step_start
        )
        return metrics

    def _report_global_step_progress(
        self,
        *,
        global_step: int,
        total_steps: int,
        metrics: dict[str, float],
    ) -> None:
        """Report one monotonic policy-update progress line per global step."""

        updates = max(
            0,
            int(
                float(
                    metrics.get(
                        "actor/ppo_optimizer_steps",
                        metrics.get("actor/ppo_updates", 0.0),
                    )
                )
            ),
        )
        parts = [f"ppo_steps={updates}"]
        valid_samples = metrics.get("actor/global_loss_mask_sum")
        total_samples = metrics.get("actor/global_ppo_samples")
        if valid_samples is not None and total_samples is not None:
            parts.append(f"samples={int(valid_samples)}/{int(total_samples)}")
        if "actor/global_batch_size" in metrics:
            parts.append(f"batch={int(metrics['actor/global_batch_size'])}")
        if "actor/micro_batch_size" in metrics:
            parts.append(f"micro={int(metrics['actor/micro_batch_size'])}")
        if "actor/loss" in metrics:
            parts.append(f"loss={float(metrics['actor/loss']):.4g}")
        if "actor/approx_kl" in metrics:
            parts.append(f"approx_kl={float(metrics['actor/approx_kl']):.4g}")
        if "actor/clip_fraction" in metrics:
            parts.append(f"clip_frac={float(metrics['actor/clip_fraction']):.4g}")
        if "actor/lr" in metrics:
            parts.append(f"lr={float(metrics['actor/lr']):.4g}")
        durations = (
            ("imagine", "time/manual_cotrain/env_interact_and_rollout_generate_s"),
            ("actor", "time/manual_cotrain/actor_run_training_s"),
            ("step", "time/manual_cotrain/global_step_s"),
        )
        for label, key in durations:
            if key in metrics:
                parts.append(f"{label}={float(metrics[key]):.1f}s")
        self.console_progress(
            int(global_step),
            int(total_steps),
            "manual-cotrain",
            unit="step",
            status=" ".join(parts),
            force=True,
        )

    @staticmethod
    def _should_sync_world_model_after_learner_update(
        learner_metrics: dict[str, float],
    ) -> bool:
        if "cls/updated" not in learner_metrics:
            return True
        return float(learner_metrics.get("cls/updated", 1.0)) > 0.5

    def _real_env_success_rate_metrics(
        self,
        env_metrics: dict[str, float],
        *,
        global_step: int,
    ) -> dict[str, float]:
        """Return real-LIBERO episode success metrics for the current actor."""

        step_episodes = int(
            max(0.0, float(env_metrics.get("env/real_env/episodes_completed", 0.0)))
        )
        step_successes = int(
            max(0.0, float(env_metrics.get("env/real_env/episodes_successful", 0.0)))
        )
        if step_successes > step_episodes:
            step_successes = step_episodes
        self._real_rollout_episodes += int(step_episodes)
        self._real_rollout_successes += int(step_successes)

        valid = float(self._real_rollout_episodes > 0)
        step_valid = float(step_episodes > 0)
        success_rate = (
            float(self._real_rollout_successes) / float(self._real_rollout_episodes)
            if self._real_rollout_episodes > 0
            else 0.0
        )
        step_success_rate = (
            float(step_successes) / float(step_episodes)
            if step_episodes > 0
            else 0.0
        )
        metrics = {
            "rollout/episodes": float(self._real_rollout_episodes),
            "rollout/successes": float(self._real_rollout_successes),
            "rollout/success_rate": float(success_rate),
            "rollout/success_rate_valid": float(valid),
            "rollout/step_episodes": float(step_episodes),
            "rollout/step_successes": float(step_successes),
            "rollout/step_success_rate": float(step_success_rate),
            "rollout/step_success_rate_valid": float(step_valid),
        }
        return metrics

    def _eval_interval_global_steps(self) -> int:
        debug_enabled = bool(OmegaConf.select(self.cfg, "training.debug", default=False))
        if debug_enabled:
            debug_interval = OmegaConf.select(
                self.cfg,
                "manual_cotrain.debug_eval_interval_global_steps",
                default=None,
            )
            if debug_interval is not None:
                return _nonnegative_int(
                    debug_interval,
                    "manual_cotrain.debug_eval_interval_global_steps",
                )
        return _nonnegative_int(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.eval_interval_global_steps",
                default=0,
            ),
            "manual_cotrain.eval_interval_global_steps",
        )

    def _ngpu(self) -> int:
        return int(OmegaConf.select(self.cfg, "manual_cotrain.ngpu", default=1))

    def _global_steps(self) -> int:
        return self._positive_manual_int("global_steps", default=1)

    def _sync_every(self) -> int:
        return self._positive_manual_int("sync_every", default=1)

    def _learner_update_step(self) -> int:
        return self._positive_manual_int("learner_update_step", default=1)

    def _learner_update_phase(self) -> str:
        phase = str(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.learner_update_phase",
                default="cotrain",
            )
        ).strip()
        allowed = {"cotrain", "wm", "classifier", "rl"}
        if phase not in allowed:
            raise ValueError(
                "manual_cotrain.learner_update_phase must be one of "
                f"{sorted(allowed)}, got {phase!r}"
            )
        return phase

    def _rollout_epoch(self) -> int:
        return self._positive_manual_int("rollout_epoch", default=1)

    def _real_rollout_epoch(self) -> int:
        return self._positive_manual_int(
            "real_rollout_epoch",
            default=self._rollout_epoch(),
        )

    def _real_env_workers(self) -> int:
        value = int(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.real_env_workers",
                default=1,
            )
        )
        minimum = 1 if self._real_env_enabled() else 0
        if value < minimum:
            relation = "positive" if minimum == 1 else "nonnegative"
            raise ValueError(
                "manual_cotrain.real_env_workers must be "
                f"{relation}, got {value}"
            )
        return value

    def _real_env_enabled(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.real_env_enabled",
                default=True,
            )
        )

    def _learner_updates_enabled(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.learner_updates_enabled",
                default=True,
            )
        )

    def _real_rollout_target_trajectories(self) -> int | None:
        value = OmegaConf.select(
            self.cfg,
            "manual_cotrain.real_rollout_target_trajectories",
            default=None,
        )
        if value is None:
            return None
        target = int(value)
        if target <= 0:
            raise ValueError(
                "manual_cotrain.real_rollout_target_trajectories must be positive, "
                f"got {target}"
            )
        return target

    def _real_rollout_epochs_by_worker(self, worker_count: int) -> list[int]:
        if int(worker_count) <= 0:
            return []
        return self._rollout_epochs_by_worker(
            worker_count,
            target_trajectories=self._real_rollout_target_trajectories(),
            envs_per_worker=self._envs_per_worker(),
            fallback_epoch=self._real_rollout_epoch(),
            target_name="manual_cotrain.real_rollout_target_trajectories",
            envs_name="manual_cotrain.envs_per_worker",
            role_name="real",
        )

    def _wm_rollout_epoch(self) -> int:
        return self._positive_manual_int(
            "wm_rollout_epoch",
            default=self._rollout_epoch(),
        )

    def _wm_rollout_lease_epochs(self) -> int:
        return self._positive_manual_int("wm_rollout_lease_epochs", default=1)

    def _refresh_wm_initial_conditions_per_lease(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.refresh_wm_initial_conditions_per_lease",
                default=False,
            )
        )

    def _wm_rollout_target_trajectories(self) -> int | None:
        value = OmegaConf.select(
            self.cfg,
            "manual_cotrain.wm_rollout_target_trajectories",
            default=None,
        )
        if value is None:
            return None
        target = int(value)
        if target <= 0:
            raise ValueError(
                "manual_cotrain.wm_rollout_target_trajectories must be positive, "
                f"got {target}"
            )
        return target

    def _wm_rollout_epochs_by_worker(self, worker_count: int) -> list[int]:
        return self._rollout_epochs_by_worker(
            worker_count,
            target_trajectories=self._wm_rollout_target_trajectories(),
            envs_per_worker=self._wm_envs_per_worker(),
            fallback_epoch=self._wm_rollout_epoch(),
            target_name="manual_cotrain.wm_rollout_target_trajectories",
            envs_name="manual_cotrain.wm_envs_per_worker",
            role_name="WM",
        )

    @staticmethod
    def _rollout_epochs_by_worker(
        worker_count: int,
        *,
        target_trajectories: int | None,
        envs_per_worker: int,
        fallback_epoch: int,
        target_name: str,
        envs_name: str,
        role_name: str,
    ) -> list[int]:
        workers = int(worker_count)
        if workers <= 0:
            return []
        if target_trajectories is None:
            return [int(fallback_epoch) for _ in range(workers)]
        target = int(target_trajectories)
        envs = int(envs_per_worker)
        if target % envs != 0:
            raise ValueError(
                f"{target_name} must be divisible by {envs_name}; "
                f"got {target} and {envs}"
            )
        total_worker_epochs = target // envs
        if total_worker_epochs < workers:
            raise ValueError(
                f"{target_name} is too small to give each {role_name} worker at "
                "least one rollout_epoch; "
                f"got target={target}, envs_per_worker={envs}, workers={workers}"
            )
        base, extra = divmod(total_worker_epochs, workers)
        return [base + (1 if rank < extra else 0) for rank in range(workers)]

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

    def _wm_envs_per_worker(self) -> int:
        return self._positive_manual_int(
            "wm_envs_per_worker",
            default=self._envs_per_worker(),
        )

    def _rollout_num_slots(self) -> int:
        return max(self._envs_per_worker(), self._wm_envs_per_worker())

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

    def _actor_receive_overlap_enabled(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.actor_receive_overlap",
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

    def _overlap_env_bootstrap_enabled(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.overlap_env_bootstrap",
                default=False,
            )
        )

    def _actor_cpu_offload_enabled(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg,
                "actor.train_cfg.fsdp.cpu_offload",
                default=False,
            )
        )

    def _maybe_prefetch_env_bootstrap(self, real_env: Any) -> Any | None:
        """Dispatch the next round's env reset to overlap actor training.

        Returns the pending group handle (not waited) when overlap is enabled
        and actor CPU offload is off; otherwise ``None``. Mirrors RLinf's guard
        that force-disables bootstrap overlap while the actor is offloaded.
        """

        if real_env is None or not self._overlap_env_bootstrap_enabled():
            return None
        if self._actor_cpu_offload_enabled():
            if not self._overlap_bootstrap_offload_warned:
                _hs_trace(
                    "[manual-cotrain] overlap_env_bootstrap skipped while actor "
                    "cpu_offload is enabled"
                )
                self._overlap_bootstrap_offload_warned = True
            return None
        return real_env.prefetch_bootstrap()

    def _wm_env_write_replay(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.wm_env_write_replay",
                default=False,
            )
        )

    def _save_replay_state(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.save_replay_state",
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

    def _wait_env_metrics_with_dynamic_wm_leases(
        self,
        *,
        real_env_results: list[Any],
        wm_env: Any,
        rollout_result: Any,
        env_channel_name: str,
        rollout_channel_name: str,
        actor_channel_name: str,
        timeout_s: float,
        poll_s: float = 1.0,
        progress: _ManualCotrainEnvProgressMonitor | None = None,
    ) -> dict[str, float]:
        """Run WM imagine as a global lease pool while real env rollout proceeds."""

        wm_workers = _worker_count(wm_env)
        total_wm_epochs = sum(self._wm_rollout_epochs_by_worker(wm_workers))
        lease_epochs = self._wm_rollout_lease_epochs()
        remaining_wm_epochs = int(total_wm_epochs)
        active_wm: dict[int, tuple[Any, int]] = {}
        completed_wm_epochs = 0
        pending_real = list(real_env_results)
        completed_metrics: list[Any] = []
        start = time.monotonic()
        configured_real_workers = (
            min(self._real_env_workers(), self._ngpu())
            if self._real_env_enabled() and self._ngpu() > 0
            else (1 if self._real_env_enabled() else 0)
        )
        real_worker_epochs = self._real_rollout_epochs_by_worker(
            int(configured_real_workers)
        )
        real_total_chunks = (
            sum(real_worker_epochs)
            * self._envs_per_worker()
            * _chunk_steps(
                self._max_steps_per_rollout_epoch(),
                self._num_action_chunks(),
            )
        )
        wm_chunks_per_epoch = self._wm_envs_per_worker() * _chunk_steps(
            self._wm_max_steps_per_rollout_epoch(),
            self._num_action_chunks(),
        )
        wm_total_chunks = int(total_wm_epochs) * int(wm_chunks_per_epoch)
        real_completed = not pending_real

        def check_rollout_failure() -> None:
            ready = rollout_result.ready()
            if ready:
                values = rollout_result.wait_refs(ready)
                raise RuntimeError(
                    "RolloutGroup.generate completed before EnvGroup.interact; "
                    f"ready_result={values!r}"
                )

        def start_wm_lease(rank: int) -> None:
            nonlocal remaining_wm_epochs
            if remaining_wm_epochs <= 0:
                return
            lease = min(int(lease_epochs), int(remaining_wm_epochs))
            _hs_trace(
                f"[wm env rank={int(rank)}] start dynamic imagine lease "
                f"rollout_epoch={int(lease)} remaining_before={int(remaining_wm_epochs)}"
            )
            if self._refresh_wm_initial_conditions_per_lease():
                wm_env.execute_on(int(rank)).refresh_wm_initial_conditions().wait()
            wm_env.execute_on(int(rank)).configure_rollout_epoch(int(lease)).wait()
            active_wm[int(rank)] = (
                wm_env.execute_on(int(rank)).interact(
                    env_channel_name,
                    rollout_channel_name,
                    actor_channel_name,
                ),
                int(lease),
            )
            remaining_wm_epochs -= int(lease)

        def progress_records() -> list[dict[str, Any]]:
            if progress is None:
                return []
            records = getattr(progress, "records", None)
            if callable(records):
                return list(records())
            return []

        def central_progress_snapshot() -> _ManualCotrainProgressSnapshot:
            records = progress_records()
            global_steps = {
                int(record.get("global_step", 0) or 0)
                for record in records
                if "global_step" in record
            }
            real_done = int(real_total_chunks if real_completed else 0)
            real_observed_done = 0
            for record in records:
                if str(record.get("role", "")) != "real_env":
                    continue
                item_total = max(0, int(record.get("total", 0) or 0))
                item_done = max(0, int(record.get("done", 0) or 0))
                if item_total > 0:
                    real_observed_done += min(item_done, item_total)
                else:
                    real_observed_done += item_done
            if not real_completed:
                real_done = min(int(real_total_chunks), int(real_observed_done))

            active_wm_chunks = 0
            active_wm_records: list[dict[str, Any]] = []
            for record in records:
                if str(record.get("role", "")) != "wm_env":
                    continue
                rank = int(record.get("rank", record.get("env_rank", -1)) or -1)
                if rank not in active_wm:
                    continue
                active_wm_records.append(record)
                item_total = max(0, int(record.get("total", 0) or 0))
                item_done = max(0, int(record.get("done", 0) or 0))
                active_wm_chunks += min(item_done, item_total) if item_total > 0 else item_done

            wm_done = min(
                int(wm_total_chunks),
                int(completed_wm_epochs) * int(wm_chunks_per_epoch) + int(active_wm_chunks),
            )
            done = min(
                int(real_total_chunks) + int(wm_total_chunks),
                int(real_done) + int(wm_done),
            )
            total = int(real_total_chunks) + int(wm_total_chunks)
            parts: list[str] = []
            if len(global_steps) == 1:
                parts.append(f"global_step={next(iter(global_steps))}")
            elif global_steps:
                parts.append(f"global_steps={min(global_steps)}-{max(global_steps)}")
            running_wm_epochs = sum(
                int(lease) for _result, lease in active_wm.values()
            )
            if (
                int(completed_wm_epochs)
                + int(running_wm_epochs)
                + int(remaining_wm_epochs)
                != int(total_wm_epochs)
            ):
                raise RuntimeError("dynamic WM lease accounting is not conserved")
            if real_total_chunks > 0:
                parts.append(
                    f"real_chunks={int(real_done)}/{int(real_total_chunks)}"
                )
            parts.extend(
                [
                    f"wm_chunks={int(wm_done)}/{int(wm_total_chunks)}",
                    f"wm_leases_done={int(completed_wm_epochs)}",
                    f"wm_leases_running={int(running_wm_epochs)}",
                    f"wm_leases_queued={int(remaining_wm_epochs)}",
                    f"wm_leases_total={int(total_wm_epochs)}",
                ]
            )
            parts.extend(
                _classifier_progress_status_parts(
                    active_wm_records,
                    metrics=_sum_metric_lists(completed_metrics),
                )
            )
            has_real_role = real_total_chunks > 0
            finished = (
                int(has_real_role and real_done >= real_total_chunks)
                + int(wm_done >= wm_total_chunks)
            )
            return _ManualCotrainProgressSnapshot(
                done=done,
                total=total,
                status=" ".join(parts),
                worker_count=int(has_real_role) + 1,
                finished_count=finished,
            )

        def report_central_progress(*, force: bool = False) -> _ManualCotrainProgressSnapshot | None:
            if progress is None:
                return None
            snapshot = central_progress_snapshot()
            report_snapshot = getattr(progress, "report_snapshot", None)
            if callable(report_snapshot):
                report_snapshot(snapshot, force=force)
            else:
                progress.report(force=force)
            return snapshot

        for rank in range(min(int(wm_workers), int(remaining_wm_epochs))):
            start_wm_lease(rank)

        report_central_progress(force=True)
        while pending_real or active_wm:
            report_central_progress()
            check_rollout_failure()

            next_pending_real: list[Any] = []
            for result in pending_real:
                if result.done():
                    real_completed = True
                    completed_metrics.append(result.wait())
                else:
                    next_pending_real.append(result)
            pending_real = next_pending_real

            completed_wm_ranks: list[int] = []
            for rank, (result, _lease) in list(active_wm.items()):
                if not result.done():
                    continue
                completed_metrics.append(result.wait())
                completed_wm_ranks.append(int(rank))
            for rank in completed_wm_ranks:
                _result, lease = active_wm[int(rank)]
                completed_wm_epochs += int(lease)
                active_wm.pop(int(rank), None)
            for rank in completed_wm_ranks:
                start_wm_lease(int(rank))

            if not pending_real and not active_wm:
                break
            if timeout_s > 0 and (time.monotonic() - start) > float(timeout_s):
                snapshot = report_central_progress(force=True)
                progress_suffix = (
                    f" Current manual cotrain progress: {snapshot.status}."
                    if snapshot is not None and snapshot.status
                    else ""
                )
                active_ranks = ",".join(str(rank) for rank in sorted(active_wm)) or "none"
                raise TimeoutError(
                    "EnvGroup.interact did not finish before "
                    f"manual_cotrain.env_rollout_timeout_s={float(timeout_s):.1f}s; "
                    "dynamic WM imagine leases are still running or "
                    "RolloutGroup.generate is waiting for StopMsg. "
                    f"active_wm_ranks={active_ranks}, "
                    f"remaining_wm_rollout_epochs={int(remaining_wm_epochs)}. "
                    "Set DVLA_COTRAIN_HANDSHAKE_TRACE=1 before launching to log "
                    "EnvGroup/RolloutGroup action handshakes."
                    f"{progress_suffix}"
                )
            time.sleep(max(0.0, float(poll_s)))

        report_central_progress(force=True)
        return _sum_metric_lists(completed_metrics)

    def _receive_actor_trajectories(
        self,
        groups: dict[str, Any],
        *,
        expected_shards: int,
        env_metrics: dict[str, float],
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
        role_counts = self._actor_shard_role_counts_from_metrics(
            groups,
            env_metrics,
            expected_shards=expected,
        )
        counts = self._actor_direct_receive_keyed_counts(
            groups,
            role_counts=role_counts,
        )
        if counts is not None:
            recv_stage_start = time.perf_counter()
            ready_start = time.perf_counter()
            self._wait_actor_channel_role_counts(
                groups["actor_channel"],
                role_counts,
                timeout_s=self._actor_channel_timeout_s(),
            )
            stage_times["time/manual_cotrain/actor_channel_wait_ready_s"] = float(
                time.perf_counter() - ready_start
            )
            recv_results = [
                actor.execute_on(rank).recv_rollout_trajectories(
                    actor_channel_name,
                    keyed_counts=count,
                )
                for rank, count in enumerate(counts)
            ]
            metrics = _sum_metric_lists([result.wait() for result in recv_results])
            if "actor/channel_get_batch_s" in metrics:
                stage_times["time/manual_cotrain/actor_channel_get_batch_s"] = float(
                    metrics["actor/channel_get_batch_s"]
                )
            if "actor/load_trajectory_shards_s" in metrics:
                stage_times[
                    "time/manual_cotrain/actor_load_trajectory_shards_s"
                ] = float(metrics["actor/load_trajectory_shards_s"])
            stage_times["time/manual_cotrain/actor_recv_rollout_trajectories_s"] = float(
                time.perf_counter() - recv_stage_start
            )
            return metrics

        ready_start = time.perf_counter()
        self._wait_actor_channel_role_counts(
            groups["actor_channel"],
            role_counts,
            timeout_s=self._actor_channel_timeout_s(),
        )
        stage_times["time/manual_cotrain/actor_channel_wait_ready_s"] = float(
            time.perf_counter() - ready_start
        )
        channel_get_start = time.perf_counter()
        shards = self._get_actor_shards_by_role(groups["actor_channel"], role_counts)
        stage_times["time/manual_cotrain/actor_channel_get_batch_s"] = float(
            time.perf_counter() - channel_get_start
        )
        load_start = time.perf_counter()
        actor.load_trajectory_shards(shards).wait()
        stage_times["time/manual_cotrain/actor_load_trajectory_shards_s"] = float(
            time.perf_counter() - load_start
        )
        return {"actor/received_shards": float(len(shards))}

    def _actor_channel_timeout_s(self) -> float:
        return float(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.actor_channel_timeout_s",
                default=self._env_rollout_timeout_s(),
            )
            or 0.0
        )

    def _start_actor_trajectory_receivers(
        self,
        groups: dict[str, Any],
        *,
        expected_shards: int,
        actor_channel_name: str,
    ) -> dict[str, Any] | None:
        role_counts = self._configured_actor_shard_role_counts(groups)
        if sum(count for _key, count in role_counts) != max(0, int(expected_shards)):
            return None
        counts = self._actor_direct_receive_keyed_counts(
            groups,
            role_counts=role_counts,
        )
        if counts is None:
            return None
        actor = groups["ActorGroup"]
        started_at = time.perf_counter()
        recv_results = [
            actor.execute_on(rank).recv_rollout_trajectories(
                actor_channel_name,
                keyed_counts=count,
            )
            for rank, count in enumerate(counts)
        ]
        return {
            "started_at": float(started_at),
            "expected_shards": int(expected_shards),
            "results": recv_results,
        }

    def _actor_direct_receive_keyed_counts(
        self,
        groups: dict[str, Any],
        role_counts: list[tuple[str, int]],
    ) -> list[list[tuple[str, int]]] | None:
        actor = groups["ActorGroup"]
        actor_ranks = len(getattr(actor, "workers", []) or [])
        if actor_ranks <= 0:
            return None
        counts_by_key = {str(key): max(0, int(count)) for key, count in role_counts}
        try:
            counts = _split_actor_keyed_shard_counts(
                real_shards=0,
                wm_shards=counts_by_key.get("wm_env", 0),
                wm_shard_batch_size=self._wm_envs_per_worker(),
                actor_ranks=actor_ranks,
                group_size=self._actor_group_size(),
            )
        except ValueError:
            return None
        if any(not rank_counts for rank_counts in counts):
            return None
        return counts

    def _configured_actor_shard_role_counts(
        self,
        groups: dict[str, Any],
    ) -> list[tuple[str, int]]:
        wm_workers = _worker_count(groups.get("WMEnvGroup"))
        wm_shards = sum(self._wm_rollout_epochs_by_worker(wm_workers))
        return [
            ("wm_env", int(wm_shards)),
        ]

    def _actor_shard_role_counts_from_metrics(
        self,
        groups: dict[str, Any],
        env_metrics: dict[str, float],
        *,
        expected_shards: int,
    ) -> list[tuple[str, int]]:
        wm_shards = int(env_metrics.get("env/wm_env/trajectory_shards", 0.0))
        if wm_shards:
            return [("wm_env", max(0, wm_shards))]
        real_shards = int(env_metrics.get("env/real_env/trajectory_shards", 0.0))
        if real_shards:
            return [("real_env", max(0, real_shards))]
        configured = self._configured_actor_shard_role_counts(groups)
        if sum(count for _key, count in configured) == max(0, int(expected_shards)):
            return configured
        return [("default", max(0, int(expected_shards)))]

    @staticmethod
    def _get_actor_shards_by_role(
        actor_channel: Any,
        role_counts: list[tuple[str, int]],
    ) -> list[Any]:
        shards: list[Any] = []
        for key, count in role_counts:
            count = max(0, int(count))
            if count <= 0:
                continue
            get_batch = getattr(actor_channel, "get_batch", None)
            if callable(get_batch):
                shards.extend(get_batch(count, key=str(key)))
            else:
                shards.extend(actor_channel.get(key=str(key)) for _ in range(count))
        return shards

    @staticmethod
    def _wait_actor_channel_role_counts(
        actor_channel: Any,
        role_counts: list[tuple[str, int]],
        *,
        timeout_s: float,
        poll_s: float = 0.5,
    ) -> dict[str, int]:
        expected_by_key: dict[str, int] = {}
        for key, count in role_counts:
            expected_by_key[str(key)] = expected_by_key.get(str(key), 0) + max(
                0,
                int(count),
            )
        expected_by_key = {
            key: count for key, count in expected_by_key.items() if count > 0
        }
        if not expected_by_key:
            return {}
        qsize = getattr(actor_channel, "qsize", None)
        if not callable(qsize):
            return {}

        start = time.monotonic()
        latest: dict[str, int] = {key: 0 for key in expected_by_key}
        while True:
            ready = True
            for key, expected in expected_by_key.items():
                latest[key] = int(qsize(key=str(key)))
                if latest[key] < expected:
                    ready = False
            if ready:
                return latest
            if timeout_s > 0 and (time.monotonic() - start) > float(timeout_s):
                status = ", ".join(
                    f"{key}:qsize={latest[key]}/expected={expected}"
                    for key, expected in sorted(expected_by_key.items())
                )
                raise TimeoutError(
                    "actor channel did not receive expected trajectory shards before "
                    f"manual_cotrain.actor_channel_timeout_s={float(timeout_s):.1f}s; "
                    f"{status}"
                )
            time.sleep(max(0.0, float(poll_s)))

    def _configured_expected_trajectory_shards(self, groups: dict[str, Any]) -> int:
        return int(
            sum(count for _key, count in self._configured_actor_shard_role_counts(groups))
        )

    def _render_backend(self) -> str:
        return str(
            OmegaConf.select(
                self.cfg,
                "render_backend",
                default=OmegaConf.select(self.cfg, "env.render_backend", default="osmesa"),
            )
        ).strip().lower()

    def _real_render_backend(self) -> str | None:
        value = OmegaConf.select(
            self.cfg,
            "manual_cotrain.real_render_backend",
            default=None,
        )
        if value is None:
            return None
        return str(value).strip().lower()

    def _real_env_cfg(self) -> dict[str, Any]:
        cfg = self._cfg_dict("env.real.cfg")
        real_render_backend = self._real_render_backend()
        if real_render_backend is not None:
            cfg["render_backend"] = real_render_backend
        elif "render_backend" not in cfg:
            cfg["render_backend"] = self._render_backend()
        cfg.setdefault("num_envs_per_worker", self._envs_per_worker())
        if str(cfg.get("render_backend", "osmesa")).strip().lower() == "egl":
            self._ensure_real_env_render_gpu_pool(cfg)
        return cfg

    def _ensure_real_env_render_gpu_pool(self, cfg: dict[str, Any]) -> None:
        for key in ("gpu_pool", "render_devices", "egl_device_pool"):
            if parse_device_ids(cfg.get(key)):
                return

        real_specs = [
            spec for spec in self._placement_plan().env_specs if spec.role == "real_env"
        ]
        devices = cuda_visible_devices_from_env()
        if devices:
            limit = max(1, len(real_specs))
            cfg["gpu_pool"] = devices[:limit]
            return

        placement_devices = [
            int(gpu)
            for spec in real_specs
            for gpu in spec.gpu_ids
        ]
        if not placement_devices:
            raise ValueError(_ZERO_GPU_EGL_ERROR)
        cfg["gpu_pool"] = placement_devices

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
        resume_step = int(payload.get("global_step", 0) or 0)
        replay_state = payload.get("replay")
        replay_sampling_state = payload.get("replay_sampling_state")
        if (replay_state is not None or replay_sampling_state is not None) and not bool(
            groups.get("replay_resume_restored", False)
        ):
            replay_group = groups.get("ReplayGroup")
            if replay_group is None:
                raise ValueError(
                    "manual checkpoint contains replay state but active config has no "
                    "ReplayGroup"
                )
            if replay_state is not None:
                replay_group.load_state_dict(dict(replay_state)).wait()
            if replay_sampling_state is not None:
                replay_group.load_sampling_state_dict(
                    dict(replay_sampling_state)
                ).wait()

        wm_env = groups.get("WMEnvGroup")
        state_dicts = payload.get("state_dicts", {})
        if not self._learner_updates_enabled():
            payload_hashes = dict(payload.get("frozen_state_hashes", {}) or {})
            if payload_hashes != self._frozen_state_hashes:
                raise RuntimeError(
                    "resume frozen hashes differ from the explicit WM/CLS checkpoints"
                )
            payload_sources = dict(payload.get("source_checkpoints", {}) or {})
            if payload_sources != self._frozen_source_checkpoints:
                raise RuntimeError(
                    "resume frozen sources differ from the explicit WM/CLS checkpoints"
                )
            payload_threshold = payload.get("classifier_threshold")
            if (
                payload_threshold is None
                or self._frozen_classifier_threshold is None
                or float(payload_threshold) != float(self._frozen_classifier_threshold)
            ):
                raise RuntimeError(
                    "resume classifier threshold differs from the explicit checkpoint"
                )
            if isinstance(state_dicts, dict) and any(
                name in state_dicts for name in ("world_model", "classifier")
            ):
                raise RuntimeError(
                    "frozen Ray resume checkpoint must not embed WM/classifier states"
                )
            self._assert_frozen_component_hashes(groups)
        elif wm_env is not None and isinstance(state_dicts, dict):
            component_states = {
                name: dict(state_dicts.get(name, {}))
                for name in ("world_model", "classifier")
                if isinstance(state_dicts.get(name), dict)
            }
            if "classifier_threshold" in payload:
                component_states["classifier_threshold"] = float(
                    payload["classifier_threshold"]
                )
            if component_states:
                shared_component_states = _share_ray_value(
                    component_states,
                    cluster=groups.get("cluster"),
                )
                wm_env.load_component_states(
                    shared_component_states,
                    resume_step,
                ).wait()

    def _maybe_save_manual_checkpoint(
        self,
        groups: dict[str, Any],
        global_step: int,
        metrics: dict[str, float],
        *,
        force: bool = False,
    ) -> Path | None:
        interval = int(
            OmegaConf.select(
                self.cfg,
                "manual_cotrain.checkpoint_every",
                default=0,
            )
            or 0
        )
        if not force and (interval <= 0 or int(global_step) % interval != 0):
            return None
        if not self._learner_updates_enabled():
            self._assert_frozen_component_hashes(groups)
        ckpt_dir = (
            self.get_checkpoint_dir() / f"manual_cotrain_step_{int(global_step)}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        actor_state = _first_nonempty_mapping(groups["ActorGroup"].state_dict().wait())
        if not self._learner_updates_enabled():
            self._policy_final_hash = state_dict_sha256(actor_state)
            if not self._policy_initial_hash:
                self._policy_initial_hash = self._policy_final_hash
        learner_group = groups.get("LearnerGroup")
        learner_states: dict[str, Any] = {}
        if learner_group is not None:
            raw_learner_states = learner_group.state_dicts(True).wait()[0]
            if not isinstance(raw_learner_states, dict):
                raise TypeError("LearnerGroup.state_dicts() must return a mapping")
            learner_states = raw_learner_states
        replay_group = groups.get("ReplayGroup")
        replay_state = None
        replay_sampling_state = None
        if replay_group is not None:
            replay_sampling_state = _first_result(
                replay_group.sampling_state_dict().wait()
            )
            if not isinstance(replay_sampling_state, dict):
                raise TypeError(
                    "ReplayGroup.sampling_state_dict() must return a mapping"
                )
        if replay_group is not None and self._save_replay_state():
            replay_state = replay_group.state_dict().wait()[0]
        ckpt_path = ckpt_dir / "manual_cotrain.ckpt"
        state_dicts = {"policy": dict(actor_state)}
        optimizer_state = _first_nonempty_mapping(
            groups["ActorGroup"].optimizer_state_dict().wait()
        )
        if not optimizer_state:
            raise RuntimeError("manual cotrain policy checkpoint has no optimizer state")
        state_dicts["policy_optimizer"] = optimizer_state
        if learner_group is not None:
            for name in (
                "world_model",
                "classifier",
                "world_model_optimizer",
                "classifier_optimizer",
            ):
                state = learner_states.get(name)
                if isinstance(state, dict) and state:
                    state_dicts[name] = dict(state)
        classifier_threshold = (
            float(learner_states["classifier_threshold"])
            if "classifier_threshold" in learner_states
            else self._frozen_classifier_threshold
        )
        payload = {
            "global_step": int(global_step),
            "cfg": _plain(self.cfg),
            "metrics": dict(metrics),
            "state_dicts": state_dicts,
            "replay": replay_state,
            "replay_sampling_state": replay_sampling_state,
        }
        if not self._learner_updates_enabled():
            payload["frozen_state_hashes"] = dict(self._frozen_state_hashes)
            payload["source_checkpoints"] = dict(
                self._frozen_source_checkpoints
            )
            payload["policy_initial_hash"] = str(self._policy_initial_hash)
            payload["policy_final_hash"] = str(self._policy_final_hash)
            payload["applied_policy_steps"] = int(self._applied_policy_steps)
        if classifier_threshold is not None:
            payload["classifier_threshold"] = classifier_threshold
        torch.save(
            payload,
            ckpt_path,
        )
        run_metadata = self._manual_checkpoint_run_metadata(ckpt_dir)
        manifest = _manual_checkpoint_manifest(
            global_step=int(global_step),
            metrics=metrics,
            ckpt_name=ckpt_path.name,
            state_dicts={
                name: state
                for name, state in state_dicts.items()
                if not name.endswith("_optimizer")
            },
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
                state_dicts={
                    name: state
                    for name, state in state_dicts.items()
                    if not name.endswith("_optimizer")
                },
                has_replay=replay_state is not None,
                run=self._manual_checkpoint_run_metadata(canonical_dir),
            )
            _atomic_write_json(
                canonical_dir / "manual_cotrain_manifest.json",
                alias_manifest,
            )
        return ckpt_path

    def _finalize_frozen_policy_run(
        self,
        groups: dict[str, Any],
        *,
        global_step: int,
        metrics: dict[str, float | int],
    ) -> None:
        checkpoint = (
            self.get_checkpoint_dir()
            / f"manual_cotrain_step_{int(global_step)}"
            / "manual_cotrain.ckpt"
        )
        if checkpoint.is_file():
            self._assert_frozen_component_hashes(groups)
        else:
            checkpoint = self._maybe_save_manual_checkpoint(
                groups,
                int(global_step),
                {str(key): float(value) for key, value in metrics.items()},
                force=True,
            )
        require_update = bool(
            OmegaConf.select(
                self.cfg,
                "training.require_policy_update",
                default=True,
            )
        )
        policy_changed = self._policy_final_hash != self._policy_initial_hash
        if require_update and self._applied_policy_steps <= 0:
            raise RuntimeError("no policy optimizer step was applied")
        if require_update and not policy_changed:
            raise RuntimeError("policy state did not change during frozen Ray RL")

        summary = {
            "schema_version": 1,
            "execution": "ray_manual_policy_only",
            "ngpu": int(self._ngpu()),
            "official_data_dir": _select_plain_str(self.cfg, "replay.seed.data_dir"),
            "official_hidden_dir": _select_plain_str(
                self.cfg,
                "replay.seed.hidden_dir",
            ),
            "source_checkpoints": dict(self._frozen_source_checkpoints),
            "frozen_hashes_before": dict(self._frozen_state_hashes),
            "frozen_hashes_after": dict(self._frozen_state_hashes),
            "policy_hash_before": str(self._policy_initial_hash),
            "policy_hash_after": str(self._policy_final_hash),
            "policy_changed": bool(policy_changed),
            "applied_policy_steps": int(self._applied_policy_steps),
            "total_updates": int(global_step),
            "classifier_threshold": self._frozen_classifier_threshold,
            "final_checkpoint": str(checkpoint) if checkpoint is not None else None,
            "last_metrics": dict(metrics),
        }
        _atomic_write_json(
            self.get_run_dir() / "frozen_rl_summary.json",
            summary,
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
        return ResourceMapPlacementStrategy(
            f"{_format_resource_group(tuple(int(gpu) for gpu in gpu_ids))}:0"
        )

    @staticmethod
    def _placement_for_gpus(gpu_ids: list[int]) -> NodePlacementStrategy | ResourceMapPlacementStrategy:
        ids = [int(gpu) for gpu in gpu_ids]
        if not ids:
            return NodePlacementStrategy(1)
        return ResourceMapPlacementStrategy(",".join(str(gpu) for gpu in ids))

    @staticmethod
    def _resource_map_for_specs(specs: list[Any]) -> NodePlacementStrategy | ResourceMapPlacementStrategy:
        if not specs or not specs[0].gpu_ids:
            return NodePlacementStrategy(max(1, len(specs)))
        segments: list[str] = []
        index = 0
        while index < len(specs):
            group = tuple(int(gpu) for gpu in specs[index].gpu_ids)
            if not group:
                return NodePlacementStrategy(max(1, len(specs)))
            stop = index + 1
            while stop < len(specs) and tuple(int(gpu) for gpu in specs[stop].gpu_ids) == group:
                stop += 1
            resource = _format_resource_group(group)
            processes = _format_process_range(index, stop - 1)
            segments.append(f"{resource}:{processes}")
            index = stop
        return ResourceMapPlacementStrategy(",".join(segments))


def _format_resource_group(gpu_ids: tuple[int, ...]) -> str:
    if not gpu_ids:
        raise ValueError("resource group must not be empty")
    if len(gpu_ids) == 1:
        return str(gpu_ids[0])
    if any(right != left + 1 for left, right in zip(gpu_ids, gpu_ids[1:], strict=False)):
        raise ValueError(
            f"manual cotrain resource groups must be contiguous, got {list(gpu_ids)}"
        )
    return f"{gpu_ids[0]}-{gpu_ids[-1]}"


def _format_process_range(start: int, end: int) -> str:
    if end < start:
        raise ValueError(f"invalid process range {start}-{end}")
    if start == end:
        return str(start)
    return f"{start}-{end}"


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
    aliases = {
        "cls/updated": "train/classifier_updated",
        "cls/updates": "train/classifier_updates",
        "cls/acc": "train/classifier_acc",
        "cls/f1": "train/classifier_f1",
        "env/wm_env/classifier_success_rate": "train/wm_env_classifier_success_rate",
        "env/wm_env/classifier_trajectory_success_rate": (
            "train/wm_env_classifier_trajectory_success_rate"
        ),
    }
    for source, target in aliases.items():
        if source in out and target not in out:
            out[target] = float(out[source])
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
        if key.endswith("/score_mean") or key.endswith("/score_p50") or key.endswith(
            "/score_p90"
        ):
            continue
        if key.endswith("/classifier_success_rate") or key.endswith(
            "/classifier_trajectory_success_rate"
        ):
            continue
        if key.endswith("/batch_size_min"):
            summed[key] = float(min(values))
        elif key.endswith("/batch_size_max"):
            summed[key] = float(max(values))
        elif key.endswith("/score_max"):
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
    for key, value in list(summed.items()):
        if not key.endswith("/score_sum"):
            continue
        prefix = key[: -len("score_sum")]
        count = summed.get(f"{prefix}score_count", 0.0)
        if count > 0:
            summed[f"{prefix}score_mean"] = float(value / count)
    for suffix in ("score_p50", "score_p90"):
        for key, values in values_by_key.items():
            if not key.endswith(f"/{suffix}"):
                continue
            prefix = key[: -len(suffix)]
            counts = values_by_key.get(f"{prefix}score_count", [])
            if len(counts) == len(values) and sum(counts) > 0.0:
                # Worker-local quantiles cannot be merged exactly without raw
                # samples; count-weighted values keep the debug metric bounded.
                total = float(sum(counts))
                summed[key] = float(
                    sum(
                        value * count
                        for value, count in zip(values, counts, strict=True)
                    )
                    / total
                )
            else:
                summed[key] = float(max(values))
    _derive_classifier_rate_metrics(summed)
    return summed


def _derive_classifier_rate_metrics(metrics: dict[str, float]) -> None:
    for key, total_chunks in list(metrics.items()):
        if not key.endswith("/classifier_total_chunks"):
            continue
        prefix = key[: -len("classifier_total_chunks")]
        if total_chunks > 0.0:
            metrics[f"{prefix}classifier_success_rate"] = float(
                metrics.get(f"{prefix}classifier_success_chunks", 0.0)
                / float(total_chunks)
            )
    for key, total_trajectories in list(metrics.items()):
        if not key.endswith("/classifier_total_trajectories"):
            continue
        prefix = key[: -len("classifier_total_trajectories")]
        if total_trajectories > 0.0:
            metrics[f"{prefix}classifier_trajectory_success_rate"] = float(
                metrics.get(f"{prefix}classifier_success_trajectories", 0.0)
                / float(total_trajectories)
            )


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
        if key in {
            "actor/trajectory_count",
            "actor/received_shards",
            "actor/loss_mask_sum",
            "actor/reward_filtered_rollouts",
        }:
            aggregated[key] = float(sum(values))
        elif key in {
            "actor/ppo_updates",
            "actor/ppo_optimizer_steps",
            "actor/ppo_forward_backward_steps",
            "actor/ppo_progress_ops",
            "actor/global_time_steps",
            "actor/global_rollout_trajectories",
            "actor/global_ppo_samples",
            "actor/global_batch_size",
            "actor/per_rank_global_batch_size",
            "actor/micro_batch_size",
            "actor/global_loss_mask_sum",
            "actor/global_logprob_token_count",
        }:
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


def _split_actor_keyed_shard_counts(
    *,
    real_shards: int,
    wm_shards: int,
    wm_shard_batch_size: int,
    actor_ranks: int,
    group_size: int,
) -> list[list[tuple[str, int]]]:
    """Split actor-channel role-keyed shard counts by trajectory columns.

    Real env shards carry one trajectory column. Batched WM shards carry
    ``wm_shard_batch_size`` columns. GRPO grouping constrains the resulting
    trajectory columns on each actor rank, not the raw actor-channel message count.
    """

    ranks = int(actor_ranks)
    group = int(group_size)
    wm_batch = int(wm_shard_batch_size)
    real_total = max(0, int(real_shards))
    wm_total = max(0, int(wm_shards))
    if ranks <= 0:
        raise ValueError("actor_ranks must be positive")
    if group <= 0:
        raise ValueError("group_size must be positive")
    if wm_batch <= 0:
        raise ValueError("wm_shard_batch_size must be positive")

    if real_total:
        raise ValueError(
            "real_env trajectories must not enter ActorGroup; "
            "write them to replay for learner WM/classifier updates instead"
        )

    total_trajectories = wm_total * wm_batch
    if total_trajectories <= 0:
        return [[] for _ in range(ranks)]
    if total_trajectories % group != 0:
        raise ValueError(
            "actor trajectory count must be divisible by group_size; "
            f"got {total_trajectories} and {group}"
        )
    shard_group = group // gcd(group, wm_batch)
    if wm_total % shard_group != 0:
        raise ValueError(
            "wm_env actor shard count must form complete group_size blocks; "
            f"got wm_shards={wm_total}, wm_shard_batch_size={wm_batch}, "
            f"group_size={group}"
        )

    wm_counts = [0 for _ in range(ranks)]
    trajectory_counts = [0 for _ in range(ranks)]

    for _ in range(wm_total // shard_group):
        rank = min(range(ranks), key=lambda item: (trajectory_counts[item], item))
        wm_counts[rank] += shard_group
        trajectory_counts[rank] += shard_group * wm_batch

    out: list[list[tuple[str, int]]] = []
    for wm_count, trajectory_count in zip(
        wm_counts,
        trajectory_counts,
        strict=True,
    ):
        if trajectory_count % group != 0:
            raise ValueError(
                "actor keyed shard split produced a non-divisible trajectory count; "
                f"got {trajectory_count} and {group}"
            )
        rank_counts: list[tuple[str, int]] = []
        if wm_count:
            rank_counts.append(("wm_env", int(wm_count)))
        out.append(rank_counts)
    return out


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


@dataclass(frozen=True)
class _ManualCotrainProgressSnapshot:
    done: int
    total: int
    status: str | None
    worker_count: int
    finished_count: int


class _ManualCotrainEnvProgressMonitor:
    def __init__(
        self,
        progress_dir: str | Path,
        console_progress: Any,
        *,
        desc: str = "manual-cotrain-env",
    ) -> None:
        self.progress_dir = Path(progress_dir)
        self._console_progress = console_progress
        self._desc = str(desc)

    def report(self, *, force: bool = False) -> _ManualCotrainProgressSnapshot:
        snapshot = _read_manual_cotrain_progress_snapshot(self.progress_dir)
        if snapshot.total > 0:
            self._console_progress(
                snapshot.done,
                snapshot.total,
                self._desc,
                unit="chunk",
                status=snapshot.status,
                force=force,
            )
        return snapshot

    def records(self) -> list[dict[str, Any]]:
        return _read_manual_cotrain_progress_records(self.progress_dir)

    def report_snapshot(
        self,
        snapshot: _ManualCotrainProgressSnapshot,
        *,
        force: bool = False,
    ) -> _ManualCotrainProgressSnapshot:
        if snapshot.total > 0:
            self._console_progress(
                snapshot.done,
                snapshot.total,
                self._desc,
                unit="chunk",
                status=snapshot.status,
                force=force,
            )
        return snapshot


def _read_manual_cotrain_progress_records(
    progress_dir: str | Path,
) -> list[dict[str, Any]]:
    path = Path(progress_dir)
    records: list[dict[str, Any]] = []
    if path.is_dir():
        for file in sorted(path.glob("*.json")):
            try:
                payload = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _read_manual_cotrain_progress_snapshot(
    progress_dir: str | Path,
) -> _ManualCotrainProgressSnapshot:
    records = _read_manual_cotrain_progress_records(progress_dir)
    if not records:
        return _ManualCotrainProgressSnapshot(
            done=0,
            total=0,
            status=None,
            worker_count=0,
            finished_count=0,
        )

    def sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
        role = str(record.get("role", ""))
        role_order = 0 if role == "real_env" else 1
        return (role_order, int(record.get("env_rank", record.get("rank", 0))), role)

    records.sort(key=sort_key)
    done = 0
    total = 0
    finished = 0
    global_steps: set[int] = set()
    parts: list[str] = []
    for record in records:
        item_done = max(0, int(record.get("done", 0) or 0))
        item_total = max(0, int(record.get("total", 0) or 0))
        done += min(item_done, item_total) if item_total > 0 else item_done
        total += item_total
        finished += int(bool(record.get("finished", False)))
        if "global_step" in record:
            global_steps.add(int(record.get("global_step", 0) or 0))
        role = str(record.get("role", "env"))
        env_rank = int(record.get("env_rank", record.get("rank", 0)) or 0)
        parts.append(f"{role}#{env_rank}={item_done}/{item_total}")
    parts.extend(_classifier_progress_status_parts(records))
    prefix: list[str] = []
    if len(global_steps) == 1:
        prefix.append(f"global_step={next(iter(global_steps))}")
    elif global_steps:
        prefix.append(f"global_steps={min(global_steps)}-{max(global_steps)}")
    prefix.append(f"finished={finished}/{len(records)}")
    status = " ".join([*prefix, *parts])
    return _ManualCotrainProgressSnapshot(
        done=done,
        total=total,
        status=status,
        worker_count=len(records),
        finished_count=finished,
    )


def _classifier_progress_status_parts(
    records: list[dict[str, Any]],
    *,
    metrics: dict[str, float] | None = None,
) -> list[str]:
    metric_values = metrics or {}

    def metric_counter(name: str) -> int:
        for prefix in ("env/wm_env/", "env/"):
            key = f"{prefix}{name}"
            if key in metric_values:
                return max(0, int(float(metric_values.get(key, 0.0) or 0.0)))
        return 0

    success_chunks = sum(
        max(0, int(record.get("classifier_success_chunks", 0) or 0))
        for record in records
    ) + metric_counter("classifier_success_chunks")
    total_chunks = sum(
        max(0, int(record.get("classifier_total_chunks", 0) or 0))
        for record in records
    ) + metric_counter("classifier_total_chunks")
    success_trajectories = sum(
        max(0, int(record.get("classifier_success_trajectories", 0) or 0))
        for record in records
    ) + metric_counter("classifier_success_trajectories")
    total_trajectories = sum(
        max(0, int(record.get("classifier_total_trajectories", 0) or 0))
        for record in records
    ) + metric_counter("classifier_total_trajectories")
    parts: list[str] = []
    if total_chunks > 0:
        parts.append(
            "wm_cls_chunk_positive_rate="
            f"{float(success_chunks) / float(total_chunks):.3f}"
        )
    if total_trajectories > 0:
        parts.append(
            "wm_cls_trajectory_positive_rate="
            f"{float(success_trajectories) / float(total_trajectories):.3f}"
        )
    return parts


def _wait_env_metrics_with_rollout_guard(
    env_results: list[Any],
    rollout_result: Any,
    *,
    timeout_s: float,
    poll_s: float = 1.0,
    progress: _ManualCotrainEnvProgressMonitor | None = None,
) -> dict[str, float]:
    """Wait for EnvGroup while surfacing RolloutGroup failures immediately."""

    start = time.monotonic()
    while not all(result.done() for result in env_results):
        if progress is not None:
            progress.report()
        ready = rollout_result.ready()
        if ready:
            values = rollout_result.wait_refs(ready)
            raise RuntimeError(
                "RolloutGroup.generate completed before EnvGroup.interact; "
                f"ready_result={values!r}"
            )
        if timeout_s > 0 and (time.monotonic() - start) > float(timeout_s):
            snapshot = progress.report(force=True) if progress is not None else None
            progress_suffix = (
                f" Current manual cotrain progress: {snapshot.status}."
                if snapshot is not None and snapshot.status
                else ""
            )
            raise TimeoutError(
                "EnvGroup.interact did not finish before "
                f"manual_cotrain.env_rollout_timeout_s={float(timeout_s):.1f}s; "
                "RolloutGroup.generate is still running or waiting for StopMsg. "
                "Set DVLA_COTRAIN_HANDSHAKE_TRACE=1 before launching to log "
                "EnvGroup/RolloutGroup action handshakes."
                f"{progress_suffix}"
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


def _select_plain_str(cfg: DictConfig, path: str) -> str | None:
    value = OmegaConf.select(cfg, path, default=None)
    if value in (None, ""):
        return None
    return str(_plain(value))


def _share_ray_value(value: Any, *, cluster: Any) -> Any:
    """Put a large broadcast payload in Ray's object store exactly once."""

    if ray.is_initialized() and callable(getattr(cluster, "find_free_port", None)):
        return ray.put(value)
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
    missing_required = [
        name for name in missing if not str(name).endswith("_optimizer")
    ]
    if missing_required:
        raise RuntimeError(
            f"{path} missing state_dicts for requested component(s): "
            f"{missing_required}"
        )
    loaded = {name: state_dicts[name] for name in names if name in state_dicts}
    if (
        isinstance(payload, dict)
        and "classifier_threshold" in payload
        and (components is None or "classifier" in names)
    ):
        loaded["classifier_threshold"] = float(payload["classifier_threshold"])
    return loaded


def _nonnegative_int(value: Any, field: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative, got {parsed}")
    return parsed


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
