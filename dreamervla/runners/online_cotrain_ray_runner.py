"""Opt-in Ray online cotrain runner.

This first runner wires the new scheduler/workers into a lightweight synthetic
online loop that exercises the same production boundaries: env rollout,
batched inference, replay insertion, learner PPO-style update, and policy
weight sync. Real LIBERO/VLA construction can plug into these boundaries
without changing the scheduler primitives.
"""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.constants import CHECKPOINT_FORMAT_VERSION
from dreamervla.dataset.collection_manifest import (
    complete_episode_ids_per_task,
    load_online_rollout_episodes,
    next_shard_index,
    online_rollout_episode_counts,
)
from dreamervla.preprocess.sidecar_schema import validate_input_token_preprocess_config
from dreamervla.runners.base_runner import (
    BaseRunner,
    _atomic_torch_save,
    _materialize_checkpoint_copy,
)
from dreamervla.runners.render_device_config import (
    parse_device_ids,
    validate_render_device_pool,
)
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import (
    ComponentPlacement,
    FlexiblePlacementStrategy,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Placement,
    PlacementStrategy,
    ResourceMapPlacementStrategy,
)
from dreamervla.scheduler.worker_group import WorkerGroup
from dreamervla.utils.resource_metrics import collect_resource_metrics
from dreamervla.workers.actor.learner_worker import LearnerWorker
from dreamervla.workers.env.env_worker import EnvWorker
from dreamervla.workers.inference.rollout_inference_worker import RolloutInferenceWorker
from dreamervla.workers.replay.replay_worker import ReplayWorker
from dreamervla.workers.rollout.dump_worker import RolloutDumpWorker


class OnlineCotrainRayRunner(BaseRunner):
    """Small Ray cotrain runner used to validate end-to-end worker overlap."""

    runner_name = "online_cotrain_ray"
    runner_status = "current"
    runner_family = "actor"

    def __init__(self, cfg: dict[str, Any] | DictConfig) -> None:
        config = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
        super().__init__(config)
        self.history: dict[str, float | int] | None = None

    def setup(self) -> None:
        """Hydra train entry lifecycle hook."""
        super().setup()

    def execute(self) -> dict[str, float | int]:
        """Hydra train entry lifecycle hook."""

        self.history = self.run()
        return self.history

    def teardown(self) -> None:
        """Hydra train entry lifecycle hook."""
        super().teardown()

    def run(self) -> dict[str, float | int]:
        cluster_cfg = OmegaConf.select(
            self.cfg,
            "cluster",
            default=OmegaConf.select(self.cfg, "scheduler.cluster", default=None),
        )
        cluster = Cluster(cluster_cfg)
        try:
            cluster.require_single_node()
            groups = self._build_components(cluster)
            metrics = self._run_loop(groups)
            metrics["env/num_logical_envs"] = int(groups["num_envs"])
            metrics["env/num_env_workers"] = int(
                groups.get("num_env_workers", groups["num_envs"])
            )
            metrics["env/envs_per_worker"] = int(groups.get("envs_per_worker", 1))
            # Expose ALL metrics in the log (stdout), independent of logger backend, so an
            # async ray run's learner_updates / overlap_events / rollout episodes +
            # success_rate / losses / timings are always visible in the captured log.
            dump = " ".join(f"{k}={metrics[k]}" for k in sorted(metrics))
            print(f"[ray-cotrain] FINAL METRICS: {dump}", flush=True)
            return metrics
        finally:
            cluster.shutdown()

    def _build_components(self, cluster: Cluster) -> dict[str, Any]:
        rollout_mode = self._ray_rollout_mode()
        if rollout_mode == "learned_actor":
            raise ValueError(
                "ray_rollout.mode=learned_actor requires a learned-actor inference worker; "
                "use no-Ray OnlineCotrainRunner or select ray_rollout.mode=oft_fixed_base."
            )
        num_env_workers = self._env_worker_count()
        num_envs = self._logical_env_count()
        horizon = self._int_from(("env.cfg.kwargs.horizon", "episode_horizon"), 3)
        seq_len = self._int_from(("replay.cfg.sequence_length", "sequence_length"), 3)
        store_name = str(
            self._select_first(
                (
                    "learner.train_cfg.syncer.store_name",
                    "sync.store_name",
                    "weight_store_name",
                ),
                "ray_cotrain_runner_weight_store",
            )
        )

        replay_cfg = self._cfg_from(
            "replay.cfg",
            {
                "capacity": int(self.cfg.get("replay_capacity", 100)),
                "sequence_length": seq_len,
                "task_ids": (0,),
                "rank": 0,
            },
        )
        replay_cfg.setdefault("sequence_length", seq_len)
        replay_group = WorkerGroup(ReplayWorker, replay_cfg).launch(
            cluster, NodePlacementStrategy(1)
        )
        replay = replay_group.workers[0]

        policy_cfg = self._cfg_from("policy.cfg", _default_policy_cfg())
        # Decouple the rollout from the concrete model: the inference worker class is
        # config-selectable. Default = VLA encoder->WM->actor InferenceWorker; OFT
        # selects RolloutInferenceWorker (model-agnostic OFTRolloutBundle producing
        # action + obs_embedding), the same path the collect runner uses.
        infer_worker_cls = _resolve_worker_cls(
            str(
                self._select_first(
                    ("inference.worker_target", "inference.worker"),
                    "dreamervla.workers.inference.inference_worker:InferenceWorker",
                )
            )
        )
        oft_plan = self._oft_worker_plan() if infer_worker_cls is RolloutInferenceWorker else None

        env_cfg = self._rollout_env_cfg(
            oft_plan=oft_plan,
            horizon=horizon,
            use_oft_collect_path=infer_worker_cls is RolloutInferenceWorker,
        )
        dump_group, dump, record_builder = self._build_rollout_dump_group(
            cluster,
            env_cfg,
            oft_plan=oft_plan,
        )
        env_placement = self._env_placement()
        self._validate_env_placement(cluster, env_placement)
        env_group = WorkerGroup(
            EnvWorker,
            env_cfg,
            task_id=0,
            replay=replay,
            dump=dump,
            record_builder=record_builder,
        ).launch(
            cluster,
            env_placement,
            env_vars=self._env_worker_env_vars(),
        )
        # Spread env workers across the configured task ids (round-robin) instead of
        # pinning every env to task 0. The env config itself now comes from the collect
        # plan for OFT, so the external set_task calls mirror collect scheduling rather
        # than compensating for hand-authored env defaults.
        initial_assignments, _task_episode_counts = self._rollout_initial_task_assignments(
            num_envs
        )
        for env_index, tid, start_episode_id in initial_assignments:
            if int(tid) != 0 or int(start_episode_id) != 0:
                self._env_set_task(
                    env_group,
                    env_id=env_index,
                    task_id=tid,
                    start_episode_id=start_episode_id,
                ).wait()

        if infer_worker_cls is RolloutInferenceWorker:
            # OFT recipe: build the OFT rollout inference cfg via the SAME programmatic
            # derivation the collect path uses (OFTRolloutBundle -> action + obs_embedding),
            # not the VLA encoder->WM->actor _default_inference_cfg. DRY: no hand-authored
            # OFT field YAML. init_ckpt stays empty — the OFT base policy loads from the
            # bundle's model_path; the learned actor trains only in imagination.
            infer_cfg = dict((oft_plan or self._oft_worker_plan())["inference"])
            infer_init_ckpt: dict[str, Any] = {}
        else:
            infer_cfg = self._cfg_from("inference.cfg", _default_inference_cfg(policy_cfg))
            infer_cfg.setdefault("policy", policy_cfg)
            infer_cfg.setdefault("device", "cpu")
            infer_init_ckpt = self._load_init_ckpt("inference.init_ckpt")
        inference_placement = self._inference_placement()
        self._validate_single_worker_placement(
            "inference",
            cluster,
            inference_placement,
        )
        infer_group = WorkerGroup(
            infer_worker_cls,
            infer_cfg,
            infer_init_ckpt,
            num_envs=num_envs,
        ).launch(cluster, inference_placement)

        learner_model_cfg = self._cfg_from("learner.model_cfg", {"policy": policy_cfg})
        learner_model_cfg.setdefault("policy", policy_cfg)
        learner_init_ckpt = self._load_init_ckpt("learner.init_ckpt")
        learner_placement = self._learner_placement()
        learner_placements = self._validate_single_worker_placement(
            "learner",
            cluster,
            learner_placement,
        )
        learner_train_cfg = self._learner_train_cfg(
            store_name,
            placement_has_gpu=any(item.visible_accelerators for item in learner_placements),
        )

        learner_group = WorkerGroup(
            LearnerWorker,
            learner_model_cfg,
            learner_init_ckpt,
            learner_train_cfg,
            replay,
        ).launch(cluster, learner_placement)
        groups = {
            "replay": replay_group,
            "envs": env_group,
            "infer": infer_group,
            "learner": learner_group,
            "store_name": store_name,
            "num_envs": num_envs,
            "num_env_workers": num_env_workers,
            "envs_per_worker": self._effective_envs_per_worker(),
        }
        if dump_group is not None:
            groups["dump"] = dump_group
        self._restore_ray_resume_state(groups)
        return groups

    def _run_loop(self, groups: dict[str, Any]) -> dict[str, float | int]:
        self.console_banner("ONLINE COTRAIN (ray)", subtitle=f"envs={groups['num_envs']}")
        try:
            if not _uses_ray_worker_groups(groups):
                result = self._run_loop_sync(groups)
            else:
                result = self._run_loop_overlap(groups)
            self._maybe_save_ray_checkpoint(
                groups,
                env_steps=int(result.get("rollout/steps", 0)),
                learner_updates=int(result.get("train/learner_updates", 0)),
                policy_version=int(result.get("sync/policy_version", 0)),
                metrics=result,
                force=True,
            )
            self.console_banner("ONLINE COTRAIN (ray)", done=True)
            return result
        finally:
            self._close_rollout_dump(groups)

    def _run_loop_sync(self, groups: dict[str, Any]) -> dict[str, float | int]:
        envs = groups["envs"]
        infer = groups["infer"]
        replay = groups["replay"]
        learner = groups["learner"]
        env_ids = list(range(int(groups["num_envs"])))
        target_env_steps = self._int_from(("rollout.steps", "rollout_steps"), 9)
        target_global_steps = self._target_global_steps()
        min_episodes = self._int_from(
            ("rollout.min_replay_episodes", "min_replay_episodes"), 1
        )
        replay_ready_kwargs = self._replay_ready_kwargs(min_episodes)
        weight_sync_every = self._int_from(
            ("sync.weight_sync_every", "weight_sync_every"), 1
        )
        learner_phase = self._learner_update_phase()

        resume_state = self._ray_loop_resume_state()
        learner_updates = int(
            resume_state.get("global_step", getattr(self, "global_step", 0) or 0)
        )
        self.global_step = learner_updates
        start_global_step = int(learner_updates)
        train_start_t = time.perf_counter()
        policy_version = int(learner_updates)
        local_infer_version = int(policy_version)
        self._policy_version = int(policy_version)
        self._wm_version = 0
        self._classifier_version = 0
        self._pending_component_updates = set()
        self._begin_rollout_round()
        local_versions = {"policy": int(local_infer_version)}
        sync_world_model_env = self._world_model_env_sync_enabled()
        last_loss = 0.0
        last_metrics: dict[str, float] = {}
        infer_stage_timing: dict[str, float] = {}
        infer_wait_s = 0.0
        env_step_wait_s = 0.0
        learner_wait_s = 0.0
        weight_sync_wait_s = 0.0
        overlap_events = 0
        pending_learn = None
        pending_learn_start = 0.0
        env_steps = int(resume_state.get("env_step", 0))
        infer_batches = 0
        episode_count = int(resume_state.get("episode_count", 0))
        episode_successes = int(resume_state.get("episode_successes", 0))
        last_episode_success = int(resume_state.get("last_episode_success", 0))
        task_state = self._make_rollout_task_state(len(env_ids))
        active_task_by_env = task_state["active_task_by_env"]
        episode_steps_by_env: dict[int, int] = {}

        while self._cotrain_should_continue(
            learner_updates,
            target_global_steps,
            env_steps,
            target_env_steps,
        ):
            self._console_cotrain_progress(
                learner_updates,
                target_global_steps,
                env_steps,
                target_env_steps,
                phase=("overlap" if pending_learn is not None else "rollout"),
                episode_count=episode_count,
                episode_successes=episode_successes,
                active_task_by_env=active_task_by_env,
                episode_steps_by_env=episode_steps_by_env,
                last_loss=last_loss,
                last_metrics=last_metrics,
            )
            obs_batch_all = self._flatten_env_obs(envs.current_obs().wait())
            for env_id, obs in enumerate(obs_batch_all):
                if isinstance(obs, dict):
                    episode_steps_by_env[int(env_id)] = int(obs.get("step", 0) or 0)
                    if "task_id" in obs:
                        active_task_by_env[int(env_id)] = int(obs["task_id"])
            active_env_ids = env_ids[
                : max(0, min(len(env_ids), target_env_steps - env_steps))
            ]
            obs_batch = [obs_batch_all[env_id] for env_id in active_env_ids]

            if (
                pending_learn is None
                and self._cotrain_can_launch_learner(
                    learner_updates, target_global_steps
                )
                and replay.ready(**replay_ready_kwargs).wait()[0]
            ):
                pending_learn = learner.update(learner_phase, 1)
                pending_learn_start = time.perf_counter()

            rollout_start = time.perf_counter()
            infer_start = time.perf_counter()
            infer_out = infer.forward_batch(obs_batch, active_env_ids).wait()[0]
            infer_wait_s += time.perf_counter() - infer_start
            infer_batches += 1
            for key, value in dict(infer_out.get("timing", {})).items():
                metric_key = f"time/infer_{key}"
                infer_stage_timing[metric_key] = infer_stage_timing.get(
                    metric_key, 0.0
                ) + float(value)
            step_results = []
            hidden_batch = list(infer_out.get("obs_embedding", [None] * len(active_env_ids)))
            if len(hidden_batch) < len(active_env_ids):
                hidden_batch.extend([None] * (len(active_env_ids) - len(hidden_batch)))
            lang_batch = list(infer_out.get("lang_emb", [None] * len(active_env_ids)))
            if len(lang_batch) < len(active_env_ids):
                lang_batch.extend([None] * (len(active_env_ids) - len(lang_batch)))
            for rank, action, hidden, lang_emb in zip(
                active_env_ids,
                infer_out["actions"],
                hidden_batch,
                lang_batch,
                strict=True,
            ):
                env_step_start = time.perf_counter()
                step_results.extend(
                    self._env_step(
                        envs,
                        env_id=rank,
                        action=action,
                        hidden=hidden,
                        lang_emb=lang_emb,
                        step_metadata=self._rollout_step_metadata(
                            env_step=env_steps + 1,
                            learner_updates=learner_updates,
                            policy_version=policy_version,
                            episode_count=episode_count,
                            env_id=rank,
                            task_state=task_state,
                        ),
                    ).wait()
                )
                env_step_wait_s += time.perf_counter() - env_step_start
                env_steps += 1
            rollout_end = time.perf_counter()

            done_envs = []
            for env_id, (_obs, done, info) in zip(
                active_env_ids, step_results, strict=True
            ):
                next_obs = _obs
                if done:
                    done_envs.append(env_id)
                    success = bool((info or {}).get("success", False))
                    episode_count += 1
                    episode_successes += int(success)
                    last_episode_success = int(success)
                    self._record_rollout_episode(
                        episode=episode_count,
                        success=success,
                        successes=episode_successes,
                    )
                    next_obs = self._maybe_rotate_rollout_task(
                        envs, int(env_id), next_obs, task_state
                    )
                    episode_steps_by_env[int(env_id)] = int(
                        next_obs.get("step", 0) if isinstance(next_obs, dict) else 0
                    )
                else:
                    episode_steps_by_env[int(env_id)] = int(
                        next_obs.get("step", episode_steps_by_env.get(int(env_id), 0) + 1)
                        if isinstance(next_obs, dict)
                        else episode_steps_by_env.get(int(env_id), 0) + 1
                    )
            if done_envs:
                infer.reset_states(done_envs).wait()

            if pending_learn is not None:
                learn_done = pending_learn.done()
                if learn_done or env_steps >= target_env_steps:
                    learner_wait_start = time.perf_counter()
                    metrics = pending_learn.wait()[0]
                    learner_wait_s += time.perf_counter() - learner_wait_start
                    last_metrics = _float_metrics(metrics)
                    last_loss = _learner_loss(metrics)
                    learner_updates += 1
                    self.global_step = learner_updates
                    update_metrics = {"train/rl_loss": last_loss, **last_metrics}
                    self._console_cotrain_metric_table(
                        global_step=learner_updates,
                        target_global_steps=target_global_steps,
                        start_global_step=start_global_step,
                        train_start_t=train_start_t,
                        env_steps=env_steps,
                        target_env_steps=target_env_steps,
                        infer_batches=infer_batches,
                        episode_count=episode_count,
                        episode_successes=episode_successes,
                        last_episode_success=last_episode_success,
                        metrics=update_metrics,
                        replay_metrics=self._replay_metric_snapshot(
                            replay, replay_ready_kwargs, ready=True
                        ),
                        timing={
                            "time/infer_wait_s": float(infer_wait_s),
                            "time/env_step_wait_s": float(env_step_wait_s),
                            "time/learner_wait_s": float(learner_wait_s),
                            "time/weight_sync_wait_s": float(weight_sync_wait_s),
                            **infer_stage_timing,
                        },
                    )
                    self.log_metrics(update_metrics, step=learner_updates)
                    sync_start = time.perf_counter()
                    self._mark_learner_update_result(metrics)
                    local_versions = self._sync_after_rollout_boundary(
                        groups,
                        local_versions=local_versions,
                        weight_sync_every=weight_sync_every,
                        sync_world_model_env=sync_world_model_env,
                    )
                    policy_version = int(self._policy_version)
                    local_infer_version = int(local_versions.get("policy", local_infer_version))
                    self._set_replay_policy_version(replay, policy_version)
                    self._begin_rollout_round()
                    weight_sync_wait_s += time.perf_counter() - sync_start
                    if (
                        pending_learn_start < rollout_end
                        and rollout_start < time.perf_counter()
                    ):
                        overlap_events += 1
                    self._maybe_save_ray_checkpoint(
                        groups,
                        env_steps=env_steps,
                        learner_updates=learner_updates,
                        policy_version=policy_version,
                        metrics=update_metrics,
                        loop_state=self._ray_loop_state(
                            env_steps=env_steps,
                            learner_updates=learner_updates,
                            policy_version=policy_version,
                            episode_count=episode_count,
                            episode_successes=episode_successes,
                            last_episode_success=last_episode_success,
                            task_episode_counts=task_state.get(
                                "task_episode_counts", {}
                            ),
                        ),
                    )
                    pending_learn = None

        if pending_learn is not None:
            learner_wait_start = time.perf_counter()
            metrics = pending_learn.wait()[0]
            learner_wait_s += time.perf_counter() - learner_wait_start
            last_metrics = _float_metrics(metrics)
            last_loss = _learner_loss(metrics)
            learner_updates += 1
            self.global_step = learner_updates
            update_metrics = {"train/rl_loss": last_loss, **last_metrics}
            self._console_cotrain_metric_table(
                global_step=learner_updates,
                target_global_steps=target_global_steps,
                start_global_step=start_global_step,
                train_start_t=train_start_t,
                env_steps=env_steps,
                target_env_steps=target_env_steps,
                infer_batches=infer_batches,
                episode_count=episode_count,
                episode_successes=episode_successes,
                last_episode_success=last_episode_success,
                metrics=update_metrics,
                replay_metrics=self._replay_metric_snapshot(
                    replay, replay_ready_kwargs, ready=True
                ),
                timing={
                    "time/infer_wait_s": float(infer_wait_s),
                    "time/env_step_wait_s": float(env_step_wait_s),
                    "time/learner_wait_s": float(learner_wait_s),
                    "time/weight_sync_wait_s": float(weight_sync_wait_s),
                    **infer_stage_timing,
                },
            )
            self.log_metrics(update_metrics, step=learner_updates)
            sync_start = time.perf_counter()
            self._mark_learner_update_result(metrics)
            local_versions = self._sync_after_rollout_boundary(
                groups,
                local_versions=local_versions,
                weight_sync_every=weight_sync_every,
                sync_world_model_env=sync_world_model_env,
            )
            policy_version = int(self._policy_version)
            local_infer_version = int(local_versions.get("policy", local_infer_version))
            self._set_replay_policy_version(replay, policy_version)
            self._begin_rollout_round()
            weight_sync_wait_s += time.perf_counter() - sync_start
            self._maybe_save_ray_checkpoint(
                groups,
                env_steps=env_steps,
                learner_updates=learner_updates,
                policy_version=policy_version,
                metrics=update_metrics,
                loop_state=self._ray_loop_state(
                    env_steps=env_steps,
                    learner_updates=learner_updates,
                    policy_version=policy_version,
                    episode_count=episode_count,
                    episode_successes=episode_successes,
                    last_episode_success=last_episode_success,
                    task_episode_counts=task_state.get("task_episode_counts", {}),
                ),
            )

        rollout_success = self._rollout_success_metrics(
            episode_count=episode_count,
            episode_successes=episode_successes,
            last_episode_success=last_episode_success,
        )
        self._console_cotrain_progress(
            learner_updates,
            target_global_steps,
            env_steps,
            target_env_steps,
            phase="complete",
            episode_count=episode_count,
            episode_successes=episode_successes,
            active_task_by_env=active_task_by_env,
            episode_steps_by_env=episode_steps_by_env,
            last_loss=last_loss,
            last_metrics=last_metrics,
        )
        self._ray_loop_state(
            env_steps=env_steps,
            learner_updates=learner_updates,
            policy_version=policy_version,
            episode_count=episode_count,
            episode_successes=episode_successes,
            last_episode_success=last_episode_success,
            task_episode_counts=task_state.get("task_episode_counts", {}),
        )
        return {
            "global_step": int(learner_updates),
            "rollout/steps": int(env_steps),
            "rollout/infer_batches": int(infer_batches),
            "train/learner_updates": int(learner_updates),
            # Compatibility for older smoke tests and dashboards.
            "train/ppo_updates": int(learner_updates),
            "sync/policy_version": int(policy_version),
            "sync/wm_version": int(self._wm_version),
            "sync/classifier_version": int(self._classifier_version),
            "time/overlap_events": int(overlap_events),
            "time/rollout_overlap_events": 0,
            "time/rollout_infer_ready_batches": int(infer_batches),
            "time/rollout_env_ready_batches": int(env_steps),
            "time/infer_wait_s": float(infer_wait_s),
            "time/env_step_wait_s": float(env_step_wait_s),
            "time/learner_wait_s": float(learner_wait_s),
            "time/weight_sync_wait_s": float(weight_sync_wait_s),
            "train/rl_loss": float(last_loss),
            **infer_stage_timing,
            **last_metrics,
            **rollout_success,
            **collect_resource_metrics(prefix="time"),
        }

    def _run_loop_overlap(self, groups: dict[str, Any]) -> dict[str, float | int]:
        import ray

        envs = groups["envs"]
        infer = groups["infer"]
        replay = groups["replay"]
        learner = groups["learner"]
        env_ids = list(range(int(groups["num_envs"])))
        target_env_steps = self._int_from(("rollout.steps", "rollout_steps"), 9)
        target_global_steps = self._target_global_steps()
        min_episodes = self._int_from(
            ("rollout.min_replay_episodes", "min_replay_episodes"), 1
        )
        replay_ready_kwargs = self._replay_ready_kwargs(min_episodes)
        weight_sync_every = self._int_from(
            ("sync.weight_sync_every", "weight_sync_every"), 1
        )
        learner_phase = self._learner_update_phase()

        ready_obs: list[tuple[int, dict[str, Any]]] = []
        pending_infers: dict[Any, tuple[list[int], Any, float]] = {}
        pending_steps: dict[
            Any, tuple[int, Any, float]
        ] = {}
        pending_learn = None
        pending_learn_start = 0.0
        pending_learn_overlapped = False

        resume_state = self._ray_loop_resume_state()
        env_steps = int(resume_state.get("env_step", 0))
        infer_batches = 0
        learner_updates = int(
            resume_state.get("global_step", getattr(self, "global_step", 0) or 0)
        )
        self.global_step = learner_updates
        start_global_step = int(learner_updates)
        train_start_t = time.perf_counter()
        policy_version = int(learner_updates)
        local_infer_version = int(policy_version)
        self._policy_version = int(policy_version)
        self._wm_version = 0
        self._classifier_version = 0
        self._pending_component_updates = set()
        self._begin_rollout_round()
        local_versions = {"policy": int(local_infer_version)}
        sync_world_model_env = self._world_model_env_sync_enabled()
        last_loss = 0.0
        last_metrics: dict[str, float] = {}
        infer_stage_timing: dict[str, float] = {}

        infer_wait_s = 0.0
        env_step_wait_s = 0.0
        learner_wait_s = 0.0
        weight_sync_wait_s = 0.0
        ray_wait_s = 0.0
        overlap_events = 0
        rollout_overlap_events = 0
        rollout_strict_overlap_events = 0
        infer_ready_batches = 0
        env_ready_batches = 0
        episode_count = int(resume_state.get("episode_count", 0))
        episode_successes = int(resume_state.get("episode_successes", 0))
        last_episode_success = int(resume_state.get("last_episode_success", 0))
        task_state = self._make_rollout_task_state(len(env_ids))
        active_task_by_env = task_state["active_task_by_env"]
        episode_steps_by_env: dict[int, int] = {}

        def add_infer_timing(infer_out: dict[str, Any]) -> None:
            for key, value in dict(infer_out.get("timing", {})).items():
                metric_key = f"time/infer_{key}"
                infer_stage_timing[metric_key] = infer_stage_timing.get(
                    metric_key, 0.0
                ) + float(value)

        def in_flight_env_steps() -> int:
            return len(pending_steps) + sum(
                len(batch_env_ids) for batch_env_ids, _result, _start in pending_infers.values()
            )

        def launch_infer() -> None:
            nonlocal infer_batches, rollout_overlap_events, rollout_strict_overlap_events
            if pending_infers or not ready_obs:
                return
            if not self._cotrain_should_continue(
                learner_updates,
                target_global_steps,
                env_steps,
                target_env_steps,
            ):
                return
            remaining = target_env_steps - env_steps - in_flight_env_steps()
            if remaining <= 0:
                return
            batch = list(ready_obs[:remaining])
            del ready_obs[:remaining]
            batch_env_ids = [env_id for env_id, _obs in batch]
            obs_batch = [obs for _env_id, obs in batch]
            if pending_steps or infer_batches > 0:
                rollout_overlap_events += 1
            if pending_steps:
                rollout_strict_overlap_events += 1
            result = infer.forward_batch(obs_batch, batch_env_ids)
            pending_infers[result.refs[0]] = (
                batch_env_ids,
                result,
                time.perf_counter(),
            )
            infer_batches += 1

        def launch_steps(batch_env_ids: list[int], infer_out: dict[str, Any]) -> None:
            hidden_batch = list(infer_out.get("obs_embedding", [None] * len(batch_env_ids)))
            if len(hidden_batch) < len(batch_env_ids):
                hidden_batch.extend([None] * (len(batch_env_ids) - len(hidden_batch)))
            lang_batch = list(infer_out.get("lang_emb", [None] * len(batch_env_ids)))
            if len(lang_batch) < len(batch_env_ids):
                lang_batch.extend([None] * (len(batch_env_ids) - len(lang_batch)))
            for env_id, action, hidden, lang_emb in zip(
                batch_env_ids,
                infer_out["actions"],
                hidden_batch,
                lang_batch,
                strict=True,
            ):
                result = self._env_step(
                    envs,
                    env_id=int(env_id),
                    action=action,
                    hidden=hidden,
                    lang_emb=lang_emb,
                    step_metadata=self._rollout_step_metadata(
                        env_step=env_steps + in_flight_env_steps() + 1,
                        learner_updates=learner_updates,
                        policy_version=policy_version,
                        episode_count=episode_count,
                        env_id=int(env_id),
                        task_state=task_state,
                    ),
                )
                pending_steps[result.refs[0]] = (
                    int(env_id),
                    result,
                    time.perf_counter(),
                )

        def handle_step_result(
            env_id: int,
            step_result: tuple[dict[str, Any], bool, dict[str, Any]],
        ) -> None:
            nonlocal env_steps, episode_count, episode_successes
            nonlocal last_episode_success
            next_obs, done, info = step_result
            env_steps += 1
            if done:
                success = bool((info or {}).get("success", False))
                episode_count += 1
                episode_successes += int(success)
                last_episode_success = int(success)
                self._record_rollout_episode(
                    episode=episode_count,
                    success=success,
                    successes=episode_successes,
                )
                next_obs = self._maybe_rotate_rollout_task(
                    envs, int(env_id), next_obs, task_state
                )
                episode_steps_by_env[int(env_id)] = int(
                    next_obs.get("step", 0) if isinstance(next_obs, dict) else 0
                )
                infer.reset_states([int(env_id)]).wait()
            else:
                episode_steps_by_env[int(env_id)] = int(
                    next_obs.get("step", episode_steps_by_env.get(int(env_id), 0) + 1)
                    if isinstance(next_obs, dict)
                    else episode_steps_by_env.get(int(env_id), 0) + 1
                )
            if (
                self._cotrain_should_continue(
                    learner_updates,
                    target_global_steps,
                    env_steps + in_flight_env_steps(),
                    target_env_steps,
                )
            ):
                ready_obs.append((int(env_id), next_obs))

        def maybe_launch_learner() -> None:
            nonlocal pending_learn, pending_learn_start, pending_learn_overlapped
            if pending_learn is not None:
                return
            if not self._cotrain_can_launch_learner(
                learner_updates, target_global_steps
            ):
                return
            if not bool(replay.ready(**replay_ready_kwargs).wait()[0]):
                return
            pending_learn = learner.update(learner_phase, 1)
            pending_learn_start = time.perf_counter()
            pending_learn_overlapped = bool(
                pending_infers
                or pending_steps
                or ready_obs
                or env_steps + in_flight_env_steps() < target_env_steps
            )

        def finish_learner(*, block: bool) -> None:
            nonlocal pending_learn, pending_learn_start, pending_learn_overlapped
            nonlocal learner_updates, policy_version, local_infer_version, local_versions
            nonlocal last_loss, last_metrics, learner_wait_s, weight_sync_wait_s
            nonlocal overlap_events
            if pending_learn is None:
                return
            if not block and (pending_infers or not pending_learn.done()):
                return

            learner_wait_start = time.perf_counter()
            metrics = pending_learn.wait()[0]
            learner_wait_s += time.perf_counter() - learner_wait_start
            last_metrics = _float_metrics(metrics)
            last_loss = _learner_loss(metrics)
            learner_updates += 1
            self.global_step = learner_updates
            update_metrics = {"train/rl_loss": last_loss, **last_metrics}
            self._console_cotrain_metric_table(
                global_step=learner_updates,
                target_global_steps=target_global_steps,
                start_global_step=start_global_step,
                train_start_t=train_start_t,
                env_steps=env_steps,
                target_env_steps=target_env_steps,
                infer_batches=infer_batches,
                episode_count=episode_count,
                episode_successes=episode_successes,
                last_episode_success=last_episode_success,
                metrics=update_metrics,
                replay_metrics=self._replay_metric_snapshot(
                    replay, replay_ready_kwargs, ready=True
                ),
                timing={
                    "time/infer_wait_s": float(infer_wait_s),
                    "time/env_step_wait_s": float(env_step_wait_s),
                    "time/learner_wait_s": float(learner_wait_s),
                    "time/weight_sync_wait_s": float(weight_sync_wait_s),
                    "time/ray_wait_s": float(ray_wait_s),
                    **infer_stage_timing,
                },
            )
            self.log_metrics(update_metrics, step=learner_updates)

            sync_start = time.perf_counter()
            self._mark_learner_update_result(metrics)
            local_versions = self._sync_after_rollout_boundary(
                groups,
                local_versions=local_versions,
                weight_sync_every=weight_sync_every,
                sync_world_model_env=sync_world_model_env,
            )
            policy_version = int(self._policy_version)
            local_infer_version = int(local_versions.get("policy", local_infer_version))
            self._set_replay_policy_version(replay, policy_version)
            self._begin_rollout_round()
            weight_sync_wait_s += time.perf_counter() - sync_start

            if pending_learn_overlapped and pending_learn_start <= time.perf_counter():
                overlap_events += 1
            self._maybe_save_ray_checkpoint(
                groups,
                env_steps=env_steps,
                learner_updates=learner_updates,
                policy_version=policy_version,
                metrics=update_metrics,
                loop_state=self._ray_loop_state(
                    env_steps=env_steps,
                    learner_updates=learner_updates,
                    policy_version=policy_version,
                    episode_count=episode_count,
                    episode_successes=episode_successes,
                    last_episode_success=last_episode_success,
                    task_episode_counts=task_state.get("task_episode_counts", {}),
                ),
            )
            pending_learn = None
            pending_learn_start = 0.0
            pending_learn_overlapped = False

        def progress_phase() -> str:
            if pending_learn is not None and (
                pending_infers or pending_steps or ready_obs
            ):
                return "overlap"
            if pending_learn is not None:
                return "learn"
            if pending_infers or pending_steps or ready_obs:
                return "rollout"
            return "idle"

        initial_obs = self._flatten_env_obs(envs.current_obs().wait())
        for env_id, obs in zip(env_ids, initial_obs, strict=True):
            if isinstance(obs, dict):
                episode_steps_by_env[int(env_id)] = int(obs.get("step", 0) or 0)
                if "task_id" in obs:
                    active_task_by_env[int(env_id)] = int(obs["task_id"])
        ready_obs.extend(
            (int(env_id), obs)
            for env_id, obs in zip(env_ids, initial_obs, strict=True)
        )
        maybe_launch_learner()
        launch_infer()

        while pending_infers or pending_steps or ready_obs:
            self._console_cotrain_progress(
                learner_updates,
                target_global_steps,
                env_steps,
                target_env_steps,
                phase=progress_phase(),
                episode_count=episode_count,
                episode_successes=episode_successes,
                active_task_by_env=active_task_by_env,
                episode_steps_by_env=episode_steps_by_env,
                last_loss=last_loss,
                last_metrics=last_metrics,
            )
            finish_learner(block=False)
            if not self._cotrain_should_continue(
                learner_updates,
                target_global_steps,
                env_steps + in_flight_env_steps(),
                target_env_steps,
            ):
                ready_obs.clear()
            maybe_launch_learner()
            launch_infer()
            refs = list(pending_infers) + list(pending_steps)
            if not refs:
                if env_steps >= target_env_steps:
                    break
                continue

            wait_start = time.perf_counter()
            ready_refs, remaining_refs = ray.wait(refs, num_returns=1)
            if remaining_refs:
                extra_ready, _ = ray.wait(
                    remaining_refs,
                    num_returns=len(remaining_refs),
                    timeout=0.0,
                )
                ready_refs.extend(extra_ready)
            ray_wait_s += time.perf_counter() - wait_start

            for ref in ready_refs:
                if ref in pending_infers:
                    batch_env_ids, result, start_time = pending_infers.pop(ref)
                    infer_out = result.wait()[0]
                    infer_wait_s += time.perf_counter() - start_time
                    infer_ready_batches += 1
                    add_infer_timing(infer_out)
                    launch_steps(batch_env_ids, infer_out)
                elif ref in pending_steps:
                    env_id, result, start_time = pending_steps.pop(ref)
                    step_result = result.wait()[0]
                    env_step_wait_s += time.perf_counter() - start_time
                    env_ready_batches += 1
                    handle_step_result(env_id, step_result)

            finish_learner(block=False)
            if not self._cotrain_should_continue(
                learner_updates,
                target_global_steps,
                env_steps,
                target_env_steps,
            ):
                ready_obs.clear()

        for _env_id, result, start_time in list(pending_steps.values()):
            result.wait()
            env_step_wait_s += time.perf_counter() - start_time
        pending_steps.clear()
        for _env_ids, result, start_time in list(pending_infers.values()):
            infer_out = result.wait()[0]
            infer_wait_s += time.perf_counter() - start_time
            add_infer_timing(infer_out)
        pending_infers.clear()
        finish_learner(block=True)

        rollout_success = self._rollout_success_metrics(
            episode_count=episode_count,
            episode_successes=episode_successes,
            last_episode_success=last_episode_success,
        )
        self._console_cotrain_progress(
            learner_updates,
            target_global_steps,
            env_steps,
            target_env_steps,
            phase="complete",
            episode_count=episode_count,
            episode_successes=episode_successes,
            active_task_by_env=active_task_by_env,
            episode_steps_by_env=episode_steps_by_env,
            last_loss=last_loss,
            last_metrics=last_metrics,
        )
        self._ray_loop_state(
            env_steps=env_steps,
            learner_updates=learner_updates,
            policy_version=policy_version,
            episode_count=episode_count,
            episode_successes=episode_successes,
            last_episode_success=last_episode_success,
            task_episode_counts=task_state.get("task_episode_counts", {}),
        )
        return {
            "global_step": int(learner_updates),
            "rollout/steps": int(env_steps),
            "rollout/infer_batches": int(infer_batches),
            "train/learner_updates": int(learner_updates),
            # Compatibility for older smoke tests and dashboards.
            "train/ppo_updates": int(learner_updates),
            "sync/policy_version": int(policy_version),
            "sync/wm_version": int(self._wm_version),
            "sync/classifier_version": int(self._classifier_version),
            "time/overlap_events": int(overlap_events),
            "time/rollout_overlap_events": int(rollout_overlap_events),
            "time/rollout_strict_overlap_events": int(rollout_strict_overlap_events),
            "time/rollout_infer_ready_batches": int(infer_ready_batches),
            "time/rollout_env_ready_batches": int(env_ready_batches),
            "time/infer_wait_s": float(infer_wait_s),
            "time/env_step_wait_s": float(env_step_wait_s),
            "time/learner_wait_s": float(learner_wait_s),
            "time/weight_sync_wait_s": float(weight_sync_wait_s),
            "time/ray_wait_s": float(ray_wait_s),
            "train/rl_loss": float(last_loss),
            **infer_stage_timing,
            **last_metrics,
            **rollout_success,
            **collect_resource_metrics(prefix="time"),
        }

    def _begin_rollout_round(self) -> None:
        self._ensure_version_state()
        self._rollout_policy_version = int(self._policy_version)
        self._rollout_wm_version = int(self._wm_version)
        self._rollout_classifier_version = int(self._classifier_version)

    def _record_transition(self, transition: dict[str, Any]) -> dict[str, Any]:
        self._ensure_version_state()
        stamped = dict(transition)
        stamped.setdefault("policy_version", int(self._rollout_policy_version))
        stamped.setdefault("wm_version", int(self._rollout_wm_version))
        stamped.setdefault("classifier_version", int(self._rollout_classifier_version))
        replay = getattr(self, "_fake_replay", None)
        if replay is not None and hasattr(replay, "add_transition"):
            replay.add_transition(stamped)
        return stamped

    def _mark_learner_update_result(self, metrics: dict[str, Any]) -> set[str]:
        self._ensure_version_state()
        pending = self._updated_components_from_metrics(metrics)
        self._pending_component_updates.update(pending)
        return set(pending)

    def _dispatch_rollout_round(
        self,
        *,
        obs_batch: list[dict[str, Any]],
        env_ids: list[int],
        groups: dict[str, Any] | None = None,
    ) -> list[Any]:
        infer = (
            groups["infer"]
            if groups is not None
            else self._fake_policy_worker
        )
        envs = (
            groups["envs"]
            if groups is not None
            else self._fake_env_group
        )
        infer_out = _wait_first(infer.forward_batch(obs_batch, env_ids))
        hidden = list(infer_out.get("obs_embedding", [None] * len(env_ids)))
        if len(hidden) < len(env_ids):
            hidden.extend([None] * (len(env_ids) - len(hidden)))
        lang = list(infer_out.get("lang_emb", [None] * len(env_ids)))
        if len(lang) < len(env_ids):
            lang.extend([None] * (len(env_ids) - len(lang)))
        step_results = []
        for env_id, action, obs_embedding, lang_emb in zip(
            env_ids,
            infer_out["actions"],
            hidden,
            lang,
            strict=True,
        ):
            step_results.extend(
                envs.execute_on(int(env_id)).step(action, obs_embedding, lang_emb).wait()
            )
        return step_results

    def _sync_after_rollout_boundary(
        self,
        groups: dict[str, Any],
        *,
        local_versions: dict[str, int] | None = None,
        weight_sync_every: int = 1,
        sync_world_model_env: bool = True,
    ) -> dict[str, int]:
        self._ensure_version_state()
        local = dict(local_versions or {})
        pending = set(self._pending_component_updates)
        self._pending_component_updates.clear()
        if not pending:
            return local

        learner = groups["learner"]
        envs = groups.get("envs")
        infer = groups.get("infer")
        store_name = str(groups.get("store_name", ""))
        state_dicts: dict[str, Any] | None = None
        for component in ("policy", "world_model", "classifier"):
            if component not in pending:
                continue
            version_attr = self._version_attr(component)
            version = int(getattr(self, version_attr)) + 1
            setattr(self, version_attr, version)
            if component != "policy" and not bool(sync_world_model_env):
                continue
            sync = getattr(learner, "sync_weights", None)
            if sync is not None:
                _wait_all(sync(component, version))
            if component == "policy":
                if infer is not None and version % max(1, int(weight_sync_every)) == 0:
                    pull = getattr(infer, "pull_weights", None)
                    if pull is not None:
                        pulled = _wait_first(
                            pull(store_name, "policy", int(local.get("policy", 0)))
                        )
                        if pulled is not None:
                            local["policy"] = int(pulled)
                continue
            if envs is None:
                continue
            method_name = (
                "load_world_model_state"
                if component == "world_model"
                else "load_classifier_state"
            )
            sync_env = getattr(envs, method_name, None)
            if sync_env is None:
                continue
            if state_dicts is None:
                state_dicts = dict(_wait_first(learner.state_dicts()))
            _wait_all(sync_env(dict(state_dicts.get(component, {})), version))
        return local

    def _world_model_env_sync_enabled(self) -> bool:
        return bool(
            OmegaConf.select(self.cfg, "sync.world_model_env", default=False)
            or OmegaConf.select(self.cfg, "env.world_model_env", default=False)
        )

    @staticmethod
    def _set_replay_policy_version(replay: Any, version: int) -> None:
        setter = getattr(replay, "set_policy_version", None)
        if setter is None:
            return
        _wait_all(setter(int(version)))

    @staticmethod
    def _version_attr(component: str) -> str:
        if component == "policy":
            return "_policy_version"
        if component == "world_model":
            return "_wm_version"
        if component == "classifier":
            return "_classifier_version"
        raise ValueError(f"unknown component {component!r}")

    def _ensure_version_state(self) -> None:
        if not hasattr(self, "_policy_version"):
            self._policy_version = 0
        if not hasattr(self, "_wm_version"):
            self._wm_version = 0
        if not hasattr(self, "_classifier_version"):
            self._classifier_version = 0
        if not hasattr(self, "_rollout_policy_version"):
            self._rollout_policy_version = int(self._policy_version)
        if not hasattr(self, "_rollout_wm_version"):
            self._rollout_wm_version = int(self._wm_version)
        if not hasattr(self, "_rollout_classifier_version"):
            self._rollout_classifier_version = int(self._classifier_version)
        if not hasattr(self, "_pending_component_updates"):
            self._pending_component_updates: set[str] = set()

    @staticmethod
    def _updated_components_from_metrics(metrics: dict[str, Any]) -> set[str]:
        updates: set[str] = set()
        raw = dict(metrics or {})
        for component in ("policy", "world_model", "classifier"):
            value = raw.get(component)
            if isinstance(value, dict) and bool(value.get("updated", False)):
                updates.add(component)
        for key in raw:
            name = str(key)
            if name.startswith("rl/") or name in {"train/rl_loss", "policy_loss"}:
                updates.add("policy")
            elif name.startswith("wm/") or name.startswith("train/world_model/"):
                updates.add("world_model")
            elif name.startswith("cls/") or name.startswith("train/classifier/"):
                updates.add("classifier")
        return updates

    def _record_rollout_episode(
        self, *, episode: int, success: bool, successes: int
    ) -> None:
        self.console_record_success(success)
        avg_success_rate = float(successes) / max(1, int(episode))
        self.console_rollout_episode(
            episode=int(episode),
            success=bool(success),
            avg_success_rate=avg_success_rate,
            window_success_rate=float(self.console_success_rate()),
        )

    def _rollout_success_metrics(
        self,
        *,
        episode_count: int,
        episode_successes: int,
        last_episode_success: int,
    ) -> dict[str, float]:
        del last_episode_success
        success_rate = (
            float(episode_successes) / float(episode_count)
            if int(episode_count) > 0
            else 0.0
        )
        recent_success_rate = (
            float(self.console_success_rate()) if int(episode_count) > 0 else 0.0
        )
        return {
            "rollout/episodes": float(episode_count),
            "rollout/successes": float(episode_successes),
            "rollout/success_rate": float(success_rate),
            "rollout/success_rate_valid": float(int(episode_count) > 0),
            "rollout/recent_success_rate": float(recent_success_rate),
            "rollout/recent_success_rate_valid": float(int(episode_count) > 0),
        }

    def _console_cotrain_metric_table(
        self,
        *,
        global_step: int,
        target_global_steps: int | None,
        start_global_step: int,
        train_start_t: float,
        env_steps: int,
        target_env_steps: int,
        infer_batches: int,
        episode_count: int,
        episode_successes: int,
        last_episode_success: int,
        metrics: dict[str, Any],
        replay_metrics: dict[str, Any] | None = None,
        timing: dict[str, Any] | None = None,
    ) -> None:
        """Print an RLinf-style metric table for one learner update."""
        core_timing_keys = {
            "time/infer_wait_s",
            "time/env_step_wait_s",
            "time/learner_wait_s",
            "time/weight_sync_wait_s",
            "time/ray_wait_s",
        }
        core_metric_map = {
            "rl/returns_mean": "rollout/returns_mean",
            "rl/actor_loss": "train/actor/actor_loss",
            "rl/critic_loss": "train/critic/critic_loss",
            "rl/value_loss": "train/critic/value_loss",
            "cls/loss": "train/classifier/loss",
            "cls/acc": "train/classifier/acc",
            "wm/loss": "train/world_model/loss",
            "train/rl_loss": "train/rl_loss",
            "train/actor/actor_loss": "train/actor/actor_loss",
            "train/critic/critic_loss": "train/critic/critic_loss",
            "train/classifier/loss": "train/classifier/loss",
            "train/classifier/acc": "train/classifier/acc",
            "train/world_model/loss": "train/world_model/loss",
        }
        display_metrics: dict[str, Any] = {}
        for key, value in dict(timing or {}).items():
            if key in core_timing_keys:
                display_metrics[key] = value
        display_metrics.update(
            {
                "rollout/env_steps": float(env_steps),
                "train/learner_updates": float(global_step),
            }
        )
        rollout_success = self._rollout_success_metrics(
            episode_count=episode_count,
            episode_successes=episode_successes,
            last_episode_success=last_episode_success,
        )
        for key in ("rollout/episodes", "rollout/success_rate", "rollout/recent_success_rate"):
            display_metrics[key] = rollout_success[key]
        display_metrics.update(replay_metrics or {})
        if int(episode_count) > 0:
            display_metrics["env/success_once"] = float(episode_successes) / max(
                1, int(episode_count)
            )
            display_metrics["env/num_trajectories"] = float(episode_count)
            display_metrics["env/last_success"] = float(last_episode_success)

        for key, value in dict(metrics).items():
            if isinstance(value, str):
                continue
            out_key = core_metric_map.get(str(key))
            if out_key is not None:
                display_metrics[out_key] = value

        total_steps = (
            int(target_global_steps)
            if target_global_steps is not None
            else max(int(global_step), 1)
        )
        self.console_metric_table(
            step=max(0, int(global_step) - 1),
            total_steps=total_steps,
            elapsed_s=time.perf_counter() - float(train_start_t),
            metrics=display_metrics,
            start_step=int(start_global_step),
        )

    def _replay_metric_snapshot(
        self,
        replay: Any,
        replay_ready_kwargs: dict[str, Any],
        *,
        ready: bool,
    ) -> dict[str, Any]:
        """Return compact replay-buffer fields for the RLinf-style table."""
        metrics: dict[str, Any] = {
            "replay_buffer/ready": float(bool(ready)),
            "replay_buffer/min_episodes": float(
                replay_ready_kwargs.get("min_episodes_per_task", 0)
            ),
            "replay_buffer/min_transitions": float(
                replay_ready_kwargs.get("min_transitions", 0)
            ),
        }
        try:
            metrics["replay_buffer/size"] = float(replay.size().wait()[0])
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
        try:
            metrics["replay_buffer/transitions"] = float(
                replay.num_transitions().wait()[0]
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
        return metrics

    def _checkpoint_cfg(self) -> dict[str, Any]:
        raw = OmegaConf.select(self.cfg, "checkpoint", default={}) or {}
        cfg = _plain(raw)
        if not isinstance(cfg, dict):
            raise TypeError("checkpoint must be a mapping")
        if "save_interval" not in cfg:
            compat_interval = cfg.get(
                "every_updates",
                OmegaConf.select(self.cfg, "training.checkpoint_every", default=0),
            )
            cfg["save_interval"] = compat_interval
        cfg.setdefault("save_final", False)
        cfg.setdefault("filename", "learner.ckpt")
        cfg.setdefault("latest_name", "latest.ckpt")
        return cfg

    def _build_rollout_dump_group(
        self,
        cluster: Cluster,
        env_cfg: dict[str, Any],
        *,
        oft_plan: dict[str, Any] | None,
    ) -> tuple[WorkerGroup | None, Any | None, Any | None]:
        dump_cfg = self._rollout_dump_cfg(oft_plan=oft_plan)
        if not dump_cfg["enabled"]:
            return None, None, None

        demos_per_shard = int(dump_cfg["demos_per_shard"])
        if demos_per_shard != 1:
            raise ValueError(
                "rollout.dump.demos_per_shard must be 1 for cotrain; "
                "cotrain persists one collected episode per HDF5 file."
            )

        env_cfg.setdefault("kwargs", {})
        env_cfg["kwargs"]["full_record"] = True
        shard_prefix = str(dump_cfg["shard_prefix"])
        reward_dir = str(dump_cfg["reward_dir"])
        hidden_dir = str(dump_cfg["hidden_dir"])
        manifest_root = str(dump_cfg["manifest_root"])
        self._rollout_episode_resume_counts_value = (
            self._rollout_episode_resume_counts_from_sources(
                reward_dir,
                hidden_dir,
                manifest_root=manifest_root,
                task_ids=tuple(self._rollout_task_ids() or [0]),
            )
        )
        self._online_rollout_manifest_root = manifest_root
        start_shard_index = next_shard_index(reward_dir, prefix=shard_prefix)
        dump_group = WorkerGroup(
            RolloutDumpWorker,
            reward_dir,
            hidden_dir,
            f"{shard_prefix}_{start_shard_index:03d}.hdf5",
            dict(dump_cfg["preprocess_config"]),
            dict(dump_cfg["data_attrs"]),
            demos_per_shard,
            start_shard_index,
            manifest_root,
            int(dump_cfg["keep_last_global_steps"]),
        ).launch(cluster, NodePlacementStrategy(1))
        return dump_group, dump_group.workers[0], _build_cotrain_dump_step

    def _rollout_dump_cfg(self, *, oft_plan: dict[str, Any] | None) -> dict[str, Any]:
        raw = OmegaConf.select(self.cfg, "rollout.dump", default={}) or {}
        cfg = _plain(raw)
        if not isinstance(cfg, dict):
            raise TypeError("rollout.dump must be a mapping")
        enabled = bool(cfg.get("enabled", False))
        run_dir = self.get_run_dir()
        plan_dump = dict((oft_plan or {}).get("dump", {}) or {})
        reward_dir = str(cfg.get("reward_dir", run_dir / "rollouts" / "reward"))
        hidden_dir = str(cfg.get("hidden_dir", run_dir / "rollouts" / "hidden"))
        manifest_root = cfg.get("manifest_root")
        if manifest_root in (None, ""):
            manifest_root = str(Path(reward_dir).expanduser().parent)
        preprocess_config = dict(
            cfg.get(
                "preprocess_config",
                plan_dump.get("preprocess_config", _default_preprocess_config()),
            )
            or {}
        )
        if enabled:
            validate_input_token_preprocess_config(
                preprocess_config,
                context="rollout.dump.preprocess_config",
            )
        return {
            "enabled": enabled,
            "reward_dir": reward_dir,
            "hidden_dir": hidden_dir,
            "manifest_root": str(manifest_root),
            "keep_last_global_steps": int(cfg.get("keep_last_global_steps", 0)),
            "shard_prefix": str(cfg.get("shard_prefix", "cotrain_episode")),
            "demos_per_shard": int(cfg.get("demos_per_shard", 1)),
            "preprocess_config": preprocess_config,
            "data_attrs": dict(
                cfg.get(
                    "data_attrs",
                    plan_dump.get(
                        "data_attrs",
                        {
                            "task_suite_name": str(
                                self._select_first(("task.suite", "task.name"), "unknown")
                            ),
                            "env_name": "libero",
                        },
                    ),
                )
                or {}
            ),
        }

    @staticmethod
    def _close_rollout_dump(groups: dict[str, Any]) -> None:
        dump = groups.get("dump")
        if dump is None:
            return
        close = getattr(dump, "close", None)
        if close is None:
            return
        _wait_all(close())

    def _restore_ray_resume_state(self, groups: dict[str, Any]) -> dict[str, Any]:
        payload = self._ray_resume_payload()
        ray_state = _ray_state_from_payload(payload) if isinstance(payload, Mapping) else None
        if not isinstance(ray_state, Mapping):
            self._ray_loop_resume_state_value = {}
            return {}

        replay = groups.get("replay")
        state = self._normalise_ray_loop_state(ray_state)
        self.global_step = int(state["global_step"])
        self._policy_version = int(state["global_step"])
        self._wm_version = 0
        self._classifier_version = 0
        self._set_replay_policy_version(replay, int(state["global_step"]))
        replay_episodes = self._load_replay_from_online_rollout_history(replay)
        self._ray_loop_resume_state_value = dict(state)
        print(
            f"[ray-cotrain] restored ray resume state "
            f"global_step={state['global_step']} env_step={state['env_step']} "
            f"replay_episodes={replay_episodes}",
            flush=True,
        )
        return state

    def _ray_loop_resume_state(self) -> dict[str, Any]:
        state = getattr(self, "_ray_loop_resume_state_value", None)
        return dict(state or {})

    def _ray_resume_payload(self) -> dict[str, Any]:
        cached = getattr(self, "_ray_resume_payload_cache", None)
        if cached is not None:
            return dict(cached)
        path = self._ray_resume_checkpoint_path()
        if path is None:
            self._ray_resume_payload_cache = {}
            return {}
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            self._ray_resume_payload_cache = {}
            return {}
        self._ray_resume_payload_cache = dict(payload)
        return dict(payload)

    def _ray_resume_checkpoint_path(self) -> Path | None:
        candidates: list[Path] = []
        resume_path = OmegaConf.select(self.cfg, "training.resume_path", default=None)
        if resume_path not in (None, "", "auto"):
            candidates.extend(
                self._resume_path_candidates(Path(str(resume_path)).expanduser())
            )

        resume_dir = OmegaConf.select(self.cfg, "training.resume_dir", default=None)
        if resume_dir not in (None, "", "auto"):
            candidates.extend(
                self._resume_path_candidates(Path(str(resume_dir)).expanduser())
            )

        if bool(OmegaConf.select(self.cfg, "training.resume", default=False)):
            candidates.extend(
                [
                    self.get_checkpoint_dir() / "latest.ckpt",
                    self.get_compat_checkpoint_dir() / "latest.ckpt",
                    self.get_run_dir() / "latest.ckpt",
                ]
            )

        init_cfg = OmegaConf.select(self.cfg, "learner.init_ckpt", default=None)
        init_plain = _plain(init_cfg) if init_cfg is not None else None
        if isinstance(init_plain, str) and init_plain:
            candidates.extend(self._resume_path_candidates(Path(init_plain).expanduser()))
        elif isinstance(init_plain, dict):
            init_path = init_plain.get("path")
            if init_path not in (None, ""):
                candidates.extend(
                    self._resume_path_candidates(Path(str(init_path)).expanduser())
                )

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _resume_path_candidates(path: Path) -> list[Path]:
        if path.is_dir():
            return [
                path / "checkpoints" / "latest.ckpt",
                path / "ckpt" / "latest.ckpt",
                path / "latest.ckpt",
            ]
        return [path]

    @staticmethod
    def _load_replay_state(replay: Any, state: Mapping[str, Any]) -> None:
        handle = replay.execute_on(0) if hasattr(replay, "execute_on") else replay
        loader = getattr(handle, "load_state_dict", None)
        if loader is None:
            raise RuntimeError("Ray replay worker does not support load_state_dict")
        _wait_all(loader(dict(state)))

    def _load_replay_from_online_rollout_history(self, replay: Any | None) -> int:
        if replay is None:
            return 0
        root = getattr(self, "_online_rollout_manifest_root", None)
        if root in (None, ""):
            return 0
        episodes = load_online_rollout_episodes(str(root))
        if not episodes:
            return 0
        target = replay.execute_on(0) if hasattr(replay, "execute_on") else replay
        add_episode = getattr(target, "add_episode", None)
        if add_episode is None:
            return 0
        loaded = 0
        for episode in episodes:
            _wait_all(add_episode(list(episode), source="online"))
            loaded += 1
        return int(loaded)

    @staticmethod
    def _normalise_ray_loop_state(raw: Mapping[str, Any]) -> dict[str, Any]:
        global_step = int(raw.get("global_step", raw.get("update_step", 0)))
        env_steps = int(raw.get("env_steps", raw.get("env_step", 0)))
        return {
            "global_step": global_step,
            "env_step": env_steps,
        }

    def _ray_loop_state(
        self,
        *,
        env_steps: int,
        learner_updates: int,
        policy_version: int,
        episode_count: int,
        episode_successes: int,
        last_episode_success: int,
        task_episode_counts: Mapping[int, int] | None = None,
    ) -> dict[str, Any]:
        del policy_version, episode_count, episode_successes, last_episode_success
        del task_episode_counts
        state = {
            "global_step": int(learner_updates),
            "env_step": int(env_steps),
        }
        self._ray_loop_runtime_state = dict(state)
        return state

    def _maybe_save_ray_checkpoint(
        self,
        groups: dict[str, Any],
        *,
        env_steps: int,
        learner_updates: int,
        policy_version: int,
        metrics: dict[str, Any],
        loop_state: dict[str, Any] | None = None,
        force: bool = False,
    ) -> Path | None:
        cfg = self._checkpoint_cfg()
        global_step = int(learner_updates)
        if global_step <= 0:
            return None
        save_interval = int(cfg.get("save_interval", 0) or 0)
        if force:
            if not bool(cfg.get("save_final", False)):
                return None
        elif (
            save_interval <= 0
            or global_step % save_interval != 0
        ):
            return None
        return self._save_ray_checkpoint(
            groups["learner"],
            replay=groups.get("replay"),
            env_steps=env_steps,
            learner_updates=global_step,
            policy_version=policy_version,
            metrics=metrics,
            loop_state=(
                loop_state
                if loop_state is not None
                else getattr(self, "_ray_loop_runtime_state", None)
            ),
            checkpoint_cfg=cfg,
        )

    def _save_ray_checkpoint(
        self,
        learner: Any,
        *,
        replay: Any | None = None,
        env_steps: int,
        learner_updates: int,
        policy_version: int,
        metrics: dict[str, Any],
        loop_state: dict[str, Any] | None = None,
        checkpoint_cfg: dict[str, Any] | None = None,
    ) -> Path:
        cfg = checkpoint_cfg if checkpoint_cfg is not None else self._checkpoint_cfg()
        global_step = int(learner_updates)
        path = self._ray_checkpoint_path(
            cfg,
            global_step=global_step,
            env_steps=int(env_steps),
            policy_version=int(policy_version),
        )
        metrics_float = _float_metrics(metrics)
        ray_state = self._checkpoint_ray_loop_state(
            env_steps=int(env_steps),
            learner_updates=global_step,
            policy_version=int(policy_version),
            metrics=metrics_float,
            loop_state=loop_state,
        )
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "global_step": global_step,
            "cfg": self.cfg,
            "state_dicts": self._ray_learner_state_dicts(learner),
            "pickles": {},
            "ray": ray_state,
            "metrics": metrics_float,
        }
        del replay
        _atomic_torch_save(payload, path)
        latest_name = str(cfg.get("latest_name", "latest.ckpt") or "")
        latest_path = None
        if latest_name:
            latest_path = Path(latest_name).expanduser()
            if not latest_path.is_absolute():
                latest_path = self._ray_checkpoint_dir(cfg) / latest_path
            _materialize_checkpoint_copy(path, latest_path)
        latest_text = f" latest={latest_path}" if latest_path is not None else ""
        print(f"[ray-cotrain] checkpoint saved path={path}{latest_text}", flush=True)
        return path

    def _checkpoint_ray_loop_state(
        self,
        *,
        env_steps: int,
        learner_updates: int,
        policy_version: int,
        metrics: dict[str, float],
        loop_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        state = dict(loop_state or {})
        global_step = int(state.get("global_step", learner_updates))
        env_steps_value = int(state.get("env_steps", state.get("env_step", env_steps)))
        del policy_version, metrics
        return {
            "global_step": global_step,
            "env_step": env_steps_value,
        }

    @staticmethod
    def _ray_replay_state_dict(replay: Any | None) -> dict[str, Any] | None:
        if replay is None:
            return None
        handle = replay.execute_on(0) if hasattr(replay, "execute_on") else replay
        state_dict = getattr(handle, "state_dict", None)
        if state_dict is None:
            return None
        state = _wait_first(state_dict())
        if state is None:
            return None
        if not isinstance(state, dict):
            raise TypeError("Ray replay state_dict() must return a mapping")
        return dict(state)

    def _ray_checkpoint_path(
        self,
        checkpoint_cfg: dict[str, Any],
        *,
        global_step: int,
        env_steps: int,
        policy_version: int,
    ) -> Path:
        ckpt_dir = self._ray_checkpoint_dir(checkpoint_cfg)
        format_str = checkpoint_cfg.get("format_str")
        if format_str not in (None, ""):
            filename = str(format_str).format(
                global_step=int(global_step),
                env_step=int(env_steps),
            )
            return ckpt_dir / filename
        filename = str(checkpoint_cfg.get("filename", "learner.ckpt") or "learner.ckpt")
        return ckpt_dir / f"global_step_{int(global_step)}" / filename

    def _ray_checkpoint_dir(self, checkpoint_cfg: dict[str, Any]) -> Path:
        raw = checkpoint_cfg.get("dir")
        if raw in (None, ""):
            return self.get_checkpoint_dir()
        path = Path(str(raw)).expanduser()
        if path.is_absolute():
            return path
        return self.get_run_dir() / path

    def _ray_learner_state_dicts(
        self, learner: Any
    ) -> dict[str, dict[str, torch.Tensor]]:
        handle = learner
        if hasattr(learner, "execute_on"):
            handle = learner.execute_on(0)
        result = handle.state_dicts()
        if hasattr(result, "wait"):
            values = result.wait()
            state_dicts = values[0] if isinstance(values, list) else values
        else:
            state_dicts = result
        if not isinstance(state_dicts, dict):
            raise TypeError("Ray learner state_dicts() must return a mapping")
        return {
            str(name): {
                str(key): value.detach().cpu()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in dict(state).items()
            }
            for name, state in state_dicts.items()
        }

    def _learner_update_phase(self) -> str:
        mode = str(
            OmegaConf.select(self.cfg, "learner.train_cfg.mode", default="synthetic_ppo")
        )
        if mode == "dreamervla_cotrain":
            return "cotrain"
        return "rl"

    def _select_first(self, paths: tuple[str, ...], default: Any) -> Any:
        for path in paths:
            value = OmegaConf.select(self.cfg, path, default=None)
            if value is not None:
                return value
        return default

    def _rollout_task_ids(self) -> list[int]:
        """Concrete env task-id list to spread rollout envs across (round-robin).

        Prefers ray_data.task_ids / env.task_ids; ignores non-list values like "all"."""
        for path in ("ray_data.task_ids", "env.task_ids", "env.cfg.kwargs.task_ids"):
            raw = OmegaConf.select(self.cfg, path, default=None)
            if raw is None or isinstance(raw, str):
                continue
            try:
                ids = [int(x) for x in raw]
            except (TypeError, ValueError):
                continue
            if ids:
                return ids
        return []

    def _ray_rollout_mode(self) -> str:
        mode = str(OmegaConf.select(self.cfg, "ray_rollout.mode", default="oft_fixed_base"))
        allowed = {"oft_fixed_base", "learned_actor"}
        if mode not in allowed:
            raise ValueError(
                f"ray_rollout.mode must be one of {sorted(allowed)}, got {mode!r}"
            )
        return mode

    def _replay_ready_kwargs(self, min_episodes: int) -> dict[str, Any]:
        return {
            "min_episodes_per_task": int(min_episodes),
            "min_transitions": self._int_from(
                ("rollout.min_replay_transitions", "min_replay_transitions"), 0
            ),
            "task_ids": tuple(self._rollout_task_ids() or [0]),
            "min_sampleable_windows": self._int_from(
                ("rollout.min_sampleable_windows", "min_sampleable_windows"), 0
            ),
            "require_classifier_evidence": bool(
                self._select_first(
                    (
                        "rollout.require_classifier_evidence",
                        "require_classifier_evidence",
                    ),
                    False,
                )
            ),
        }

    def _make_rollout_task_state(self, num_envs: int) -> dict[str, Any]:
        assignments, task_episode_counts = self._rollout_initial_task_assignments(num_envs)
        task_ids = self._rollout_task_ids()
        active_task_by_env = {
            int(env_id): int(task_id) for env_id, task_id, _episode_id in assignments
        }
        active_episode_id_by_env = {
            int(env_id): int(episode_id)
            for env_id, _task_id, episode_id in assignments
        }
        return {
            "task_ids": task_ids,
            "active_task_by_env": active_task_by_env,
            "active_episode_id_by_env": active_episode_id_by_env,
            "task_episode_counts": task_episode_counts,
            "next_task_index": int(num_envs) if task_ids else 0,
        }

    def _rollout_initial_task_assignments(
        self,
        num_envs: int,
    ) -> tuple[list[tuple[int, int, int]], dict[int, int]]:
        task_ids = self._rollout_task_ids()
        if not task_ids:
            return [], {}
        task_episode_counts = self._rollout_episode_resume_counts(task_ids)
        assignments: list[tuple[int, int, int]] = []
        for env_id in range(max(0, int(num_envs))):
            task_id = int(task_ids[env_id % len(task_ids)])
            start_episode_id = int(task_episode_counts.get(task_id, 0))
            assignments.append((int(env_id), task_id, start_episode_id))
            task_episode_counts[task_id] = start_episode_id + 1
        return assignments, task_episode_counts

    def _rollout_episode_resume_counts(self, task_ids: list[int]) -> dict[int, int]:
        raw = getattr(self, "_rollout_episode_resume_counts_value", {}) or {}
        counts = _normalise_task_episode_counts(raw)
        return {int(task_id): int(counts.get(int(task_id), 0)) for task_id in task_ids}

    def _rollout_episode_resume_counts_from_sources(
        self,
        reward_dir: str,
        hidden_dir: str,
        *,
        manifest_root: str | None = None,
        task_ids: tuple[int, ...],
    ) -> dict[int, int]:
        counts: dict[int, int] = {}
        if manifest_root not in (None, ""):
            counts = online_rollout_episode_counts(
                str(manifest_root),
                task_ids=task_ids,
            )
        fallback = _rollout_episode_resume_counts_from_dump(
            reward_dir,
            hidden_dir,
            task_ids=task_ids,
        )
        for task_id, value in fallback.items():
            counts[int(task_id)] = max(int(counts.get(int(task_id), 0)), int(value))
        return counts

    def _maybe_rotate_rollout_task(
        self,
        envs: Any,
        env_id: int,
        next_obs: dict[str, Any],
        task_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Switch an env to the next configured task after an episode boundary.

        This is deliberately episode-boundary based: low-level env/rollout steps only
        update progress status; they do not drive task switching or train progress.
        """

        task_ids = list(task_state.get("task_ids") or [])
        if not task_ids:
            return next_obs
        active_task_by_env = task_state["active_task_by_env"]
        active_episode_id_by_env = task_state.setdefault("active_episode_id_by_env", {})
        task_episode_counts = task_state["task_episode_counts"]
        env_id = int(env_id)
        next_index = int(task_state.get("next_task_index", 0))
        next_task = int(task_ids[next_index % len(task_ids)])
        task_state["next_task_index"] = next_index + 1
        active_task_by_env[env_id] = next_task
        start_episode_id = int(task_episode_counts.get(next_task, 0))
        task_episode_counts[next_task] = start_episode_id + 1
        active_episode_id_by_env[env_id] = start_episode_id
        switched = self._env_set_task(
            envs,
            env_id=env_id,
            task_id=next_task,
            start_episode_id=start_episode_id,
        ).wait()
        if isinstance(switched, list):
            return dict(switched[0])
        return dict(switched)

    def _cotrain_progress_status(
        self,
        *,
        phase: str,
        global_step: int,
        train_step: str,
        env_steps: int,
        target_env_steps: int,
        episode_count: int,
        episode_successes: int,
        active_task_by_env: dict[int, int] | None = None,
        episode_steps_by_env: dict[int, int] | None = None,
        last_loss: float = 0.0,
        last_metrics: dict[str, float] | None = None,
    ) -> str:
        parts = [
            f"phase={str(phase)}",
            f"global_step={int(global_step)}",
            f"train_step={str(train_step)}",
            f"env_step={int(env_steps)}/{int(target_env_steps)}",
        ]
        active = dict(active_task_by_env or {})
        ep_steps = dict(episode_steps_by_env or {})
        if active:
            pairs = []
            for env_id in sorted(active):
                task_id = int(active[env_id])
                step = int(ep_steps.get(int(env_id), 0))
                pairs.append(f"t{task_id}:s{step}")
            parts.append("rollout_step=" + ",".join(pairs[:4]))
        if int(episode_count) > 0:
            success_rate = float(episode_successes) / max(1, int(episode_count))
            parts.append(f"eps={int(episode_count)}")
            parts.append(f"succ={success_rate:.3f}")
        if float(last_loss) != 0.0:
            parts.append(f"loss={float(last_loss):.3f}")
        metrics = last_metrics or {}
        if "cls/acc" in metrics:
            parts.append(f"cls_acc={float(metrics['cls/acc']):.3f}")
        elif "cls/f1" in metrics:
            parts.append(f"cls_f1={float(metrics['cls/f1']):.3f}")
        if "rl/returns_mean" in metrics:
            parts.append(f"ret={float(metrics['rl/returns_mean']):.3f}")
        return " ".join(parts)

    def _learner_progress_file(self) -> Path | None:
        raw = getattr(self, "_learner_progress_path", None)
        if raw in (None, ""):
            raw = OmegaConf.select(
                self.cfg,
                "learner.train_cfg.progress_path",
                default=None,
            )
        if raw in (None, ""):
            return None
        return Path(str(raw)).expanduser()

    def _read_learner_train_progress(self) -> dict[str, Any]:
        path = self._learner_progress_file()
        if path is None or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _learner_train_step_status(self, phase: str) -> str:
        progress = self._read_learner_train_progress()
        if progress:
            train_step = int(progress.get("train_step", 0) or 0)
            total = int(progress.get("total_train_steps", 0) or 0)
            wm_step = int(progress.get("wm_step", 0) or 0)
            wm_total = int(progress.get("wm_total_steps", 0) or 0)
            cls_step = int(progress.get("cls_step", 0) or 0)
            cls_total = int(progress.get("cls_total_steps", 0) or 0)
            vlarl_step = int(progress.get("vlarl_step", 0) or 0)
            vlarl_total = int(progress.get("vlarl_total_steps", 0) or 0)
            return (
                f"{train_step}/{total} "
                f"wm_step={wm_step}/{wm_total} "
                f"cls_step={cls_step}/{cls_total} "
                f"vlarl_step={vlarl_step}/{vlarl_total}"
            )
        if str(phase) in {"learn", "overlap"}:
            return "0/0 wm_step=0/0 cls_step=0/0 vlarl_step=0/0"
        return "0/0 wm_step=0/0 cls_step=0/0 vlarl_step=0/0"

    def _egl_device_pool(self) -> list[int]:
        """Explicit physical GPU ids reserved for egl env rendering."""
        backend = str(
            self._select_first(("render_backend", "env.render_backend"), "egl")
        ).lower()
        if backend != "egl":
            return []
        num_envs = self._int_from(("env.num_workers", "num_env_workers"), 1)
        render_devices = self._select_first(("render_devices", "env.render_devices"), [])
        return validate_render_device_pool(
            render_backend=backend,
            num_envs=num_envs,
            render_devices=render_devices,
            compute_devices=self._ray_compute_device_pool(),
            render_key="render_devices",
        )

    def _env_placement(self) -> PlacementStrategy:
        """RLinf-style placement for env workers.

        osmesa stays CPU-only. EGL places env workers directly on the configured
        render GPU pool, allowing multiple env workers to share one render GPU.
        The worker-level CUDA/EGL regime is injected by WorkerGroup, and env
        subprocesses inherit it.
        """
        num_envs = self._env_worker_count()
        backend = str(
            self._select_first(("render_backend", "env.render_backend"), "osmesa")
        ).lower()
        if backend != "egl":
            return NodePlacementStrategy(num_envs)
        component_strategy = self._component_placement_strategy("env")
        if component_strategy is not None:
            return component_strategy

        placement_cfg = self._cfg_from("env.placement", {})
        strategy = str(placement_cfg.get("strategy", "")).strip().lower()
        if strategy in {"", "shared", "shared_accelerator", "shared_gpu"}:
            devices = self._egl_device_pool()
            return ResourceMapPlacementStrategy(
                self._shared_resource_rank_map(devices, num_envs)
            )
        if strategy == "packed":
            start_gpu = int(
                placement_cfg.get("gpu_id", placement_cfg.get("start_gpu", 0))
            )
            end_gpu = int(placement_cfg.get("end_gpu", start_gpu))
            num_gpus_per_worker = int(placement_cfg.get("num_gpus_per_worker", 1))
            return PackedPlacementStrategy(
                start_gpu,
                end_gpu,
                num_gpus_per_worker=num_gpus_per_worker,
            )
        if strategy == "flexible":
            groups = placement_cfg.get("accelerator_groups")
            if groups is None:
                groups = placement_cfg.get("groups")
            if groups is None:
                raise ValueError(
                    "env.placement.accelerator_groups is required for flexible placement"
                )
            return FlexiblePlacementStrategy(groups)
        raise ValueError(
            "env.placement.strategy must be one of shared, packed, or flexible; "
            f"got {strategy!r}"
        )

    def _validate_env_placement(
        self,
        cluster: Cluster,
        placement: PlacementStrategy,
    ) -> list[Placement]:
        placements = placement.get_placement(cluster)
        expected = self._env_worker_count()
        actual = len(placements)
        if actual != expected:
            raise ValueError(
                "env placement produced "
                f"{actual} worker(s), but env.num_workers={expected}; set "
                "env.num_workers to match cluster.component_placement.env."
            )
        return placements

    def _validate_single_worker_placement(
        self,
        component: str,
        cluster: Cluster,
        placement: PlacementStrategy,
    ) -> list[Placement]:
        placements = placement.get_placement(cluster)
        if len(placements) != 1:
            raise ValueError(
                f"{component} placement must resolve to a single worker in "
                "OnlineCotrainRayRunner; multi-worker compute placement is not "
                f"supported by this runner yet (got {len(placements)} workers)."
            )
        return placements

    def _component_placement_strategy(
        self, *component_names: str
    ) -> PlacementStrategy | None:
        if OmegaConf.select(self.cfg, "cluster.component_placement", default=None) is None:
            return None
        placement = ComponentPlacement(self.cfg)
        for name in component_names:
            if placement.has_component(name):
                return placement.get_strategy(name)
        return None

    def _env_worker_count(self) -> int:
        return self._int_from(("env.num_workers", "num_env_workers"), 1)

    def _envs_per_worker(self) -> int:
        return self._int_from(("env.envs_per_worker", "envs_per_worker"), 1)

    def _effective_envs_per_worker(self) -> int:
        backend = str(
            self._select_first(("render_backend", "env.render_backend"), "osmesa")
        ).lower()
        if backend != "egl":
            return 1
        return self._envs_per_worker()

    def _logical_env_count(self) -> int:
        explicit = OmegaConf.select(self.cfg, "env.total_num_envs", default=None)
        if explicit is not None:
            return int(explicit)
        return self._env_worker_count() * self._effective_envs_per_worker()

    def _env_worker_rank_slot(self, env_id: int) -> tuple[int, int]:
        envs_per_worker = self._effective_envs_per_worker()
        if envs_per_worker < 1:
            raise ValueError(f"env.envs_per_worker must be >= 1, got {envs_per_worker}")
        env_id = int(env_id)
        return env_id // envs_per_worker, env_id % envs_per_worker

    def _flatten_env_obs(self, worker_obs: list[Any]) -> list[dict[str, Any]]:
        if self._effective_envs_per_worker() == 1:
            return list(worker_obs)
        flattened: list[dict[str, Any]] = []
        for item in worker_obs:
            if not isinstance(item, (list, tuple)):
                raise TypeError(
                    "EnvWorker.current_obs() must return a list when env.envs_per_worker>1"
                )
            flattened.extend(dict(obs) for obs in item)
        return flattened

    def _env_step(
        self,
        envs: Any,
        *,
        env_id: int,
        action: Any,
        hidden: Any,
        lang_emb: Any | None = None,
        step_metadata: dict[str, Any] | None = None,
    ) -> Any:
        if self._effective_envs_per_worker() == 1:
            if step_metadata is None:
                return envs.execute_on(int(env_id)).step(action, hidden, lang_emb)
            return envs.execute_on(int(env_id)).step(
                action,
                hidden,
                lang_emb,
                step_metadata,
            )
        worker_rank, slot_id = self._env_worker_rank_slot(int(env_id))
        if step_metadata is None:
            return envs.execute_on(worker_rank).step_slot(
                slot_id,
                action,
                hidden,
                lang_emb,
            )
        return envs.execute_on(worker_rank).step_slot(
            slot_id,
            action,
            hidden,
            lang_emb,
            step_metadata,
        )

    def _rollout_step_metadata(
        self,
        *,
        env_step: int,
        learner_updates: int,
        policy_version: int,
        episode_count: int,
        env_id: int,
        task_state: dict[str, Any],
    ) -> dict[str, Any]:
        del policy_version, episode_count, env_id, task_state
        return {
            "global_step": int(learner_updates),
            "env_step": int(env_step),
        }

    def _env_set_task(
        self,
        envs: Any,
        *,
        env_id: int,
        task_id: int,
        start_episode_id: int = 0,
    ) -> Any:
        if self._effective_envs_per_worker() == 1:
            return envs.execute_on(int(env_id)).set_task(task_id, start_episode_id)
        worker_rank, slot_id = self._env_worker_rank_slot(int(env_id))
        return envs.execute_on(worker_rank).set_task_slot(
            slot_id, task_id, start_episode_id
        )

    @staticmethod
    def _shared_resource_rank_map(devices: list[int], num_workers: int) -> str:
        if not devices:
            raise ValueError("render_devices must not be empty")
        workers = int(num_workers)
        if workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        if workers % len(devices) != 0:
            raise ValueError(
                "RLinf-style shared placement requires env.num_workers to be "
                f"divisible by render device count ({workers} vs {len(devices)})"
            )
        workers_per_device = workers // len(devices)
        parts: list[str] = []
        start_rank = 0
        for device in devices:
            end_rank = start_rank + workers_per_device - 1
            ranks = (
                str(start_rank)
                if start_rank == end_rank
                else f"{start_rank}-{end_rank}"
            )
            parts.append(f"{int(device)}:{ranks}")
            start_rank = end_rank + 1
        return ",".join(parts)

    def _env_worker_env_vars(self) -> dict[str, str]:
        backend = str(
            self._select_first(("render_backend", "env.render_backend"), "egl")
        ).lower()
        if backend != "egl":
            return {}
        return {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}

    def _ray_compute_device_pool(self) -> list[int]:
        """GPU ids used by non-env Ray components from Hydra placement config."""
        devices: set[int] = set()
        devices.update(self._packed_placement_devices("inference.placement"))
        devices.update(self._packed_placement_devices("learner.placement"))
        devices.update(self._flexible_placement_devices("learner.placement"))
        return sorted(devices)

    def _packed_placement_devices(self, path: str) -> list[int]:
        cfg = self._cfg_from(path, {"strategy": "node"})
        strategy = str(cfg.get("strategy", "node")).strip().lower()
        if strategy != "packed":
            return []
        start = int(cfg.get("gpu_id", cfg.get("start_gpu", 0)))
        end = int(cfg.get("end_gpu", start))
        return list(range(start, end + 1))

    def _flexible_placement_devices(self, path: str) -> list[int]:
        cfg = self._cfg_from(path, {"strategy": "node"})
        strategy = str(cfg.get("strategy", "node")).strip().lower()
        if strategy != "flexible":
            return []
        groups = cfg.get("accelerator_groups", cfg.get("groups", []))
        devices: list[int] = []
        for group in groups or []:
            devices.extend(parse_device_ids(group))
        return devices

    def _oft_worker_plan(self) -> dict[str, Any]:
        from dreamervla.runners.cold_start_ray_collect_runner import (
            ColdStartRayCollectRunner,
        )

        return ColdStartRayCollectRunner.build_oft_worker_plan(self)

    def _rollout_env_cfg(
        self,
        *,
        oft_plan: dict[str, Any] | None = None,
        horizon: int | None = None,
        use_oft_collect_path: bool | None = None,
    ) -> dict[str, Any]:
        use_collect_plan = (
            self._ray_rollout_mode() == "oft_fixed_base"
            if use_oft_collect_path is None
            else bool(use_oft_collect_path)
        )
        if use_collect_plan:
            plan = oft_plan if oft_plan is not None else self._oft_worker_plan()
            env_cfg = _plain(plan["env"])
        else:
            env_cfg = self._cfg_from(
                "env.cfg",
                {
                    "target": "dreamervla.workers.env._test_envs:CounterEnv",
                    "kwargs": {
                        "horizon": int(horizon or 3),
                        "image_shape": (4, 4, 3),
                        "embedding_dim": 4,
                    },
                },
            )
        env_cfg.setdefault("kwargs", {})
        # `horizon` is a synthetic-test-env (CounterEnv) kwarg; the real
        # DreamerVLAOnlineTrainEnv (use_from_config) rejects it and uses max_steps.
        if not bool(env_cfg.get("use_from_config")):
            env_cfg["kwargs"].setdefault("horizon", int(horizon or 3))
        render_backend = str(
            self._select_first(("render_backend", "env.render_backend"), "osmesa")
        ).lower()
        env_cfg["render_backend"] = render_backend
        env_cfg["num_envs_per_worker"] = self._effective_envs_per_worker()
        if (
            render_backend == "egl"
            and self._component_placement_strategy("env") is None
        ):
            self._egl_device_pool()
        return env_cfg

    def _int_from(self, paths: tuple[str, ...], default: int) -> int:
        return int(self._select_first(paths, default))

    def _target_global_steps(self) -> int | None:
        for path in (
            "training.max_steps",
            "training.max_train_steps",
            "runner.max_steps",
        ):
            value = OmegaConf.select(self.cfg, path, default=None)
            if value is None:
                continue
            steps = int(value)
            return steps if steps >= 0 else None
        return None

    def _cotrain_should_continue(
        self,
        global_step: int,
        target_global_steps: int | None,
        env_steps: int,
        target_env_steps: int,
    ) -> bool:
        if (
            target_global_steps is not None
            and int(global_step) >= int(target_global_steps)
        ):
            return False
        return int(env_steps) < int(target_env_steps)

    def _cotrain_can_launch_learner(
        self, global_step: int, target_global_steps: int | None
    ) -> bool:
        return target_global_steps is None or int(global_step) < int(target_global_steps)

    def _console_cotrain_progress(
        self,
        global_step: int,
        target_global_steps: int | None,
        env_steps: int,
        target_env_steps: int,
        *,
        phase: str = "rollout",
        episode_count: int = 0,
        episode_successes: int = 0,
        active_task_by_env: dict[int, int] | None = None,
        episode_steps_by_env: dict[int, int] | None = None,
        last_loss: float = 0.0,
        last_metrics: dict[str, float] | None = None,
    ) -> None:
        status = self._cotrain_progress_status(
            phase=str(phase),
            global_step=int(global_step),
            train_step=self._learner_train_step_status(str(phase)),
            env_steps=int(env_steps),
            target_env_steps=int(target_env_steps),
            episode_count=int(episode_count),
            episode_successes=int(episode_successes),
            active_task_by_env=active_task_by_env,
            episode_steps_by_env=episode_steps_by_env,
            last_loss=float(last_loss),
            last_metrics=last_metrics,
        )
        if target_global_steps is not None:
            self.console_progress(
                int(global_step),
                int(target_global_steps),
                "train",
                unit="step",
                status=status,
            )
            return
        self.console_progress(
            int(env_steps),
            int(target_env_steps),
            "rollout",
            unit="env",
            status=status,
        )

    def _cfg_from(self, path: str, default: dict[str, Any]) -> dict[str, Any]:
        value = OmegaConf.select(self.cfg, path, default=None)
        if value is None:
            return _plain(default)
        return _plain(value)

    def _load_init_ckpt(self, path: str) -> dict[str, Any]:
        if path == "learner.init_ckpt":
            payload = self._ray_resume_payload()
            state_dicts = payload.get("state_dicts") if isinstance(payload, dict) else None
            if isinstance(state_dicts, dict):
                init_cfg = OmegaConf.select(self.cfg, path, default=None)
                init_plain = _plain(init_cfg) if init_cfg is not None else {}
                components = (
                    init_plain.get("components")
                    if isinstance(init_plain, dict)
                    else None
                )
                names = list(state_dicts) if components is None else [str(item) for item in components]
                missing = [name for name in names if name not in state_dicts]
                if missing:
                    raise RuntimeError(
                        "resume checkpoint missing learner state_dicts for "
                        f"component(s): {missing}"
                    )
                return {name: state_dicts[name] for name in names}

        cfg = OmegaConf.select(self.cfg, path, default=None)
        if cfg is None:
            return {}
        plain = _plain(cfg)
        if not plain:
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

    def _learner_placement(self) -> PlacementStrategy:
        component_strategy = self._component_placement_strategy("learner", "actor")
        if component_strategy is not None:
            return component_strategy
        num_workers_raw = OmegaConf.select(self.cfg, "learner.num_workers", default=None)
        num_workers = int(num_workers_raw) if num_workers_raw is not None else None
        placement_cfg = self._cfg_from("learner.placement", {"strategy": "node"})
        strategy = str(placement_cfg.get("strategy", "node")).strip().lower()

        if strategy in {"", "node", "cpu"}:
            return NodePlacementStrategy(num_workers or 1)
        if strategy == "packed":
            num_gpus_per_worker = int(placement_cfg.get("num_gpus_per_worker", 1))
            start_gpu = int(placement_cfg.get("start_gpu", 0))
            if "end_gpu" in placement_cfg:
                end_gpu = int(placement_cfg["end_gpu"])
            else:
                workers = num_workers or 1
                end_gpu = start_gpu + workers * num_gpus_per_worker - 1
            packed = PackedPlacementStrategy(
                start_gpu,
                end_gpu,
                num_gpus_per_worker=num_gpus_per_worker,
            )
            actual_workers = (end_gpu - start_gpu + 1) // num_gpus_per_worker
            if num_workers is not None and actual_workers != num_workers:
                raise ValueError(
                    "learner.num_workers must match packed learner placement "
                    f"({num_workers} != {actual_workers})"
                )
            return packed
        if strategy == "flexible":
            groups = placement_cfg.get("accelerator_groups")
            if groups is None:
                groups = placement_cfg.get("groups")
            if groups is None:
                raise ValueError(
                    "learner.placement.accelerator_groups is required for flexible placement"
                )
            flexible = FlexiblePlacementStrategy(groups)
            actual_workers = len(flexible.accelerator_groups)
            if num_workers is not None and actual_workers != num_workers:
                raise ValueError(
                    "learner.num_workers must match flexible learner placement "
                    f"({num_workers} != {actual_workers})"
                )
            return flexible
        raise ValueError(
            "learner.placement.strategy must be one of node, packed, or flexible; "
            f"got {strategy!r}"
        )

    def _inference_placement(self) -> PlacementStrategy:
        component_strategy = self._component_placement_strategy("inference", "rollout")
        if component_strategy is not None:
            return component_strategy
        placement_cfg = self._cfg_from("inference.placement", {"strategy": "node"})
        strategy = str(placement_cfg.get("strategy", "node")).strip().lower()
        if strategy in {"", "node", "cpu"}:
            return NodePlacementStrategy(1)
        if strategy == "packed":
            gpu_id = int(placement_cfg.get("gpu_id", placement_cfg.get("start_gpu", 0)))
            end_gpu = int(placement_cfg.get("end_gpu", gpu_id))
            return PackedPlacementStrategy(gpu_id, end_gpu, num_gpus_per_worker=1)
        raise ValueError(
            "inference.placement.strategy must be one of node or packed; "
            f"got {strategy!r}"
        )

    def _learner_train_cfg(
        self,
        store_name: str,
        *,
        placement_has_gpu: bool,
    ) -> dict[str, Any]:
        learner_train_cfg = self._cfg_from(
            "learner.train_cfg",
            {
                "mode": "synthetic_ppo",
                "batch_size": int(self.cfg.get("ppo_batch_size", 2)),
                "lr": float(self.cfg.get("ppo_lr", 0.05)),
                "syncer": {"store_name": store_name},
            },
        )
        learner_train_cfg.setdefault("mode", "synthetic_ppo")
        learner_train_cfg.setdefault("batch_size", int(self.cfg.get("ppo_batch_size", 2)))
        learner_train_cfg.setdefault("lr", float(self.cfg.get("ppo_lr", 0.05)))
        progress_path = learner_train_cfg.get("progress_path")
        if progress_path in (None, ""):
            try:
                progress_path = self.get_diagnostics_dir() / "learner_progress.json"
            except AttributeError:
                run_dir = Path(
                    str(OmegaConf.select(self.cfg, "training.out_dir", default="."))
                ).expanduser()
                progress_path = run_dir / "diagnostics" / "learner_progress.json"
        self._learner_progress_path = str(Path(str(progress_path)).expanduser())
        learner_train_cfg["progress_path"] = self._learner_progress_path
        raw_device = str(
            learner_train_cfg.get("device", "auto" if placement_has_gpu else "cpu")
        ).strip()
        if raw_device.lower() in {"", "auto"}:
            learner_train_cfg["device"] = "cuda:0" if placement_has_gpu else "cpu"
        else:
            learner_train_cfg["device"] = raw_device
        learner_train_cfg.setdefault("syncer", {})
        learner_train_cfg["syncer"].setdefault("store_name", store_name)
        return learner_train_cfg


def _default_policy_cfg() -> dict[str, Any]:
    return {
        "target": "dreamervla.workers.actor._test_models:TinySharedPolicy",
        "kwargs": {"hidden_dim": 4, "action_dim": 7},
    }


def _default_inference_cfg(policy_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "encoder": {"target": "dreamervla.workers.inference._test_models:TinyEncoder"},
        "world_model": {
            "target": "dreamervla.workers.inference._test_models:TinyWorldModel",
            "kwargs": {"hidden_dim": 4, "action_dim": 7},
        },
        "policy": _plain(policy_cfg),
        "device": "cpu",
    }


def _default_preprocess_config() -> dict[str, Any]:
    return {
        "action_head_type": "oft_discrete_token",
        "obs_hidden_source": "input_token_embedding",
        "history": 1,
        "include_state": False,
        "hidden_key": "obs_embedding",
        "token_count": 256,
        "token_dim": 4096,
        "hidden_dim": 1048576,
        "obs_embedding_shape": [256, 4096],
        "hidden_storage_format": "tokenized",
        "num_images_in_input": 1,
        "patches_per_image": 256,
        "sidecar_schema_version": 1,
        "required_demo_datasets": ["obs_embedding"],
    }


def _build_cotrain_dump_step(
    env: Any,
    obs: dict[str, Any],
    action: Any,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    obs_embedding: Any,
    lang_emb: Any | None = None,
) -> dict[str, Any]:
    from dreamervla.workers.rollout.record_adapter import build_dump_step

    done = bool(terminated or truncated)
    success = bool(info.get("success", terminated))
    wm_action = info.get("wm_action", info.get("env_action", action))
    full_record = obs.get("_full_record") if isinstance(obs, dict) else None
    if full_record is None:
        full_record = env.full_record()
    step = build_dump_step(
        full_record=full_record,
        obs_embedding=obs_embedding,
        lang_emb=lang_emb,
        action=wm_action,
        reward=0.0,
        sparse_reward=(1 if done and success else 0),
        done=done,
    )
    step["task_id"] = int(info.get("task_id", obs.get("task_id", 0)))
    step["episode_id"] = int(info.get("episode_id", obs.get("episode_id", 0)))
    init_state_index = info.get(
        "init_state_index",
        obs.get("init_state_index", full_record.get("init_state_index")),
    )
    if init_state_index is not None:
        step["init_state_index"] = int(init_state_index)
    step["task_description"] = str(
        info.get("task_description", obs.get("task_description", ""))
    )
    step["success"] = success
    return step


def _rollout_episode_resume_counts_from_dump(
    reward_dir: str,
    hidden_dir: str,
    *,
    task_ids: tuple[int, ...],
) -> dict[int, int]:
    complete_ids = complete_episode_ids_per_task(reward_dir, hidden_dir)
    counts: dict[int, int] = {}
    for task_id in task_ids:
        ids = complete_ids.get(int(task_id), set())
        counts[int(task_id)] = (max(ids) + 1) if ids else 0
    return counts


def _normalise_task_episode_counts(value: Any) -> dict[int, int]:
    if not isinstance(value, Mapping):
        return {}
    counts: dict[int, int] = {}
    for key, item in value.items():
        try:
            counts[int(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return counts


def _ray_state_from_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = payload.get("ray")
    if isinstance(raw, Mapping):
        state = dict(raw)
    elif "global_step" in payload or "env_step" in payload or "env_steps" in payload:
        state = {}
    else:
        return None

    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    if "global_step" not in state and "global_step" in payload:
        state["global_step"] = payload["global_step"]
    if "env_step" not in state and "env_steps" not in state:
        env_value = payload.get("env_step", payload.get("env_steps", metrics.get("rollout/steps", 0)))
        state["env_step"] = env_value
    return {
        "global_step": int(state.get("global_step", state.get("update_step", 0))),
        "env_step": int(state.get("env_step", state.get("env_steps", 0))),
    }


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


def _uses_ray_worker_groups(groups: dict[str, Any]) -> bool:
    return all(
        hasattr(groups.get(name), "workers")
        for name in ("envs", "infer", "replay", "learner")
    )


def _resolve_worker_cls(target: str) -> type:
    """Resolve a ``module:Class`` (or ``module.Class``) worker target to its class."""
    text = str(target)
    if ":" in text:
        module_name, class_name = text.split(":", 1)
    else:
        module_name, class_name = text.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _load_runner_state_dicts(
    ckpt_path: str,
    *,
    components: list[str] | None,
) -> dict[str, Any]:
    path = Path(ckpt_path).expanduser()
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
        # Pipeline warmup ckpts store the component under its own top-level key
        # (wm_warmup.ckpt -> {"world_model": sd, "global_step": ...};
        # classifier_warmup.ckpt -> {"classifier": sd, "classifier_threshold": ...}).
        # Extract it so the ray async runner's component-mapping init_ckpt can consume the
        # pipeline warmup files directly (the warmup -> async bridge). Must precede the
        # all-string-keys catch-all, which would otherwise return the whole wrapper dict.
        component_sd = payload.get(component)
        if isinstance(component_sd, dict):
            return component_sd
        if all(isinstance(key, str) for key in payload):
            return payload
    raise RuntimeError(
        f"{path} does not contain a usable state_dict for component {component!r}"
    )


def _learner_loss(metrics: dict[str, Any]) -> float:
    for key in ("train/rl_loss", "rl/actor_loss", "wm/loss", "cls/loss"):
        if key in metrics:
            return float(metrics[key])
    return 0.0


def _float_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in dict(metrics).items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _wait_all(result: Any) -> list[Any]:
    if hasattr(result, "wait"):
        return list(result.wait())
    return [result]


def _wait_first(result: Any) -> Any:
    values = _wait_all(result)
    return values[0] if values else None
