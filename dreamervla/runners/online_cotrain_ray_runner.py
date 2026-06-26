"""Opt-in Ray online cotrain runner.

This first runner wires the new scheduler/workers into a lightweight synthetic
online loop that exercises the same production boundaries: env rollout,
batched inference, replay insertion, learner PPO-style update, and policy
weight sync. Real LIBERO/VLA construction can plug into these boundaries
without changing the scheduler primitives.
"""

from __future__ import annotations

import importlib
import os
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.constants import CHECKPOINT_FORMAT_VERSION
from dreamervla.runners.base_runner import (
    BaseRunner,
    _atomic_torch_save,
    _materialize_checkpoint_copy,
)
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import (
    FlexiblePlacementStrategy,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    PlacementStrategy,
)
from dreamervla.scheduler.worker_group import WorkerGroup
from dreamervla.utils.resource_metrics import collect_resource_metrics
from dreamervla.workers.actor.learner_worker import LearnerWorker
from dreamervla.workers.env.env_worker import EnvWorker
from dreamervla.workers.inference.rollout_inference_worker import RolloutInferenceWorker
from dreamervla.workers.replay.replay_worker import ReplayWorker


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
            metrics["env/num_env_workers"] = int(groups["num_envs"])
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
        num_envs = self._int_from(("env.num_workers", "num_env_workers"), 2)
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
        # config-selectable. Default = RynnVLA encoder->WM->actor InferenceWorker; OFT
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
        env_group = WorkerGroup(EnvWorker, env_cfg, task_id=0, replay=replay).launch(
            cluster, NodePlacementStrategy(num_envs)
        )
        # Spread env workers across the configured task ids (round-robin) instead of
        # pinning every env to task 0. The env config itself now comes from the collect
        # plan for OFT, so the external set_task calls mirror collect scheduling rather
        # than compensating for hand-authored env defaults.
        rollout_task_ids = self._rollout_task_ids()
        if rollout_task_ids:
            for env_index in range(num_envs):
                tid = int(rollout_task_ids[env_index % len(rollout_task_ids)])
                if tid != 0:
                    env_group.execute_on(env_index).set_task(tid).wait()

        if infer_worker_cls is RolloutInferenceWorker:
            # OFT recipe: build the OFT rollout inference cfg via the SAME programmatic
            # derivation the collect path uses (OFTRolloutBundle -> action + obs_embedding),
            # not the RynnVLA encoder->WM->actor _default_inference_cfg. DRY: no hand-authored
            # OFT field YAML. init_ckpt stays empty — the OFT base policy loads from the
            # bundle's model_path; the learned actor trains only in imagination.
            infer_cfg = dict((oft_plan or self._oft_worker_plan())["inference"])
            infer_init_ckpt: dict[str, Any] = {}
        else:
            infer_cfg = self._cfg_from("inference.cfg", _default_inference_cfg(policy_cfg))
            infer_cfg.setdefault("policy", policy_cfg)
            infer_cfg.setdefault("device", "cpu")
            infer_init_ckpt = self._load_init_ckpt("inference.init_ckpt")
        infer_group = WorkerGroup(
            infer_worker_cls,
            infer_cfg,
            infer_init_ckpt,
            num_envs=num_envs,
        ).launch(cluster, self._inference_placement())

        learner_model_cfg = self._cfg_from("learner.model_cfg", {"policy": policy_cfg})
        learner_model_cfg.setdefault("policy", policy_cfg)
        learner_init_ckpt = self._load_init_ckpt("learner.init_ckpt")
        learner_placement = self._learner_placement()
        learner_placements = learner_placement.get_placement(cluster)
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
        return {
            "replay": replay_group,
            "envs": env_group,
            "infer": infer_group,
            "learner": learner_group,
            "store_name": store_name,
            "num_envs": num_envs,
        }

    def _run_loop(self, groups: dict[str, Any]) -> dict[str, float | int]:
        self.console_banner("ONLINE COTRAIN (ray)", subtitle=f"envs={groups['num_envs']}")
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

        learner_updates = int(getattr(self, "global_step", 0) or 0)
        self.global_step = learner_updates
        start_global_step = int(learner_updates)
        train_start_t = time.perf_counter()
        policy_version = 0
        local_infer_version = 0
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
        env_steps = 0
        infer_batches = 0
        episode_count = 0
        episode_successes = 0
        last_episode_success = 0
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
                episode_count=episode_count,
                episode_successes=episode_successes,
                active_task_by_env=active_task_by_env,
                episode_steps_by_env=episode_steps_by_env,
                last_loss=last_loss,
                last_metrics=last_metrics,
            )
            obs_batch_all = envs.current_obs().wait()
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
                    envs.execute_on(rank).step(action, hidden, lang_emb).wait()
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
            episode_count=episode_count,
            episode_successes=episode_successes,
            active_task_by_env=active_task_by_env,
            episode_steps_by_env=episode_steps_by_env,
            last_loss=last_loss,
            last_metrics=last_metrics,
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

        env_steps = 0
        infer_batches = 0
        learner_updates = int(getattr(self, "global_step", 0) or 0)
        self.global_step = learner_updates
        start_global_step = int(learner_updates)
        train_start_t = time.perf_counter()
        policy_version = 0
        local_infer_version = 0
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
        episode_count = 0
        episode_successes = 0
        last_episode_success = 0
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
                result = envs.execute_on(int(env_id)).step(action, hidden, lang_emb)
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
            )
            pending_learn = None
            pending_learn_start = 0.0
            pending_learn_overlapped = False

        initial_obs = envs.current_obs().wait()
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
            episode_count=episode_count,
            episode_successes=episode_successes,
            active_task_by_env=active_task_by_env,
            episode_steps_by_env=episode_steps_by_env,
            last_loss=last_loss,
            last_metrics=last_metrics,
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
            legacy_interval = cfg.get(
                "every_updates",
                OmegaConf.select(self.cfg, "training.checkpoint_every", default=0),
            )
            cfg["save_interval"] = legacy_interval
        cfg.setdefault("save_final", False)
        cfg.setdefault("filename", "learner.ckpt")
        cfg.setdefault("latest_name", "latest.ckpt")
        return cfg

    def _maybe_save_ray_checkpoint(
        self,
        groups: dict[str, Any],
        *,
        env_steps: int,
        learner_updates: int,
        policy_version: int,
        metrics: dict[str, Any],
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
            env_steps=env_steps,
            learner_updates=global_step,
            policy_version=policy_version,
            metrics=metrics,
            checkpoint_cfg=cfg,
        )

    def _save_ray_checkpoint(
        self,
        learner: Any,
        *,
        env_steps: int,
        learner_updates: int,
        policy_version: int,
        metrics: dict[str, Any],
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
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "global_step": global_step,
            "cfg": self.cfg,
            "state_dicts": self._ray_learner_state_dicts(learner),
            "pickles": {},
            "ray": {
                "global_step": global_step,
                "env_step": int(env_steps),
                "update_step": global_step,
                "policy_version": int(policy_version),
            },
            "metrics": _float_metrics(metrics),
        }
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
                update_step=int(global_step),
                policy_version=int(policy_version),
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
        task_ids = self._rollout_task_ids()
        active_task_by_env: dict[int, int] = {}
        if task_ids:
            for env_id in range(max(0, int(num_envs))):
                active_task_by_env[int(env_id)] = int(task_ids[env_id % len(task_ids)])
        return {
            "task_ids": task_ids,
            "active_task_by_env": active_task_by_env,
            "task_episode_counts": {int(task_id): 0 for task_id in task_ids},
            "next_task_index": int(num_envs) if task_ids else 0,
        }

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
        task_episode_counts = task_state["task_episode_counts"]
        env_id = int(env_id)
        current_task = active_task_by_env.get(env_id)
        if current_task is not None:
            task_episode_counts[int(current_task)] = (
                int(task_episode_counts.get(int(current_task), 0)) + 1
            )
        next_index = int(task_state.get("next_task_index", 0))
        next_task = int(task_ids[next_index % len(task_ids)])
        task_state["next_task_index"] = next_index + 1
        active_task_by_env[env_id] = next_task
        start_episode_id = int(task_episode_counts.get(next_task, 0))
        switched = envs.execute_on(env_id).set_task(next_task, start_episode_id).wait()
        if isinstance(switched, list):
            return dict(switched[0])
        return dict(switched)

    def _cotrain_progress_status(
        self,
        *,
        env_steps: int,
        target_env_steps: int,
        episode_count: int,
        episode_successes: int,
        active_task_by_env: dict[int, int] | None = None,
        episode_steps_by_env: dict[int, int] | None = None,
        last_loss: float = 0.0,
        last_metrics: dict[str, float] | None = None,
    ) -> str:
        parts = [f"env_steps={int(env_steps)}/{int(target_env_steps)}"]
        active = dict(active_task_by_env or {})
        ep_steps = dict(episode_steps_by_env or {})
        if active:
            pairs = []
            for env_id in sorted(active):
                task_id = int(active[env_id])
                step = int(ep_steps.get(int(env_id), 0))
                pairs.append(f"t{task_id}:s{step}")
            parts.append("collect=" + ",".join(pairs[:4]))
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

    def _egl_device_pool(self) -> list[int]:
        """Physical GPU ids for egl env rendering (default backend). Read from the driver's
        CUDA_VISIBLE_DEVICES; empty when the render backend is osmesa."""
        backend = str(
            self._select_first(("render_backend", "env.render_backend"), "egl")
        ).lower()
        if backend != "egl":
            return []
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        return [int(x) for x in cvd.split(",") if x.strip().isdigit()]

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
        egl_pool = self._egl_device_pool()
        if egl_pool:
            env_cfg["egl_device_pool"] = egl_pool
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
        episode_count: int = 0,
        episode_successes: int = 0,
        active_task_by_env: dict[int, int] | None = None,
        episode_steps_by_env: dict[int, int] | None = None,
        last_loss: float = 0.0,
        last_metrics: dict[str, float] | None = None,
    ) -> None:
        status = self._cotrain_progress_status(
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
