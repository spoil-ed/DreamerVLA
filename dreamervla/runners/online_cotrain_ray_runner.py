"""Opt-in Ray online cotrain runner.

This first runner wires the new scheduler/workers into a lightweight synthetic
online loop that exercises the same production boundaries: env rollout,
batched inference, replay insertion, learner PPO-style update, and policy
weight sync. Real LIBERO/VLA construction can plug into these boundaries
without changing the scheduler primitives.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.runners.base_runner import BaseRunner
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
from dreamervla.workers.inference.inference_worker import InferenceWorker
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
            return metrics
        finally:
            cluster.shutdown()

    def _build_components(self, cluster: Cluster) -> dict[str, Any]:
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

        env_cfg = self._cfg_from(
            "env.cfg",
            {
                "target": "dreamervla.workers.env._test_envs:CounterEnv",
                "kwargs": {"horizon": horizon, "image_shape": (4, 4, 3), "embedding_dim": 4},
            },
        )
        env_cfg.setdefault("kwargs", {})
        env_cfg["kwargs"].setdefault("horizon", horizon)
        env_group = WorkerGroup(EnvWorker, env_cfg, task_id=0, replay=replay).launch(
            cluster, NodePlacementStrategy(num_envs)
        )

        policy_cfg = self._cfg_from("policy.cfg", _default_policy_cfg())
        infer_cfg = self._cfg_from("inference.cfg", _default_inference_cfg(policy_cfg))
        infer_cfg.setdefault("policy", policy_cfg)
        infer_cfg.setdefault("device", "cpu")
        infer_init_ckpt = self._load_init_ckpt("inference.init_ckpt")
        infer_group = WorkerGroup(
            InferenceWorker,
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
        self.console_banner("ONLINE COTRAIN (ray)", done=True)
        return result

    def _run_loop_sync(self, groups: dict[str, Any]) -> dict[str, float | int]:
        envs = groups["envs"]
        infer = groups["infer"]
        replay = groups["replay"]
        learner = groups["learner"]
        env_ids = list(range(int(groups["num_envs"])))
        rollout_steps = self._int_from(("rollout.steps", "rollout_steps"), 9)
        min_episodes = self._int_from(
            ("rollout.min_replay_episodes", "min_replay_episodes"), 1
        )
        weight_sync_every = self._int_from(
            ("sync.weight_sync_every", "weight_sync_every"), 1
        )
        learner_phase = self._learner_update_phase()

        learner_updates = 0
        policy_version = 0
        local_infer_version = 0
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

        for step in range(rollout_steps):
            obs_batch = envs.current_obs().wait()

            if pending_learn is None and replay.ready(min_episodes).wait()[0]:
                pending_learn = learner.update(learner_phase, 1)
                pending_learn_start = time.perf_counter()

            rollout_start = time.perf_counter()
            infer_start = time.perf_counter()
            infer_out = infer.forward_batch(obs_batch, env_ids).wait()[0]
            infer_wait_s += time.perf_counter() - infer_start
            for key, value in dict(infer_out.get("timing", {})).items():
                metric_key = f"time/infer_{key}"
                infer_stage_timing[metric_key] = infer_stage_timing.get(
                    metric_key, 0.0
                ) + float(value)
            step_results = []
            for rank, action, hidden in zip(
                env_ids, infer_out["actions"], infer_out["obs_embedding"], strict=True
            ):
                env_step_start = time.perf_counter()
                step_results.extend(envs.execute_on(rank).step(action, hidden).wait())
                env_step_wait_s += time.perf_counter() - env_step_start
            rollout_end = time.perf_counter()

            done_envs = []
            for env_id, (_obs, done, info) in zip(env_ids, step_results, strict=True):
                if done:
                    done_envs.append(env_id)
                    self.console_record_success(bool((info or {}).get("success", False)))
            if done_envs:
                infer.reset_states(done_envs).wait()

            if pending_learn is not None:
                learn_done = pending_learn.done()
                if learn_done or step == rollout_steps - 1:
                    learner_wait_start = time.perf_counter()
                    metrics = pending_learn.wait()[0]
                    learner_wait_s += time.perf_counter() - learner_wait_start
                    last_metrics = _float_metrics(metrics)
                    last_loss = _learner_loss(metrics)
                    learner_updates += 1
                    policy_version += 1
                    update_metrics = {"train/rl_loss": last_loss, **last_metrics}
                    self.console_metrics(f"cotrain · step {learner_updates}", update_metrics)
                    self.log_metrics(update_metrics, step=learner_updates)
                    sync_start = time.perf_counter()
                    learner.sync_weights("policy", policy_version).wait()
                    if policy_version % weight_sync_every == 0:
                        pulled = infer.pull_weights(
                            groups["store_name"], "policy", local_infer_version
                        ).wait()[0]
                        if pulled is not None:
                            local_infer_version = int(pulled)
                    weight_sync_wait_s += time.perf_counter() - sync_start
                    if (
                        pending_learn_start < rollout_end
                        and rollout_start < time.perf_counter()
                    ):
                        overlap_events += 1
                    pending_learn = None

        if pending_learn is not None:
            learner_wait_start = time.perf_counter()
            metrics = pending_learn.wait()[0]
            learner_wait_s += time.perf_counter() - learner_wait_start
            last_metrics = _float_metrics(metrics)
            last_loss = _learner_loss(metrics)
            learner_updates += 1
            policy_version += 1
            update_metrics = {"train/rl_loss": last_loss, **last_metrics}
            self.console_metrics(f"cotrain · step {learner_updates}", update_metrics)
            self.log_metrics(update_metrics, step=learner_updates)
            learner.sync_weights("policy", policy_version).wait()

        return {
            "rollout/episodes": int(replay.size().wait()[0]),
            "rollout/steps": int(rollout_steps),
            "train/learner_updates": int(learner_updates),
            # Compatibility for older smoke tests and dashboards.
            "train/ppo_updates": int(learner_updates),
            "sync/policy_version": int(policy_version),
            "time/overlap_events": int(overlap_events),
            "time/rollout_overlap_events": 0,
            "time/rollout_infer_ready_batches": int(rollout_steps),
            "time/rollout_env_ready_batches": int(rollout_steps * len(env_ids)),
            "time/infer_wait_s": float(infer_wait_s),
            "time/env_step_wait_s": float(env_step_wait_s),
            "time/learner_wait_s": float(learner_wait_s),
            "time/weight_sync_wait_s": float(weight_sync_wait_s),
            "train/rl_loss": float(last_loss),
            **infer_stage_timing,
            **last_metrics,
            **collect_resource_metrics(prefix="time"),
        }

    def _run_loop_overlap(self, groups: dict[str, Any]) -> dict[str, float | int]:
        import ray

        envs = groups["envs"]
        infer = groups["infer"]
        replay = groups["replay"]
        learner = groups["learner"]
        env_ids = list(range(int(groups["num_envs"])))
        rollout_steps = self._int_from(("rollout.steps", "rollout_steps"), 9)
        min_episodes = self._int_from(
            ("rollout.min_replay_episodes", "min_replay_episodes"), 1
        )
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

        steps = 0
        learner_updates = 0
        policy_version = 0
        local_infer_version = 0
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

        def add_infer_timing(infer_out: dict[str, Any]) -> None:
            for key, value in dict(infer_out.get("timing", {})).items():
                metric_key = f"time/infer_{key}"
                infer_stage_timing[metric_key] = infer_stage_timing.get(
                    metric_key, 0.0
                ) + float(value)

        def launch_infer() -> None:
            nonlocal steps, rollout_overlap_events, rollout_strict_overlap_events
            if pending_infers or not ready_obs or steps >= rollout_steps:
                return
            batch = list(ready_obs)
            ready_obs.clear()
            batch_env_ids = [env_id for env_id, _obs in batch]
            obs_batch = [obs for _env_id, obs in batch]
            if pending_steps or steps > 0:
                rollout_overlap_events += 1
            if pending_steps:
                rollout_strict_overlap_events += 1
            result = infer.forward_batch(obs_batch, batch_env_ids)
            pending_infers[result.refs[0]] = (
                batch_env_ids,
                result,
                time.perf_counter(),
            )
            steps += 1

        def launch_steps(batch_env_ids: list[int], infer_out: dict[str, Any]) -> None:
            for env_id, action, hidden in zip(
                batch_env_ids,
                infer_out["actions"],
                infer_out["obs_embedding"],
                strict=True,
            ):
                result = envs.execute_on(int(env_id)).step(action, hidden)
                pending_steps[result.refs[0]] = (
                    int(env_id),
                    result,
                    time.perf_counter(),
                )

        def handle_step_result(
            env_id: int,
            step_result: tuple[dict[str, Any], bool, dict[str, Any]],
        ) -> None:
            next_obs, done, info = step_result
            if done:
                self.console_record_success(bool((info or {}).get("success", False)))
                infer.reset_states([int(env_id)]).wait()
            if steps < rollout_steps:
                ready_obs.append((int(env_id), next_obs))

        def maybe_launch_learner() -> None:
            nonlocal pending_learn, pending_learn_start, pending_learn_overlapped
            if pending_learn is not None:
                return
            if not bool(replay.ready(min_episodes).wait()[0]):
                return
            pending_learn = learner.update(learner_phase, 1)
            pending_learn_start = time.perf_counter()
            pending_learn_overlapped = bool(
                pending_infers or pending_steps or ready_obs or steps < rollout_steps
            )

        def finish_learner(*, block: bool) -> None:
            nonlocal pending_learn, pending_learn_start, pending_learn_overlapped
            nonlocal learner_updates, policy_version, local_infer_version
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
            policy_version += 1
            update_metrics = {"train/rl_loss": last_loss, **last_metrics}
            self.console_metrics(f"cotrain · step {learner_updates}", update_metrics)
            self.log_metrics(update_metrics, step=learner_updates)

            sync_start = time.perf_counter()
            learner.sync_weights("policy", policy_version).wait()
            if policy_version % weight_sync_every == 0:
                pulled = infer.pull_weights(
                    groups["store_name"], "policy", local_infer_version
                ).wait()[0]
                if pulled is not None:
                    local_infer_version = int(pulled)
            weight_sync_wait_s += time.perf_counter() - sync_start

            if pending_learn_overlapped and pending_learn_start <= time.perf_counter():
                overlap_events += 1
            pending_learn = None
            pending_learn_start = 0.0
            pending_learn_overlapped = False

        initial_obs = envs.current_obs().wait()
        ready_obs.extend(
            (int(env_id), obs)
            for env_id, obs in zip(env_ids, initial_obs, strict=True)
        )
        maybe_launch_learner()
        launch_infer()

        while pending_infers or pending_steps or ready_obs:
            finish_learner(block=False)
            maybe_launch_learner()
            launch_infer()
            refs = list(pending_infers) + list(pending_steps)
            if not refs:
                if steps >= rollout_steps:
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
            if steps >= rollout_steps:
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

        return {
            "rollout/episodes": int(replay.size().wait()[0]),
            "rollout/steps": int(steps),
            "train/learner_updates": int(learner_updates),
            # Compatibility for older smoke tests and dashboards.
            "train/ppo_updates": int(learner_updates),
            "sync/policy_version": int(policy_version),
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
            **collect_resource_metrics(prefix="time"),
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

    def _int_from(self, paths: tuple[str, ...], default: int) -> int:
        return int(self._select_first(paths, default))

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
