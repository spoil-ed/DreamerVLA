"""Opt-in Ray cold-start rollout collector."""

from __future__ import annotations

import time
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.dataset.collection_manifest import (
    complete_episode_ids_per_task,
    next_shard_index,
)
from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.collect_parallel_rollouts import _build_resume_work_list
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import NodePlacementStrategy, PackedPlacementStrategy
from dreamervla.scheduler.worker_group import WorkerGroup
from dreamervla.utils.egl_device import _ZERO_GPU_EGL_ERROR
from dreamervla.utils.paths import data_path
from dreamervla.utils.resource_metrics import collect_resource_metrics
from dreamervla.workers.env.env_worker import EnvWorker
from dreamervla.workers.inference.inference_worker import InferenceWorker
from dreamervla.workers.inference.rollout_inference_worker import RolloutInferenceWorker
from dreamervla.workers.rollout.dump_worker import RolloutDumpWorker


def _shard_env_ids_by_worker(env_ids: list[int], num_workers: int) -> dict[int, list[int]]:
    """Stable partition of env ids to inference workers (``env_id % num_workers``).

    Each env id always maps to the same owner worker so the worker's per-env
    extractor / action-queue state stays consistent across steps. Returns
    ``{worker_rank: [env_id, ...]}`` preserving input order within each worker;
    workers with no env ids are omitted. ``num_workers==1`` -> ``{0: env_ids}``.
    """
    num_workers = max(1, int(num_workers))
    groups: dict[int, list[int]] = {}
    for env_id in env_ids:
        groups.setdefault(int(env_id) % num_workers, []).append(int(env_id))
    return groups


def _ensure_collect_render_device_pool(
    env_cfg: dict[str, Any],
    cluster: Cluster,
    *,
    collect_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return collect env cfg with an EGL render pool when the user did not set one."""
    resolved = dict(env_cfg)
    if str(resolved.get("render_backend", "osmesa")).strip().lower() != "egl":
        return resolved

    from dreamervla.runners.render_device_config import parse_device_ids

    for key in ("gpu_pool", "render_devices", "egl_device_pool"):
        if parse_device_ids(resolved.get(key)):
            return resolved

    num_gpus = int(getattr(cluster, "num_gpus", 0))
    if num_gpus <= 0:
        raise ValueError(_ZERO_GPU_EGL_ERROR)
    collect_cfg = dict(collect_cfg or {})
    first_infer_gpu = int(collect_cfg.get("gpu_id", 0))
    num_infer = max(1, int(collect_cfg.get("num_inference_workers", 1)))
    inference_gpus = {
        gpu
        for gpu in range(first_infer_gpu, first_infer_gpu + num_infer)
        if 0 <= gpu < num_gpus
    }
    render_devices = [gpu for gpu in range(num_gpus) if gpu not in inference_gpus]
    if not render_devices:
        raise ValueError(
            "collect render_backend=egl needs a render GPU disjoint from inference "
            f"GPU(s) {sorted(inference_gpus)}; expose another GPU and set "
            "+env.cfg.render_devices, or use render_backend=osmesa."
        )
    resolved["render_devices"] = render_devices
    return resolved


class ColdStartRayCollectRunner(BaseRunner):
    """Collect rollout episodes with Ray env actors and write HDF5 sidecars."""

    runner_name = "collect_rollouts_ray"
    runner_status = "current"
    runner_family = "rollout"

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
        cluster = Cluster(self.cfg.get("cluster"))
        try:
            cluster.require_single_node()
            groups = self._build_components(cluster)
            return self._run_loop(groups)
        finally:
            cluster.shutdown()

    def _build_components(self, cluster: Cluster) -> dict[str, Any]:
        mode = str(self._select_first(("mode",), "synthetic")).lower()
        if mode == "oft":
            return self._build_oft_components(cluster)

        num_envs = self._int_from(("env.num_workers", "num_env_workers"), 1)
        demos_per_shard = self._int_from(("collect.demos_per_shard", "demos_per_shard"), 0)
        reward_dir_value = self._select_first(("dump.reward_dir", "reward_dir"), None)
        hidden_dir_value = self._select_first(("dump.hidden_dir", "hidden_dir"), None)
        reward_dir = str(
            reward_dir_value
            if reward_dir_value is not None
            else data_path("collected_rollouts", "ray_synthetic", "reward")
        )
        hidden_dir = str(
            hidden_dir_value
            if hidden_dir_value is not None
            else data_path("collected_rollouts", "ray_synthetic", "hidden")
        )
        shard_name = str(self._select_first(("dump.shard_name", "shard_name"), "ray_shard_000.hdf5"))
        preprocess_config = self._cfg_from("dump.preprocess_config", _default_preprocess_config())
        data_attrs = self._cfg_from("dump.data_attrs", {"task_suite_name": "synthetic", "env_name": "ray"})

        dump_group = WorkerGroup(
            RolloutDumpWorker,
            reward_dir,
            hidden_dir,
            shard_name,
            preprocess_config,
            data_attrs,
            demos_per_shard,
        ).launch(cluster, NodePlacementStrategy(1))
        dump = dump_group.workers[0]

        env_cfg = self._cfg_from(
            "env.cfg",
            {
                "target": "dreamervla.workers.env._test_envs:DumpCounterEnv",
                "kwargs": {"horizon": 3, "image_shape": (4, 4, 3), "embedding_dim": 4},
            },
        )
        env_group = WorkerGroup(EnvWorker, env_cfg, task_id=0, replay=dump).launch(
            cluster, NodePlacementStrategy(num_envs)
        )

        policy_cfg = self._cfg_from("policy.cfg", _default_policy_cfg())
        infer_cfg = self._cfg_from("inference.cfg", _default_inference_cfg(policy_cfg))
        infer_cfg.setdefault("policy", policy_cfg)
        infer_cfg.setdefault("device", "cpu")
        infer_group = WorkerGroup(InferenceWorker, infer_cfg, {}, num_envs=num_envs).launch(
            cluster, NodePlacementStrategy(1)
        )

        return {
            "dump": dump_group,
            "envs": env_group,
            "infer": infer_group,
            "num_envs": num_envs,
        }

    def build_oft_worker_plan(self) -> dict[str, Any]:
        """Assemble real OFT Ray worker configs without loading Ray or a model."""
        from dreamervla.runners.collect_rollouts_runner import CollectRolloutsRunner
        from dreamervla.runners.oft_collect_common import make_preprocess_config

        collect_cfg = CollectRolloutsRunner._build_collect_cfg(self)
        mode = (
            "l1"
            if collect_cfg["expected_action_head_type"] == "oft_l1_regression"
            else "discrete"
        )
        collect_cfg["_policy_mode"] = mode
        collect_cfg["_use_proprio"] = bool(collect_cfg["expected_include_state"])
        preprocess_config = make_preprocess_config(collect_cfg)
        policy_cfg = {
            "model_path": collect_cfg["model_path"],
            "policy_mode": collect_cfg["policy_mode"],
            "num_images_in_input": collect_cfg["num_images_in_input"],
            "unnorm_key": collect_cfg["unnorm_key"],
        }
        inference_device = str(
            self._select_first(("inference.device", "collect.inference_device"), "cuda")
        )
        env_cfg = self._cfg_from(
            "env.cfg",
            {
                "target": "dreamervla.envs.train_env:DreamerVLAOnlineTrainEnv",
                "use_from_config": True,
            },
        )
        env_kwargs = dict(env_cfg.get("kwargs", {}))
        env_kwargs.setdefault("task_suite_name", collect_cfg["task_suite_name"])
        env_kwargs.setdefault("task_id", _first_task_id(collect_cfg.get("task_ids", 0)))
        env_kwargs.setdefault("resolution", collect_cfg["resolution"])
        env_kwargs.setdefault("full_record", True)
        env_kwargs.setdefault("init_state_sampling", "sequential")
        env_kwargs.setdefault("action_input", "raw")
        env_kwargs.setdefault("pixel_rotate_180", False)
        env_kwargs.setdefault("vla_rotate_180", True)
        env_kwargs.setdefault("history_length", collect_cfg["expected_history"])
        env_kwargs.setdefault("include_state", collect_cfg["expected_include_state"])
        env_kwargs.setdefault("obs_hidden_source", collect_cfg["expected_obs_hidden_source"])
        env_kwargs.setdefault("action_head_type", collect_cfg["expected_action_head_type"])
        env_kwargs.setdefault("validate_canonical", False)
        env_kwargs.setdefault("max_steps", collect_cfg["episode_horizon"])
        env_cfg["kwargs"] = env_kwargs
        return {
            "collect": collect_cfg,
            "env": env_cfg,
            "inference": {
                "action_dim": collect_cfg["action_dim"],
                "action_steps": collect_cfg["chunk_size"],
                "device": inference_device,
                "decoder": {
                    "target": "dreamervla.workers.inference.oft_rollout:OFTRolloutBundle",
                    "kwargs": {
                        "policy_cfg": policy_cfg,
                        "unnorm_key": collect_cfg["unnorm_key"],
                        "image_keys": collect_cfg["image_keys"],
                        "history": collect_cfg["expected_history"],
                        "rotate_images_180": collect_cfg["expected_rotate_images_180"],
                        "center_crop": True,
                        "obs_hidden_source": collect_cfg["expected_obs_hidden_source"],
                        "expected_action_head_type": collect_cfg["expected_action_head_type"],
                        "expected_include_state": collect_cfg["expected_include_state"],
                        "device": inference_device,
                    },
                },
            },
            "dump": {
                "reward_dir": collect_cfg["reward_dir"],
                "hidden_dir": collect_cfg["hidden_dir"],
                "shard_name": "ray_shard_000.hdf5",
                "preprocess_config": preprocess_config,
                "data_attrs": {
                    "task_suite_name": collect_cfg["task_suite_name"],
                    "env_name": "libero",
                },
            },
        }

    def _build_oft_components(self, cluster: Cluster) -> dict[str, Any]:
        plan = self.build_oft_worker_plan()
        collect_cfg = plan["collect"]
        num_envs = self._int_from(("env.num_workers", "collect.envs_per_gpu", "num_env_workers"), 1)
        demos_per_shard = self._int_from(("collect.demos_per_shard", "demos_per_shard"), 0)
        task_ids = _resolve_ray_task_ids(
            collect_cfg.get("task_ids", 0),
            num_tasks=collect_cfg.get("num_tasks"),
            suite=str(collect_cfg.get("task_suite_name", "")),
        )
        episodes_per_task = int(collect_cfg.get("episodes_per_task", 1))

        dump_cfg = plan["dump"]
        reward_dir = str(dump_cfg["reward_dir"])
        hidden_dir = str(dump_cfg["hidden_dir"])
        complete_ids = complete_episode_ids_per_task(reward_dir, hidden_dir)
        full_work = _build_resume_work_list(task_ids, episodes_per_task, complete_ids)
        pending_work = list(full_work)
        # Resume-aware shard naming: append at the next free index instead of overwriting
        # ray_shard_000 on a relaunch (the no-Ray collector does the same).
        shard_start = next_shard_index(reward_dir, prefix="ray_shard")
        dump_group = WorkerGroup(
            RolloutDumpWorker,
            reward_dir,
            hidden_dir,
            f"ray_shard_{shard_start:03d}.hdf5",
            dump_cfg["preprocess_config"],
            dump_cfg["data_attrs"],
            demos_per_shard,
            shard_start,
        ).launch(cluster, NodePlacementStrategy(1))
        dump = dump_group.workers[0]

        env_cfg = _ensure_collect_render_device_pool(
            plan["env"],
            cluster,
            collect_cfg=collect_cfg,
        )
        env_kwargs = dict(env_cfg.get("kwargs", {}))
        initial_task = task_ids[0] if task_ids else int(env_kwargs["task_id"])
        env_group = WorkerGroup(
            EnvWorker,
            env_cfg,
            task_id=initial_task,
            replay=dump,
            record_builder=_build_oft_dump_step,
        ).launch(cluster, NodePlacementStrategy(num_envs))
        env_task_ids: list[int | None] = [None] * num_envs
        for env_id in range(num_envs):
            if not pending_work:
                break
            task_id, episode_id = pending_work.pop(0)
            env_task_ids[env_id] = int(task_id)
            env_group.execute_on(env_id).set_task(int(task_id), int(episode_id)).wait()

        gpu_id = int(collect_cfg.get("gpu_id", 0))
        num_infer = max(1, int(collect_cfg.get("num_inference_workers", 1)))
        inference_device = str(plan["inference"].get("device", "cuda")).strip().lower()
        if inference_device.startswith("cuda"):
            # Data-parallel inference across a contiguous GPU range
            # [gpu_id, gpu_id+num_infer-1]. Each worker is built with the full
            # num_envs (per-env state indexed by env_id) but only serves its stable
            # env-id shard (env_id % num_infer), so per-env state stays consistent.
            infer_placement = PackedPlacementStrategy(
                gpu_id,
                gpu_id + num_infer - 1,
                num_gpus_per_worker=1,
            )
        else:
            infer_placement = NodePlacementStrategy(num_infer)
        infer_group = WorkerGroup(
            RolloutInferenceWorker,
            plan["inference"],
            {},
            num_envs=num_envs,
        ).launch(cluster, infer_placement)
        return {
            "dump": dump_group,
            "envs": env_group,
            "infer": infer_group,
            "num_envs": num_envs,
            "num_infer": num_infer,
            "env_task_ids": env_task_ids,
            "task_ids": task_ids,
            "pending_work": pending_work,
            "episodes_per_task": episodes_per_task,
            "target_episodes": len(full_work),
        }

    def _run_loop(self, groups: dict[str, Any]) -> dict[str, float | int]:
        if bool(self._select_first(("rollout.overlap", "overlap"), False)):
            return self._run_loop_overlap(groups)

        envs = groups["envs"]
        infer = groups["infer"]
        dump = groups["dump"]
        num_envs = int(groups["num_envs"])
        num_infer = int(groups.get("num_infer", 1))
        scheduled = "env_task_ids" in groups
        env_task_ids = list(groups.get("env_task_ids", [None] * num_envs))
        pending_work = list(groups.get("pending_work", []))
        env_ids = (
            [idx for idx, task_id in enumerate(env_task_ids) if task_id is not None]
            if scheduled
            else list(range(num_envs))
        )
        target_episodes = int(
            groups.get(
                "target_episodes",
                self._int_from(
                    ("rollout.target_episodes", "target_episodes"), num_envs
                ),
            )
        )
        episode_horizon = self._int_from(
            ("collect.episode_horizon", "episode_horizon"), 300
        )
        max_steps = self._int_from(
            ("rollout.max_steps", "rollout_steps"), target_episodes * episode_horizon
        )

        steps = 0
        driver_roundtrips = 0
        driver_step_calls = 0
        driver_step_waits = 0

        self.console_banner(
            "COLDSTART COLLECT",
            subtitle=f"target={target_episodes} episodes · envs={num_envs}",
        )

        def wait_result(result: Any) -> list[Any]:
            nonlocal driver_roundtrips
            driver_roundtrips += 1
            return result.wait()

        def wait_results(results: list[Any]) -> list[Any]:
            nonlocal driver_roundtrips
            if not results:
                return []
            driver_roundtrips += 1
            return _wait_worker_results(results)

        while env_ids and steps < max_steps:
            done_count = int(wait_result(dump.size())[0])
            self.console_progress(done_count, target_episodes, "collect", unit="ep")
            if done_count >= target_episodes:
                break
            if scheduled:
                obs_batch = wait_results(
                    [envs.execute_on(env_id).current_obs() for env_id in env_ids]
                )
            else:
                obs_batch = wait_result(envs.current_obs())
            # Data-parallel inference: route each env's obs to its stable owner worker
            # (env_id % num_infer) so multi-GPU collection keeps per-env state consistent.
            shards = _shard_env_ids_by_worker(env_ids, num_infer)
            obs_by_env = dict(zip(env_ids, obs_batch, strict=True))
            infer_calls = {
                w: infer.execute_on(w).forward_batch([obs_by_env[e] for e in ids], ids)
                for w, ids in shards.items()
            }
            out_by_env: dict[int, tuple[Any, Any, Any | None]] = {}
            for w, ids in shards.items():
                out = wait_result(infer_calls[w])[0]
                lang_emb = out.get("lang_emb") or [None] * len(ids)
                for e, action, hidden, lang in zip(
                    ids, out["actions"], out["obs_embedding"], lang_emb, strict=True
                ):
                    out_by_env[e] = (action, hidden, lang)
            step_calls = [
                envs.execute_on(env_id).step(
                    out_by_env[env_id][0],
                    out_by_env[env_id][1],
                    out_by_env[env_id][2],
                )
                for env_id in env_ids
            ]
            driver_step_calls += len(step_calls)
            if step_calls:
                driver_step_waits += 1
            step_results = wait_results(step_calls)
            done_envs = []
            for env_id, (_obs, done, _info) in zip(env_ids, step_results, strict=True):
                if done:
                    done_envs.append(env_id)
                    self.console_record_success(bool((_info or {}).get("success", False)))
            if done_envs:
                reset_shards = _shard_env_ids_by_worker(done_envs, num_infer)
                wait_results(
                    [infer.execute_on(w).reset_states(ids) for w, ids in reset_shards.items()]
                )
            if scheduled and done_envs:
                set_task_calls = []
                for env_id in done_envs:
                    reserved = pending_work.pop(0) if pending_work else None
                    next_task = None if reserved is None else reserved[0]
                    env_task_ids[int(env_id)] = next_task
                    if reserved is not None:
                        set_task_calls.append(
                            envs.execute_on(int(env_id)).set_task(
                                int(next_task), int(reserved[1])
                            )
                        )
                wait_results(set_task_calls)
                env_ids = [
                    idx for idx, task_id in enumerate(env_task_ids) if task_id is not None
                ]
            steps += 1

        episodes = int(wait_result(dump.size())[0])
        self.console_progress(episodes, target_episodes, "collect", unit="ep")
        wait_result(dump.close())
        wait_result(envs.close())
        succ_rate = self.console_success_rate()
        self.console_banner(
            "COLDSTART COLLECT",
            done=True,
            subtitle=f"{episodes} episodes · succ {succ_rate:.3f}",
        )
        self.console_metrics(
            "collect",
            {
                "collect/episodes": episodes,
                "collect/steps": int(steps),
                "collect/success_rate": succ_rate,
                "env/num_env_workers": int(num_envs),
            },
            force=True,
        )
        return {
            "rollout/episodes": episodes,
            "rollout/steps": int(steps),
            "env/num_env_workers": int(num_envs),
            "time/driver_roundtrips": int(driver_roundtrips),
            "time/driver_step_calls": int(driver_step_calls),
            "time/driver_step_waits": int(driver_step_waits),
            **collect_resource_metrics(prefix="time"),
        }

    def _run_loop_overlap(self, groups: dict[str, Any]) -> dict[str, float | int]:
        import ray

        envs = groups["envs"]
        infer = groups["infer"]
        dump = groups["dump"]
        num_envs = int(groups["num_envs"])
        if int(groups.get("num_infer", 1)) > 1:
            raise NotImplementedError(
                "multi-GPU inference (collect.num_inference_workers>1) is supported on the "
                "default rollout loop only; set rollout.overlap=false (or "
                "collect.num_inference_workers=1) for the overlap path."
        )
        scheduled = "env_task_ids" in groups
        env_task_ids = list(groups.get("env_task_ids", [None] * num_envs))
        pending_work = list(groups.get("pending_work", []))
        env_ids = (
            [idx for idx, task_id in enumerate(env_task_ids) if task_id is not None]
            if scheduled
            else list(range(num_envs))
        )
        target_episodes = int(
            groups.get(
                "target_episodes",
                self._int_from(("rollout.target_episodes", "target_episodes"), num_envs),
            )
        )
        episode_horizon = self._int_from(
            ("collect.episode_horizon", "episode_horizon"), 300
        )
        max_steps = self._int_from(
            ("rollout.max_steps", "rollout_steps"), target_episodes * episode_horizon
        )

        self.console_banner(
            "COLDSTART COLLECT",
            subtitle=f"target={target_episodes} episodes · envs={num_envs} · overlap",
        )

        pending_steps: dict[Any, tuple[int, Any, float]] = {}
        pending_infers: dict[Any, tuple[list[int], Any, float]] = {}
        ready_obs: list[tuple[int, dict[str, Any]]] = []
        steps = 0
        overlap_events = 0
        stop_launching = False
        infer_wait_s = 0.0
        env_step_wait_s = 0.0
        dump_wait_s = 0.0
        ray_wait_s = 0.0
        env_ready_batches = 0
        infer_ready_batches = 0

        def refresh_stop() -> None:
            nonlocal stop_launching, dump_wait_s
            if stop_launching:
                return
            start = time.perf_counter()
            episodes_so_far = int(dump.size().wait()[0])
            dump_wait_s += time.perf_counter() - start
            self.console_progress(episodes_so_far, target_episodes, "collect", unit="ep")
            if episodes_so_far >= target_episodes:
                stop_launching = True
                ready_obs.clear()

        def launch_infer() -> None:
            nonlocal steps, overlap_events
            if stop_launching or pending_infers or not ready_obs or steps >= max_steps:
                return
            batch = list(ready_obs)
            ready_obs.clear()
            batch_env_ids = [env_id for env_id, _obs in batch]
            obs_batch = [obs for _env_id, obs in batch]
            if pending_steps or steps > 0:
                overlap_events += 1
            result = infer.forward_batch(obs_batch, batch_env_ids)
            pending_infers[result.refs[0]] = (batch_env_ids, result, time.perf_counter())
            steps += 1

        def launch_steps(batch_env_ids: list[int], infer_out: dict[str, Any]) -> None:
            lang_emb = infer_out.get("lang_emb") or [None] * len(batch_env_ids)
            for env_id, action, hidden, lang in zip(
                batch_env_ids,
                infer_out["actions"],
                infer_out["obs_embedding"],
                lang_emb,
                strict=True,
            ):
                result = envs.execute_on(int(env_id)).step(action, hidden, lang)
                pending_steps[result.refs[0]] = (int(env_id), result, time.perf_counter())

        def handle_step_result(
            env_id: int,
            step_result: tuple[dict[str, Any], bool, dict[str, Any]],
        ) -> None:
            next_obs, done, _info = step_result
            if done:
                self.console_record_success(bool((_info or {}).get("success", False)))
                infer.reset_states([int(env_id)]).wait()
                if scheduled:
                    reserved = pending_work.pop(0) if pending_work else None
                    next_task = None if reserved is None else reserved[0]
                    env_task_ids[int(env_id)] = next_task
                    if reserved is None:
                        return
                    next_obs = (
                        envs.execute_on(int(env_id))
                        .set_task(int(next_task), int(reserved[1]))
                        .wait()[0]
                    )
            if not stop_launching:
                ready_obs.append((int(env_id), next_obs))

        if scheduled:
            initial_obs = _wait_worker_results(
                [envs.execute_on(env_id).current_obs() for env_id in env_ids]
            )
        else:
            initial_obs = envs.current_obs().wait()
        ready_obs.extend(
            (int(env_id), obs)
            for env_id, obs in zip(env_ids, initial_obs, strict=True)
        )
        launch_infer()

        while pending_steps or pending_infers or ready_obs:
            refresh_stop()
            launch_infer()
            refs = list(pending_infers) + list(pending_steps)
            if not refs:
                if stop_launching or steps >= max_steps:
                    break
                launch_infer()
                refs = list(pending_infers) + list(pending_steps)
                if not refs:
                    break

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
                    if not stop_launching:
                        launch_steps(batch_env_ids, infer_out)
                elif ref in pending_steps:
                    env_id, result, start_time = pending_steps.pop(ref)
                    step_result = result.wait()[0]
                    env_step_wait_s += time.perf_counter() - start_time
                    env_ready_batches += 1
                    handle_step_result(env_id, step_result)

            if steps >= max_steps:
                stop_launching = True
                ready_obs.clear()

        for _env_id, result, start_time in list(pending_steps.values()):
            result.wait()
            env_step_wait_s += time.perf_counter() - start_time
        pending_steps.clear()
        for _env_ids, result, start_time in list(pending_infers.values()):
            result.wait()
            infer_wait_s += time.perf_counter() - start_time
        pending_infers.clear()

        start = time.perf_counter()
        episodes = int(dump.size().wait()[0])
        dump.close().wait()
        envs.close().wait()
        dump_wait_s += time.perf_counter() - start
        self.console_progress(episodes, target_episodes, "collect", unit="ep")
        succ_rate = self.console_success_rate()
        self.console_banner(
            "COLDSTART COLLECT",
            done=True,
            subtitle=f"{episodes} episodes · succ {succ_rate:.3f}",
        )
        self.console_metrics(
            "collect",
            {
                "collect/episodes": episodes,
                "collect/steps": int(steps),
                "collect/success_rate": succ_rate,
                "env/num_env_workers": int(num_envs),
            },
            force=True,
        )
        return {
            "rollout/episodes": episodes,
            "rollout/steps": int(steps),
            "env/num_env_workers": int(num_envs),
            "time/overlap_events": int(overlap_events),
            "time/infer_wait_s": float(infer_wait_s),
            "time/env_step_wait_s": float(env_step_wait_s),
            "time/dump_wait_s": float(dump_wait_s),
            "time/ray_wait_s": float(ray_wait_s),
            "time/env_ready_batches": int(env_ready_batches),
            "time/infer_ready_batches": int(infer_ready_batches),
            **collect_resource_metrics(prefix="time"),
        }

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
        "history": 1,
        "include_state": False,
        "hidden_key": "obs_embedding",
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


def _wait_worker_results(results: list[Any]) -> list[Any]:
    """Collect multiple WorkerGroupFuncResult objects with one driver wait."""
    import ray

    refs: list[Any] = []
    for result in results:
        refs.extend(list(getattr(result, "refs", [])))
    if not refs:
        return []
    return list(ray.get(refs))


def _first_task_id(task_ids: Any) -> int:
    if isinstance(task_ids, (list, tuple)):
        return int(task_ids[0]) if task_ids else 0
    if str(task_ids).strip().lower() == "all":
        return 0
    return int(task_ids)


def _resolve_ray_task_ids(
    task_ids: Any,
    *,
    num_tasks: Any | None,
    suite: str,
) -> list[int]:
    if isinstance(task_ids, (list, tuple)):
        return [int(task_id) for task_id in task_ids]
    if isinstance(task_ids, str):
        value = task_ids.strip()
        if value.lower() == "all":
            n_tasks = int(num_tasks) if num_tasks is not None else _default_num_tasks(suite)
            return list(range(n_tasks))
        if "," in value:
            return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(task_ids)]


def _default_num_tasks(suite: str) -> int:
    # LIBERO goal/object/spatial/10 each expose ten task ids. Keep this as a
    # fallback for the Ray driver, where querying a real env just to expand
    # "all" would eagerly initialize robosuite before actor placement.
    if str(suite).startswith("libero"):
        return 10
    return 1


def _next_ray_task_id(
    task_ids: list[int],
    task_counts: dict[int, int],
    episodes_per_task: int,
) -> tuple[int, int] | None:
    """Reserve the next task below target; return ``(task_id, scheduled_index)``.

    ``scheduled_index`` is this pass's 0-based episode index for the task (the count
    before the increment). The caller adds the on-disk resume offset to turn it into the
    episode_id / init_state, so every reserved episode gets a DISTINCT init_state.
    """
    candidates = [
        (int(task_counts.get(int(task_id), 0)), index, int(task_id))
        for index, task_id in enumerate(task_ids)
        if int(task_counts.get(int(task_id), 0)) < int(episodes_per_task)
    ]
    if not candidates:
        return None
    current, _index, task_id = min(candidates)
    task_counts[task_id] = current + 1
    return task_id, current


def _ray_start_episode_id(
    offset_per_task: dict[int, int], task_id: int, scheduled_index: int
) -> int:
    """init_state / episode_id for a reserved episode = resume offset + this-pass index."""
    return int(offset_per_task.get(int(task_id), 0)) + int(scheduled_index)


def _build_oft_dump_step(
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
