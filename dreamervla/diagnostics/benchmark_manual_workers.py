"""Benchmark manual-cotrain workers with production-shaped messages.

Example:
  CUDA_VISIBLE_DEVICES=1 python -m dreamervla.diagnostics.benchmark_manual_workers \
    --component wm-env --profile config \
    --output-json logs/debug_logs/wm_env_worker_bench.json \
    experiment=openvla_onetraj_libero_cotrain \
    task=openvla_onetraj_coldstart_libero
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

import dreamervla.workers.env.trajectory_env_worker as trajectory_env_worker
from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.utils.paths import PROJECT_ROOT
from dreamervla.workers.cotrain.messages import (
    ObservationMsg,
    RolloutResultBatchMsg,
    RolloutResultMsg,
    rollout_result_batch_to_messages,
)
from dreamervla.workers.env.trajectory_env_worker import WMEnvWorker
from dreamervla.workers.rollout.multistep_rollout_worker import MultiStepRolloutWorker


def build_synthetic_observation(
    *,
    env_rank: int,
    slot_id: int,
    step: int,
    latent_dim: int,
    lang_dim: int = 0,
    proprio_dim: int = 0,
    task_id: int = 0,
    episode_id: int | None = None,
) -> ObservationMsg:
    """Create an ObservationMsg shaped like manual cotrain WMEnv output."""

    value = float(step + slot_id)
    obs: dict[str, Any] = {
        "obs_embedding": np.full((int(latent_dim),), value, dtype=np.float32),
        "state": np.full((max(1, int(proprio_dim)),), value, dtype=np.float32),
        "task_id": int(task_id),
        "episode_id": int(slot_id if episode_id is None else episode_id),
        "step": int(step),
        "task_description": f"task {int(task_id)}",
        "is_first": int(step) == 0,
    }
    if int(lang_dim) > 0:
        obs["lang_emb"] = np.full((int(lang_dim),), value + 0.5, dtype=np.float32)
    if int(proprio_dim) > 0:
        obs["proprio"] = np.full((int(proprio_dim),), value + 0.25, dtype=np.float32)
    return ObservationMsg(
        env_rank=int(env_rank),
        slot_id=int(slot_id),
        task_id=int(task_id),
        episode_id=int(slot_id if episode_id is None else episode_id),
        step=int(step),
        obs=obs,
        versions={"policy": 0},
    )


def build_rollout_result(
    obs_msg: ObservationMsg,
    *,
    action_dim: int,
    chunk_size: int,
    policy_version: int = 0,
) -> RolloutResultMsg:
    """Create a RolloutResultMsg matching RolloutWorker.generate_once output."""

    if "obs_embedding" in obs_msg.obs:
        hidden_source = obs_msg.obs["obs_embedding"]
    elif "latent" in obs_msg.obs:
        hidden_source = obs_msg.obs["latent"]
    else:
        raise ValueError("ObservationMsg.obs must include obs_embedding or latent")
    hidden = np.asarray(hidden_source, dtype=np.float32).reshape(1, -1)
    action = np.zeros((int(chunk_size), int(action_dim)), dtype=np.float32)
    action[:, 0] = float(obs_msg.slot_id)
    forward_inputs: dict[str, Any] = {
        "hidden": hidden,
        "action": action.reshape(1, int(chunk_size), int(action_dim)),
    }
    if "lang_emb" in obs_msg.obs:
        forward_inputs["lang_emb"] = np.asarray(obs_msg.obs["lang_emb"], dtype=np.float32)
    return RolloutResultMsg(
        env_rank=int(obs_msg.env_rank),
        slot_id=int(obs_msg.slot_id),
        task_id=int(obs_msg.task_id),
        episode_id=int(obs_msg.episode_id),
        step=int(obs_msg.step),
        actions=action,
        prev_logprobs=np.zeros((1,), dtype=np.float32),
        prev_values=None,
        forward_inputs=forward_inputs,
        versions={"policy": int(policy_version)},
    )


class SyntheticBatchWMEnv:
    """Small batched WMEnv contract implementation for diagnostics and tests."""

    def __init__(
        self,
        num_envs: int = 8,
        latent_dim: int = 4,
        action_dim: int = 7,
        lang_dim: int = 0,
        proprio_dim: int = 0,
        horizon: int = 512,
        device: str = "cpu",
        matmul_dim: int = 0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.lang_dim = int(lang_dim)
        self.proprio_dim = int(proprio_dim)
        self.horizon = int(horizon)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.matmul_dim = int(matmul_dim)
        self.task_ids = [0 for _ in range(self.num_envs)]
        self.episode_ids = [idx for idx in range(self.num_envs)]
        self.steps = [0 for _ in range(self.num_envs)]
        self.batch_calls: list[tuple[list[int], tuple[int, ...]]] = []
        self.forward_time_s = 0.0
        self._work_tensor: torch.Tensor | None = None
        if self.matmul_dim > 0:
            self._work_tensor = torch.randn(
                self.matmul_dim,
                self.matmul_dim,
                device=self.device,
                dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )

    def reset_slot(
        self,
        slot_id: int,
        *,
        task_id: int = 0,
        episode_id: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        slot = int(slot_id)
        self.task_ids[slot] = int(task_id)
        self.episode_ids[slot] = int(episode_id)
        self.steps[slot] = 0
        return self._obs(slot, is_first=True), {"episode_id": int(episode_id)}

    def step_batch(
        self,
        actions: Any,
        env_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[list[dict[str, Any]], list[float], list[bool], list[bool], list[dict[str, Any]]]:
        slots = [int(v) for v in (range(self.num_envs) if env_ids is None else env_ids)]
        action_arr = np.asarray(actions, dtype=np.float32)
        self.batch_calls.append((slots, tuple(int(v) for v in action_arr.shape)))
        start = time.perf_counter()
        if self._work_tensor is not None:
            work = self._work_tensor
            _ = work @ work.T
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        self.forward_time_s += time.perf_counter() - start

        obs, rewards, terms, truncs, infos = [], [], [], [], []
        for index, slot in enumerate(slots):
            self.steps[slot] += 1
            done = self.steps[slot] >= self.horizon
            obs.append(self._obs(slot, is_first=False))
            rewards.append(float(done))
            terms.append(bool(done))
            truncs.append(False)
            infos.append(
                {
                    "success": bool(done),
                    "wm_action": action_arr[index].reshape(-1)[: self.action_dim],
                }
            )
        return obs, rewards, terms, truncs, infos

    def step_slot(
        self, slot_id: int, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, rewards, terms, truncs, infos = self.step_batch([action], env_ids=[int(slot_id)])
        return obs[0], rewards[0], terms[0], truncs[0], infos[0]

    def chunk_step_batch(
        self,
        actions: Any,
        env_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        slots = [int(v) for v in (range(self.num_envs) if env_ids is None else env_ids)]
        action_arr = np.asarray(actions, dtype=np.float32).reshape(
            len(slots),
            -1,
            self.action_dim,
        )
        chunk_len = int(action_arr.shape[1])
        self.batch_calls.append((slots, tuple(int(v) for v in action_arr.shape)))
        start = time.perf_counter()
        if self._work_tensor is not None:
            work = self._work_tensor
            _ = work @ work.T
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        self.forward_time_s += time.perf_counter() - start

        rewards = np.zeros((len(slots), chunk_len), dtype=np.float32)
        terminations = np.zeros((len(slots), chunk_len), dtype=np.bool_)
        truncations = np.zeros((len(slots), chunk_len), dtype=np.bool_)
        obs, infos = [], []
        offsets = np.arange(1, chunk_len + 1, dtype=np.int64)
        for index, slot in enumerate(slots):
            start_step = int(self.steps[slot])
            elapsed = start_step + offsets
            done_steps = elapsed >= self.horizon
            rewards[index] = done_steps.astype(np.float32)
            terminations[index] = done_steps
            self.steps[slot] = int(start_step + chunk_len)
            obs.append(self._obs(slot, is_first=False))
            infos.append(
                {
                    "success": bool(done_steps.any()),
                    "wm_action": action_arr[index, -1].reshape(-1)[: self.action_dim],
                }
            )
        return obs, rewards, terminations, truncations, infos

    def make_transition(
        self,
        obs: dict[str, Any],
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        done = bool(terminated or truncated)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[: self.action_dim]
        return {
            "state": np.asarray(obs["state"], dtype=np.float32),
            "obs_embedding": np.asarray(obs["obs_embedding"], dtype=np.float32),
            "action": action_arr,
            "wm_action": np.asarray((info or {}).get("wm_action", action_arr), dtype=np.float32),
            "policy_action": action_arr,
            "reward": np.float32(reward),
            "done": np.float32(done),
            "discount": np.float32(0.0 if terminated else 1.0),
            "is_first": bool(obs.get("is_first", False)),
            "is_terminal": bool(terminated),
            "is_last": bool(done),
            "task_id": int(obs["task_id"]),
            "episode_id": int(obs["episode_id"]),
            "step": int(obs["step"]),
            "task_description": str(obs["task_description"]),
            "success": bool((info or {}).get("success", False)),
        }

    def get_metrics(self, *, reset: bool = False) -> dict[str, float]:
        batch_sizes = [float(len(slots)) for slots, _shape in self.batch_calls]
        batch_size_sum = float(sum(batch_sizes))
        metrics = {
            "model_forwards": float(len(self.batch_calls)),
            "wm_forward_calls": float(len(self.batch_calls)),
            "classifier_forward_calls": 0.0,
            "wm_forward_time_s": float(self.forward_time_s),
            "batch_size_sum": batch_size_sum,
            "batch_size_avg": float(batch_size_sum / max(1, len(self.batch_calls))),
            "batch_size_min": float(min(batch_sizes, default=0.0)),
            "batch_size_max": float(max(batch_sizes, default=0.0)),
        }
        if reset:
            self.batch_calls.clear()
            self.forward_time_s = 0.0
        return metrics

    def close(self) -> None:
        self._work_tensor = None

    def _obs(self, slot_id: int, *, is_first: bool) -> dict[str, Any]:
        value = float(self.steps[slot_id] + self.episode_ids[slot_id] * 10)
        obs = {
            "state": np.full((max(1, self.proprio_dim),), value, dtype=np.float32),
            "obs_embedding": np.full((self.latent_dim,), value, dtype=np.float32),
            "task_id": int(self.task_ids[slot_id]),
            "episode_id": int(self.episode_ids[slot_id]),
            "step": int(self.steps[slot_id]),
            "task_description": f"task {int(self.task_ids[slot_id])}",
            "is_first": bool(is_first),
        }
        if self.lang_dim > 0:
            obs["lang_emb"] = np.full((self.lang_dim,), value + 0.5, dtype=np.float32)
        if self.proprio_dim > 0:
            obs["proprio"] = np.full((self.proprio_dim,), value + 0.25, dtype=np.float32)
        return obs


class GpuSampler:
    """Best-effort nvidia-smi sampler for long idle gaps and burst detection."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = float(interval_s)
        self.samples: list[dict[str, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuSampler:
        if shutil.which("nvidia-smi") is None:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any]:
        by_gpu: dict[int, list[dict[str, int]]] = {}
        for sample in self.samples:
            by_gpu.setdefault(int(sample["index"]), []).append(sample)
        metrics: dict[str, Any] = {"gpu/sample_count": len(self.samples)}
        for index, samples in sorted(by_gpu.items()):
            utils = [int(sample["util_gpu"]) for sample in samples]
            mems = [int(sample["memory_used_mb"]) for sample in samples]
            zero_run = _longest_zero_run(utils)
            metrics[f"gpu/{index}/util_avg"] = float(sum(utils) / max(1, len(utils)))
            metrics[f"gpu/{index}/util_max"] = int(max(utils, default=0))
            metrics[f"gpu/{index}/util_nonzero_samples"] = int(sum(v > 0 for v in utils))
            metrics[f"gpu/{index}/util_zero_run_max_samples"] = int(zero_run)
            metrics[f"gpu/{index}/util_zero_run_max_s"] = float(zero_run * self.interval_s)
            metrics[f"gpu/{index}/memory_used_mb_max"] = int(max(mems, default=0))
        return metrics

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.extend(_query_gpu_once())
            self._stop.wait(self.interval_s)


class _ReadyPut:
    def wait(self) -> None:
        return None


class _MemoryChannel:
    def __init__(self, initial: list[Any] | None = None) -> None:
        self.queue = list(initial or [])
        self.puts: list[tuple[str, Any]] = []
        self.put_no_wait_calls: list[tuple[str, Any]] = []
        self.gets: list[str] = []

    def put(self, item: Any, *, key: str = "default") -> None:
        self.puts.append((str(key), item))

    def put_no_wait(self, item: Any, *, key: str = "default") -> _ReadyPut:
        self.put_no_wait_calls.append((str(key), item))
        self.put(item, key=key)
        return _ReadyPut()

    def get(self, *, key: str = "default") -> Any:
        self.gets.append(str(key))
        if not self.queue:
            raise RuntimeError(f"channel {key!r} is empty")
        return self.queue.pop(0)


def run_wm_env_direct_benchmark(
    *,
    env_cfg: Mapping[str, Any],
    num_slots: int,
    chunk_steps: int,
    action_dim: int,
    chunk_size: int,
    latent_dim: int,
    lang_dim: int = 0,
    proprio_dim: int = 0,
    output_json: str | Path | None = None,
    gpu_sample_interval_s: float = 0.5,
) -> dict[str, Any]:
    """Run WMEnvWorker's batched apply path without Ray scheduling noise."""

    start = time.perf_counter()
    worker = WMEnvWorker(
        env_cfg=dict(env_cfg),
        num_slots=int(num_slots),
        rollout_epoch=1,
        max_steps_per_rollout_epoch=int(chunk_steps),
        num_action_chunks=int(chunk_size),
        task_id=0,
    )
    worker.init()
    init_s = time.perf_counter() - start
    shards = []
    try:
        obs_msgs = worker.bootstrap_obs()
        with GpuSampler(interval_s=gpu_sample_interval_s) as sampler:
            loop_start = time.perf_counter()
            for step in range(int(chunk_steps)):
                results = [
                    build_rollout_result(
                        obs_msg,
                        action_dim=int(action_dim),
                        chunk_size=int(chunk_size),
                    )
                    for obs_msg in obs_msgs
                ]
                shards.extend(worker._apply_wm_rollout_results_batch(results))
                obs_msgs = [
                    ObservationMsg(
                        env_rank=int(worker.rank),
                        slot_id=slot_id,
                        task_id=int(worker._obs_by_slot[slot_id].get("task_id", 0)),
                        episode_id=int(worker._obs_by_slot[slot_id].get("episode_id", slot_id)),
                        step=int(worker._obs_by_slot[slot_id].get("step", step + 1)),
                        obs=dict(worker._obs_by_slot[slot_id]),
                        versions=dict(worker._model_versions),
                    )
                    for slot_id in range(int(num_slots))
                ]
            loop_s = time.perf_counter() - loop_start
        env_metrics = _env_metrics(worker)
        metrics: dict[str, Any] = {
            "worker/component": "wm-env-direct",
            "worker/env_class": type(worker.envs[0]).__name__ if worker.envs else "",
            "worker/init_s": float(init_s),
            "worker/loop_s": float(loop_s),
            "worker/chunk_steps": int(chunk_steps),
            "worker/slot_count": int(num_slots),
            "worker/action_chunk_size": int(chunk_size),
            "worker/action_dim": int(action_dim),
            "worker/trajectory_shards": int(len(shards)),
            "worker/chunk_steps_per_s": float(int(chunk_steps) / max(loop_s, 1e-9)),
            **{f"env/wm_env/{k}": v for k, v in env_metrics.items()},
            **sampler.summary(),
        }
    finally:
        worker.close()
    _write_json_if_requested(metrics, output_json)
    return metrics


def run_wm_env_interact_benchmark(
    *,
    env_cfg: Mapping[str, Any],
    num_slots: int,
    chunk_steps: int,
    action_dim: int,
    chunk_size: int,
    latent_dim: int,
    lang_dim: int = 0,
    proprio_dim: int = 0,
    output_json: str | Path | None = None,
    gpu_sample_interval_s: float = 0.5,
) -> dict[str, Any]:
    """Run WMEnvWorker.interact with in-memory channels and synthetic rollout batches."""

    rank = 0
    rollout_batches = []
    for step in range(int(chunk_steps)):
        rollout_batches.append(
            RolloutResultBatchMsg(
                env_rank=rank,
                results=[
                    build_rollout_result(
                        build_synthetic_observation(
                            env_rank=rank,
                            slot_id=slot_id,
                            step=step,
                            latent_dim=int(latent_dim),
                            lang_dim=int(lang_dim),
                            proprio_dim=int(proprio_dim),
                        ),
                        action_dim=int(action_dim),
                        chunk_size=int(chunk_size),
                    )
                    for slot_id in range(int(num_slots))
                ],
            )
        )
    channels = {
        "env": _MemoryChannel(),
        "rollout": _MemoryChannel(rollout_batches),
        "actor": _MemoryChannel(),
    }
    worker = WMEnvWorker(
        env_cfg=dict(env_cfg),
        num_slots=int(num_slots),
        rollout_epoch=1,
        max_steps_per_rollout_epoch=int(chunk_steps) * int(chunk_size),
        num_action_chunks=int(chunk_size),
        task_id=0,
        request_final_bootstrap=False,
    )
    start = time.perf_counter()
    worker.init()
    init_s = time.perf_counter() - start
    original_connect = trajectory_env_worker.Channel.connect
    try:
        trajectory_env_worker.Channel.connect = staticmethod(lambda name: channels[str(name)])
        with GpuSampler(interval_s=gpu_sample_interval_s) as sampler:
            loop_start = time.perf_counter()
            interact_metrics = worker.interact("env", "rollout", "actor")
            loop_s = time.perf_counter() - loop_start
        metrics: dict[str, Any] = {
            "worker/component": "wm-env-interact",
            "worker/env_class": type(worker.envs[0]).__name__ if worker.envs else "",
            "worker/init_s": float(init_s),
            "worker/loop_s": float(loop_s),
            "worker/chunk_steps": int(chunk_steps),
            "worker/slot_count": int(num_slots),
            "worker/action_chunk_size": int(chunk_size),
            "worker/action_dim": int(action_dim),
            "worker/trajectory_shards": int(len(channels["actor"].puts)),
            "worker/chunk_steps_per_s": float(int(chunk_steps) / max(loop_s, 1e-9)),
            **interact_metrics,
            **sampler.summary(),
        }
    finally:
        trajectory_env_worker.Channel.connect = original_connect
        worker.close()
    _write_json_if_requested(metrics, output_json)
    return metrics


def run_rollout_direct_benchmark(
    *,
    policy_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    encoder_cfg: Mapping[str, Any] | None,
    num_slots: int,
    chunk_steps: int,
    latent_dim: int,
    lang_dim: int = 0,
    proprio_dim: int = 0,
    output_json: str | Path | None = None,
    gpu_sample_interval_s: float = 0.5,
) -> dict[str, Any]:
    """Run RolloutWorker.generate_once with WMEnv-shaped observations."""

    worker = MultiStepRolloutWorker(
        policy_cfg=dict(policy_cfg),
        encoder_cfg=dict(encoder_cfg) if encoder_cfg else None,
        init_ckpt={},
        train_cfg=dict(train_cfg),
    )
    start = time.perf_counter()
    worker.init()
    init_s = time.perf_counter() - start
    generated = 0
    try:
        with GpuSampler(interval_s=gpu_sample_interval_s) as sampler:
            loop_start = time.perf_counter()
            for step in range(int(chunk_steps)):
                for slot_id in range(int(num_slots)):
                    obs = build_synthetic_observation(
                        env_rank=int(worker.rank),
                        slot_id=slot_id,
                        step=step,
                        latent_dim=int(latent_dim),
                        lang_dim=int(lang_dim),
                        proprio_dim=int(proprio_dim),
                    )
                    worker.generate_once(obs)
                    generated += 1
            loop_s = time.perf_counter() - loop_start
        metrics: dict[str, Any] = {
            "worker/component": "rollout-direct",
            "worker/init_s": float(init_s),
            "worker/loop_s": float(loop_s),
            "worker/generated": int(generated),
            "worker/slot_count": int(num_slots),
            "worker/chunk_steps": int(chunk_steps),
            "worker/generated_per_s": float(generated / max(loop_s, 1e-9)),
            **sampler.summary(),
        }
    finally:
        if hasattr(worker.policy, "cpu"):
            worker.policy.cpu()
    _write_json_if_requested(metrics, output_json)
    return metrics


def run_pair_direct_benchmark(
    *,
    env_cfg: Mapping[str, Any],
    policy_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    encoder_cfg: Mapping[str, Any] | None,
    num_slots: int,
    chunk_steps: int,
    action_dim: int,
    chunk_size: int,
    output_json: str | Path | None = None,
    gpu_sample_interval_s: float = 0.5,
) -> dict[str, Any]:
    """Run RolloutWorker serial inference feeding WMEnvWorker batched stepping."""

    rollout = MultiStepRolloutWorker(
        policy_cfg=dict(policy_cfg),
        encoder_cfg=dict(encoder_cfg) if encoder_cfg else None,
        init_ckpt={},
        train_cfg=dict(train_cfg),
    )
    wm_env = WMEnvWorker(
        env_cfg=dict(env_cfg),
        num_slots=int(num_slots),
        rollout_epoch=1,
        max_steps_per_rollout_epoch=int(chunk_steps),
        num_action_chunks=int(chunk_size),
        task_id=0,
    )
    start = time.perf_counter()
    rollout.init()
    rollout_init_s = time.perf_counter() - start
    start = time.perf_counter()
    wm_env.init()
    wm_init_s = time.perf_counter() - start
    generated = 0
    shards = []
    try:
        obs_msgs = wm_env.bootstrap_obs()
        with GpuSampler(interval_s=gpu_sample_interval_s) as sampler:
            loop_start = time.perf_counter()
            rollout_s = 0.0
            wm_s = 0.0
            for _step in range(int(chunk_steps)):
                obs_batch = wm_env._observation_batch_msg(obs_msgs)
                t0 = time.perf_counter()
                step_result_batch = rollout.generate_result_batch(
                    obs_batch.observations,
                    batched_obs=obs_batch.batched_obs,
                )
                rollout_s += time.perf_counter() - t0
                generated += len(step_result_batch.slot_ids or step_result_batch.results)
                step_results = rollout_result_batch_to_messages(step_result_batch)
                t0 = time.perf_counter()
                shards.extend(wm_env._apply_wm_rollout_results_batch(step_results))
                wm_s += time.perf_counter() - t0
                obs_msgs = [
                    ObservationMsg(
                        env_rank=int(wm_env.rank),
                        slot_id=slot_id,
                        task_id=int(wm_env._obs_by_slot[slot_id].get("task_id", 0)),
                        episode_id=int(wm_env._obs_by_slot[slot_id].get("episode_id", slot_id)),
                        step=int(wm_env._obs_by_slot[slot_id].get("step", 0)),
                        obs=dict(wm_env._obs_by_slot[slot_id]),
                        versions=dict(wm_env._model_versions),
                    )
                    for slot_id in range(int(num_slots))
                ]
            loop_s = time.perf_counter() - loop_start
        metrics: dict[str, Any] = {
            "worker/component": "rollout-wm-env-pair-direct",
            "worker/rollout_init_s": float(rollout_init_s),
            "worker/wm_env_init_s": float(wm_init_s),
            "worker/loop_s": float(loop_s),
            "worker/rollout_generate_s": float(rollout_s),
            "worker/wm_env_apply_s": float(wm_s),
            "worker/generated": int(generated),
            "worker/trajectory_shards": int(len(shards)),
            "worker/slot_count": int(num_slots),
            "worker/chunk_steps": int(chunk_steps),
            "worker/serial_rollout_fraction": float(rollout_s / max(loop_s, 1e-9)),
            "worker/wm_env_fraction": float(wm_s / max(loop_s, 1e-9)),
            **{f"env/wm_env/{k}": v for k, v in _env_metrics(wm_env).items()},
            **sampler.summary(),
        }
    finally:
        wm_env.close()
        if hasattr(rollout.policy, "cpu"):
            rollout.policy.cpu()
    _write_json_if_requested(metrics, output_json)
    return metrics


def main(argv: list[str] | None = None) -> None:
    args, overrides = _parse_args(list(sys.argv[1:] if argv is None else argv))
    cfg = _compose_cfg(args, overrides) if args.profile == "config" else None
    dims = _resolve_dims(args, cfg)
    if args.component == "all":
        metrics = _run_all_components(args, cfg, dims)
        _write_json_if_requested(metrics, args.output_json)
    elif args.component == "wm-env":
        metrics = run_wm_env_direct_benchmark(
            env_cfg=_env_cfg(args, cfg, dims),
            num_slots=args.num_slots,
            chunk_steps=args.chunk_steps,
            action_dim=dims["action_dim"],
            chunk_size=dims["chunk_size"],
            latent_dim=dims["latent_dim"],
            lang_dim=dims["lang_dim"],
            proprio_dim=dims["proprio_dim"],
            output_json=args.output_json,
            gpu_sample_interval_s=args.gpu_sample_interval_s,
        )
    elif args.component == "wm-env-interact":
        metrics = run_wm_env_interact_benchmark(
            env_cfg=_env_cfg(args, cfg, dims),
            num_slots=args.num_slots,
            chunk_steps=args.chunk_steps,
            action_dim=dims["action_dim"],
            chunk_size=dims["chunk_size"],
            latent_dim=dims["latent_dim"],
            lang_dim=dims["lang_dim"],
            proprio_dim=dims["proprio_dim"],
            output_json=args.output_json,
            gpu_sample_interval_s=args.gpu_sample_interval_s,
        )
    elif args.component == "rollout":
        policy_cfg, train_cfg, encoder_cfg = _rollout_cfg(args, cfg, dims)
        metrics = run_rollout_direct_benchmark(
            policy_cfg=policy_cfg,
            train_cfg=train_cfg,
            encoder_cfg=encoder_cfg,
            num_slots=args.num_slots,
            chunk_steps=args.chunk_steps,
            latent_dim=dims["latent_dim"],
            lang_dim=dims["lang_dim"],
            proprio_dim=dims["proprio_dim"],
            output_json=args.output_json,
            gpu_sample_interval_s=args.gpu_sample_interval_s,
        )
    else:
        policy_cfg, train_cfg, encoder_cfg = _rollout_cfg(args, cfg, dims)
        metrics = run_pair_direct_benchmark(
            env_cfg=_env_cfg(args, cfg, dims),
            policy_cfg=policy_cfg,
            train_cfg=train_cfg,
            encoder_cfg=encoder_cfg,
            num_slots=args.num_slots,
            chunk_steps=args.chunk_steps,
            action_dim=dims["action_dim"],
            chunk_size=dims["chunk_size"],
            output_json=args.output_json,
            gpu_sample_interval_s=args.gpu_sample_interval_s,
        )
    print(json.dumps(_jsonable(metrics), indent=2, sort_keys=True))


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("wm-env", "wm-env-interact", "rollout", "pair", "all"),
        required=True,
    )
    parser.add_argument("--profile", choices=("tiny", "config"), default="tiny")
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--chunk-steps", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--lang-dim", type=int, default=0)
    parser.add_argument("--proprio-dim", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--synthetic-matmul-dim", type=int, default=0)
    parser.add_argument("--gpu-sample-interval-s", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--no-rollout-encoder", action="store_true")
    args, overrides = parser.parse_known_args(argv)
    return args, overrides


def _run_all_components(
    args: argparse.Namespace,
    cfg: DictConfig | None,
    dims: Mapping[str, int],
) -> dict[str, Any]:
    policy_cfg, train_cfg, encoder_cfg = _rollout_cfg(args, cfg, dims)
    return {
        "worker/component": "all",
        "wm-env": run_wm_env_direct_benchmark(
            env_cfg=_env_cfg(args, cfg, dims),
            num_slots=args.num_slots,
            chunk_steps=args.chunk_steps,
            action_dim=dims["action_dim"],
            chunk_size=dims["chunk_size"],
            latent_dim=dims["latent_dim"],
            lang_dim=dims["lang_dim"],
            proprio_dim=dims["proprio_dim"],
            output_json=None,
            gpu_sample_interval_s=args.gpu_sample_interval_s,
        ),
        "rollout": run_rollout_direct_benchmark(
            policy_cfg=policy_cfg,
            train_cfg=train_cfg,
            encoder_cfg=encoder_cfg,
            num_slots=args.num_slots,
            chunk_steps=args.chunk_steps,
            latent_dim=dims["latent_dim"],
            lang_dim=dims["lang_dim"],
            proprio_dim=dims["proprio_dim"],
            output_json=None,
            gpu_sample_interval_s=args.gpu_sample_interval_s,
        ),
        "pair": run_pair_direct_benchmark(
            env_cfg=_env_cfg(args, cfg, dims),
            policy_cfg=policy_cfg,
            train_cfg=train_cfg,
            encoder_cfg=encoder_cfg,
            num_slots=args.num_slots,
            chunk_steps=args.chunk_steps,
            action_dim=dims["action_dim"],
            chunk_size=dims["chunk_size"],
            output_json=None,
            gpu_sample_interval_s=args.gpu_sample_interval_s,
        ),
    }


def _compose_cfg(args: argparse.Namespace, overrides: list[str]) -> DictConfig:
    register_dreamervla_resolvers()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def _resolve_dims(args: argparse.Namespace, cfg: DictConfig | None) -> dict[str, int]:
    return {
        "chunk_size": int(
            _select(cfg, "task.openvla_oft.hidden_token.chunk_size", args.chunk_size)
        ),
        "action_dim": int(_select(cfg, "task.action_dim", args.action_dim)),
        "latent_dim": int(
            _select(cfg, "task.openvla_oft.hidden_token.wm_obs_dim", args.latent_dim)
        ),
        "lang_dim": int(_select(cfg, "task.openvla_oft.hidden_token.lang_dim", args.lang_dim)),
        "proprio_dim": int(
            _select(cfg, "task.openvla_oft.hidden_token.proprio_dim", args.proprio_dim)
        ),
    }


def _env_cfg(
    args: argparse.Namespace,
    cfg: DictConfig | None,
    dims: Mapping[str, int],
) -> dict[str, Any]:
    if cfg is not None:
        env_cfg = OmegaConf.to_container(OmegaConf.select(cfg, "env.wm.cfg"), resolve=True)
        if not isinstance(env_cfg, Mapping):
            raise TypeError("env.wm.cfg must resolve to a mapping")
        env_cfg = dict(env_cfg)
        kwargs = dict(env_cfg.get("kwargs", {}))
        kwargs["num_envs"] = int(args.num_slots)
        kwargs["device"] = str(args.device)
        env_cfg["kwargs"] = kwargs
        return env_cfg
    return {
        "target": "dreamervla.diagnostics.benchmark_manual_workers:SyntheticBatchWMEnv",
        "kwargs": {
            "num_envs": int(args.num_slots),
            "latent_dim": int(dims["latent_dim"]),
            "action_dim": int(dims["action_dim"]),
            "lang_dim": int(dims["lang_dim"]),
            "proprio_dim": int(dims["proprio_dim"]),
            "horizon": max(1, int(args.chunk_steps) * int(dims["chunk_size"]) + 1),
            "device": str(args.device),
            "matmul_dim": int(args.synthetic_matmul_dim),
        },
    }


def _rollout_cfg(
    args: argparse.Namespace,
    cfg: DictConfig | None,
    dims: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    if cfg is not None:
        policy_cfg = OmegaConf.to_container(
            OmegaConf.select(cfg, "rollout.policy_cfg"), resolve=True
        )
        train_cfg = OmegaConf.to_container(OmegaConf.select(cfg, "rollout.train_cfg"), resolve=True)
        encoder_cfg = None
        if not args.no_rollout_encoder:
            raw_encoder_cfg = OmegaConf.select(cfg, "rollout.encoder_cfg", default=None)
            if raw_encoder_cfg is not None:
                encoder_cfg = OmegaConf.to_container(raw_encoder_cfg, resolve=True)
        if not isinstance(policy_cfg, Mapping) or not isinstance(train_cfg, Mapping):
            raise TypeError("rollout.policy_cfg and rollout.train_cfg must be mappings")
        train_cfg = dict(train_cfg)
        train_cfg["device"] = str(args.device)
        return (
            dict(policy_cfg),
            train_cfg,
            dict(encoder_cfg) if isinstance(encoder_cfg, Mapping) else None,
        )
    return (
        {
            "target": "dreamervla.workers.actor._test_models:TinyLumosPolicy",
            "kwargs": {
                "hidden_dim": int(dims["latent_dim"]),
                "action_dim": int(dims["action_dim"]),
                "chunk_size": int(dims["chunk_size"]),
            },
        },
        {"device": str(args.device) if torch.cuda.is_available() else "cpu"},
        None,
    )


def _select(cfg: DictConfig | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    value = OmegaConf.select(cfg, key, default=None)
    return default if value is None else value


def _env_metrics(worker: WMEnvWorker) -> dict[str, Any]:
    if not worker.envs:
        return {}
    getter = getattr(worker.envs[0], "get_metrics", None)
    if getter is None:
        return {}
    return dict(getter(reset=True))


def _query_gpu_once() -> list[dict[str, int]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2]),
                "util_gpu": int(parts[3]),
                "util_memory": int(parts[4]),
            }
        )
    return rows


def _longest_zero_run(values: list[int]) -> int:
    longest = 0
    current = 0
    for value in values:
        if int(value) == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _write_json_if_requested(metrics: Mapping[str, Any], output_json: str | Path | None) -> None:
    if output_json is None:
        return
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(metrics)), indent=2, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


if __name__ == "__main__":
    main()
