"""Opt-in Ray online cotrain runner.

This first runner wires the new scheduler/workers into a lightweight synthetic
online loop that exercises the same production boundaries: env rollout,
batched inference, replay insertion, learner PPO-style update, and policy
weight sync. Real LIBERO/VLA construction can plug into these boundaries
without changing the scheduler primitives.
"""

from __future__ import annotations

import time
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.runners.base_runner import BaseRunner
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import NodePlacementStrategy
from dreamervla.scheduler.worker_group import WorkerGroup
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
        cluster = Cluster(self.cfg.get("cluster"))
        try:
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
                ("learner.train_cfg.syncer.store_name", "sync.store_name", "weight_store_name"),
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
        infer_group = WorkerGroup(InferenceWorker, infer_cfg, {}, num_envs=num_envs).launch(
            cluster, NodePlacementStrategy(1)
        )

        learner_model_cfg = self._cfg_from("learner.model_cfg", {"policy": policy_cfg})
        learner_model_cfg.setdefault("policy", policy_cfg)
        learner_train_cfg = self._cfg_from(
            "learner.train_cfg",
            {
                "mode": "synthetic_ppo",
                "batch_size": int(self.cfg.get("ppo_batch_size", 2)),
                "lr": float(self.cfg.get("ppo_lr", 0.05)),
                "device": "cpu",
                "syncer": {"store_name": store_name},
            },
        )
        learner_train_cfg.setdefault("mode", "synthetic_ppo")
        learner_train_cfg.setdefault("batch_size", int(self.cfg.get("ppo_batch_size", 2)))
        learner_train_cfg.setdefault("lr", float(self.cfg.get("ppo_lr", 0.05)))
        learner_train_cfg.setdefault("device", "cpu")
        learner_train_cfg.setdefault("syncer", {})
        learner_train_cfg["syncer"].setdefault("store_name", store_name)

        learner_group = WorkerGroup(
            LearnerWorker,
            learner_model_cfg,
            {},
            learner_train_cfg,
            replay,
        ).launch(cluster, NodePlacementStrategy(1))
        return {
            "replay": replay_group,
            "envs": env_group,
            "infer": infer_group,
            "learner": learner_group,
            "store_name": store_name,
            "num_envs": num_envs,
        }

    def _run_loop(self, groups: dict[str, Any]) -> dict[str, float | int]:
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

        ppo_updates = 0
        policy_version = 0
        local_infer_version = 0
        last_loss = 0.0
        overlap_events = 0
        pending_learn = None
        pending_learn_start = 0.0

        for step in range(rollout_steps):
            obs_batch = envs.current_obs().wait()

            if pending_learn is None and replay.ready(min_episodes).wait()[0]:
                pending_learn = learner.update("rl", 1)
                pending_learn_start = time.perf_counter()

            rollout_start = time.perf_counter()
            infer_out = infer.forward_batch(obs_batch, env_ids).wait()[0]
            step_results = []
            for rank, action, hidden in zip(
                env_ids, infer_out["actions"], infer_out["obs_embedding"], strict=True
            ):
                step_results.extend(envs.execute_on(rank).step(action, hidden).wait())
            rollout_end = time.perf_counter()

            done_envs = [env_id for env_id, (_obs, done, _info) in zip(env_ids, step_results, strict=True) if done]
            if done_envs:
                infer.reset_states(done_envs).wait()

            if pending_learn is not None:
                learn_done = pending_learn.done()
                if learn_done or step == rollout_steps - 1:
                    metrics = pending_learn.wait()[0]
                    last_loss = float(metrics.get("train/rl_loss", 0.0))
                    ppo_updates += 1
                    policy_version += 1
                    learner.sync_weights("policy", policy_version).wait()
                    if policy_version % weight_sync_every == 0:
                        pulled = infer.pull_weights(
                            groups["store_name"], "policy", local_infer_version
                        ).wait()[0]
                        if pulled is not None:
                            local_infer_version = int(pulled)
                    if pending_learn_start < rollout_end and rollout_start < time.perf_counter():
                        overlap_events += 1
                    pending_learn = None

        if pending_learn is not None:
            metrics = pending_learn.wait()[0]
            last_loss = float(metrics.get("train/rl_loss", 0.0))
            ppo_updates += 1
            policy_version += 1
            learner.sync_weights("policy", policy_version).wait()

        return {
            "rollout/episodes": int(replay.size().wait()[0]),
            "train/ppo_updates": int(ppo_updates),
            "sync/policy_version": int(policy_version),
            "time/overlap_events": int(overlap_events),
            "train/rl_loss": float(last_loss),
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
