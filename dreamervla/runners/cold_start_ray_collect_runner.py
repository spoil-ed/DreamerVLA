"""Opt-in Ray cold-start rollout collector."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.runners.base_runner import BaseRunner
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import NodePlacementStrategy
from dreamervla.scheduler.worker_group import WorkerGroup
from dreamervla.workers.env.env_worker import EnvWorker
from dreamervla.workers.inference.inference_worker import InferenceWorker
from dreamervla.workers.rollout.dump_worker import RolloutDumpWorker


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
            groups = self._build_components(cluster)
            return self._run_loop(groups)
        finally:
            cluster.shutdown()

    def _build_components(self, cluster: Cluster) -> dict[str, Any]:
        num_envs = self._int_from(("env.num_workers", "num_env_workers"), 1)
        reward_dir = str(self._select_first(("dump.reward_dir", "reward_dir"), "data/collected_rollouts/ray_synthetic/reward"))
        hidden_dir = str(self._select_first(("dump.hidden_dir", "hidden_dir"), "data/collected_rollouts/ray_synthetic/hidden"))
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

    def _run_loop(self, groups: dict[str, Any]) -> dict[str, float | int]:
        envs = groups["envs"]
        infer = groups["infer"]
        dump = groups["dump"]
        num_envs = int(groups["num_envs"])
        env_ids = list(range(num_envs))
        target_episodes = self._int_from(("rollout.target_episodes", "target_episodes"), num_envs)
        max_steps = self._int_from(("rollout.max_steps", "rollout_steps"), target_episodes * 8)

        steps = 0
        while steps < max_steps and int(dump.size().wait()[0]) < target_episodes:
            obs_batch = envs.current_obs().wait()
            infer_out = infer.forward_batch(obs_batch, env_ids).wait()[0]
            step_results = []
            for rank, action, hidden in zip(
                env_ids, infer_out["actions"], infer_out["obs_embedding"], strict=True
            ):
                step_results.extend(envs.execute_on(rank).step(action, hidden).wait())
            done_envs = [
                env_id
                for env_id, (_obs, done, _info) in zip(env_ids, step_results, strict=True)
                if done
            ]
            if done_envs:
                infer.reset_states(done_envs).wait()
            steps += 1

        episodes = int(dump.size().wait()[0])
        dump.close().wait()
        envs.close().wait()
        return {
            "rollout/episodes": episodes,
            "rollout/steps": int(steps),
            "env/num_env_workers": int(num_envs),
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
