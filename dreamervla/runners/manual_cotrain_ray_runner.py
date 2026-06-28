"""Target manual-cotrain Ray runner.

This route follows ``spec/99_manual_notes.md``: LearnerGroup owns WM/classifier,
ActorGroup owns VLA PPO, RolloutGroup owns no-grad policy inference, and
EnvGroup owns real/WM env interaction.
"""

from __future__ import annotations

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
from dreamervla.workers.cotrain.placement import (
    ManualCotrainPlacementPlan,
    build_manual_cotrain_placement,
)
from dreamervla.workers.cotrain.messages import StopMsg
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
            print(
                "[manual-cotrain] groups="
                + ",".join(self._target_group_names()),
                flush=True,
            )
            metrics: dict[str, float | int] = {}
            for global_step in range(1, self._global_steps() + 1):
                step_metrics = self._run_global_step(groups, global_step)
                metrics.update(step_metrics)
                self.log_metrics(step_metrics, step=global_step)
            metrics["global_step"] = self._global_steps()
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
            ).launch(cluster, NodePlacementStrategy(1), name="ManualReplay")
            replay = replay_group.workers[0]

        real_env_group = WorkerGroup(
            RealEnvWorker,
            self._cfg_dict("env.real.cfg"),
            self._envs_per_worker(),
            self._rollout_epoch(),
            self._max_steps_per_rollout_epoch(),
            self._num_action_chunks(),
            task_id=self._task_id(),
            replay=replay,
            dump=None,
            rank_offset=0,
        ).launch(
            cluster,
            self._placement_for(plan.env_specs[0].gpu_ids),
            name="ManualRealEnvWorker",
        )
        wm_gpus = [spec.gpu_ids[0] for spec in plan.env_specs if spec.role == "wm_env"]
        wm_env_group = None
        if wm_gpus:
            wm_env_group = WorkerGroup(
                WMEnvWorker,
                self._cfg_dict("env.wm.cfg"),
                self._envs_per_worker(),
                self._rollout_epoch(),
                self._max_steps_per_rollout_epoch(),
                self._num_action_chunks(),
                task_id=self._task_id(),
                replay=replay,
                dump=None,
                rank_offset=1,
            ).launch(
                cluster,
                ResourceMapPlacementStrategy(",".join(str(gpu) for gpu in wm_gpus)),
                name="ManualWMEnvWorker",
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
            name="ManualRolloutWorker",
        )

        actor_group = WorkerGroup(
            EmbodiedFSDPActor,
            self._cfg_dict("actor.policy_cfg"),
            self._load_init_ckpt("actor.init_ckpt"),
            self._cfg_dict("actor.train_cfg"),
        ).launch(
            cluster,
            self._resource_map_for_specs(plan.actor_specs),
            name="ManualActor",
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
            name="ManualLearner",
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

        actor.set_global_step(global_step).wait()
        rollout.set_global_step(global_step).wait()
        if global_step % self._sync_every() == 0:
            actor.sync_model_to_rollout("policy", global_step).wait()
            rollout.sync_model_from_actor("policy").wait()
            replay_group = groups.get("ReplayGroup")
            if replay_group is not None:
                replay_group.set_policy_version(global_step).wait()

        env_results = [real_env.interact(env_channel_name, rollout_channel_name, actor_channel_name)]
        if wm_env is not None:
            env_results.append(wm_env.interact(env_channel_name, rollout_channel_name, actor_channel_name))
        rollout_result = rollout.generate(
            env_channel_name,
            rollout_channel_name,
            self._envs_per_worker(),
        )

        env_metrics = _sum_metric_lists([result.wait() for result in env_results])
        for rollout_rank, _ in enumerate(groups["RolloutGroup"].workers):
            for slot_id in range(self._envs_per_worker()):
                groups["env_channel"].put(
                    StopMsg(reason="global_step_complete"),
                    key=f"{int(rollout_rank)}:{int(slot_id)}",
                )
        rollout_metrics = _sum_metric_lists([rollout_result.wait()])
        expected_shards = int(env_metrics.get("env/trajectory_shards", 0.0))
        shards = [
            groups["actor_channel"].get()
            for _ in range(max(0, expected_shards))
        ]
        actor.load_trajectory_shards(shards).wait()
        advantage_metrics = _merge_metric_lists([actor.compute_advantages_and_returns().wait()])
        train_metrics = _merge_metric_lists([actor.run_training().wait()])

        learner_metrics: dict[str, float] = {}
        if global_step % self._learner_update_step() == 0:
            learner_metrics = _merge_metric_lists([learner.update("cotrain", 1).wait()])
            learner.sync_weights("world_model", global_step).wait()
            learner.sync_weights("classifier", global_step).wait()
            if wm_env is not None:
                state_dicts = _first_result(learner.state_dicts().wait())
                if not isinstance(state_dicts, dict):
                    raise TypeError("LearnerGroup.state_dicts() must return a mapping")
                wm_env.load_world_model_state(
                    dict(state_dicts.get("world_model", {})),
                    global_step,
                ).wait()
                wm_env.load_classifier_state(
                    dict(state_dicts.get("classifier", {})),
                    global_step,
                ).wait()

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
            **replay_metrics,
            **advantage_metrics,
            **train_metrics,
            **learner_metrics,
        }
        self._maybe_save_manual_checkpoint(groups, global_step, metrics)
        return metrics

    def _ngpu(self) -> int:
        return int(OmegaConf.select(self.cfg, "manual_cotrain.ngpu", default=1))

    def _global_steps(self) -> int:
        return max(1, int(OmegaConf.select(self.cfg, "manual_cotrain.global_steps", default=1)))

    def _sync_every(self) -> int:
        return max(1, int(OmegaConf.select(self.cfg, "manual_cotrain.sync_every", default=1)))

    def _learner_update_step(self) -> int:
        return max(1, int(OmegaConf.select(self.cfg, "manual_cotrain.learner_update_step", default=1)))

    def _rollout_epoch(self) -> int:
        return max(1, int(OmegaConf.select(self.cfg, "manual_cotrain.rollout_epoch", default=1)))

    def _max_steps_per_rollout_epoch(self) -> int:
        return max(1, int(OmegaConf.select(self.cfg, "manual_cotrain.max_steps_per_rollout_epoch", default=1)))

    def _num_action_chunks(self) -> int:
        return max(1, int(OmegaConf.select(self.cfg, "manual_cotrain.num_action_chunks", default=1)))

    def _envs_per_worker(self) -> int:
        return max(1, int(OmegaConf.select(self.cfg, "manual_cotrain.envs_per_worker", default=1)))

    def _task_id(self) -> int:
        return int(OmegaConf.select(self.cfg, "manual_cotrain.task_id", default=0))

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
        torch.save(
            {
                "global_step": int(global_step),
                "metrics": dict(metrics),
                "state_dicts": {
                    "policy": dict(actor_state),
                    "world_model": dict(learner_states.get("world_model", {})),
                    "classifier": dict(learner_states.get("classifier", {})),
                },
            },
            ckpt_dir / "manual_cotrain.ckpt",
        )

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


def _sum_metric_lists(items: list[Any]) -> dict[str, float]:
    summed: dict[str, float] = {}
    for item in items:
        values = item
        if not isinstance(values, list):
            values = [values]
        for nested in values:
            for key, value in _float_metrics(nested).items():
                summed[key] = summed.get(key, 0.0) + float(value)
    return summed


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
        component_sd = payload.get(component)
        if isinstance(component_sd, dict):
            return component_sd
        if all(isinstance(key, str) for key in payload):
            return payload
    raise RuntimeError(f"{path} does not contain a usable state dict")


__all__ = ["ManualCotrainRayRunner"]
