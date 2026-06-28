"""Trajectory-oriented EnvWorkers for the target cotrain channel topology."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.worker import Worker
from dreamervla.workers.cotrain.messages import (
    ObservationMsg,
    RolloutResultMsg,
    TrajectoryShard,
    as_tensor,
)


def _plain_dict(value: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            converted = OmegaConf.to_container(value, resolve=True)
            if isinstance(converted, Mapping):
                return dict(converted)
    except ImportError:
        pass
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"expected mapping config, got {type(value).__name__}")


def _build_env_from_cfg(env_cfg: Mapping[str, Any]) -> Any:
    cfg = _plain_dict(env_cfg)
    target = cfg.get("target") or cfg.get("_target_") or cfg.get("class_path")
    if not target:
        raise ValueError("env_cfg must include target/_target_/class_path")
    kwargs = _plain_dict(cfg.get("kwargs", {}))
    if ":" in str(target):
        module_name, class_name = str(target).split(":", 1)
    else:
        module_name, class_name = str(target).rsplit(".", 1)
    module = importlib.import_module(module_name)
    env_cls = getattr(module, class_name)
    if hasattr(env_cls, "from_config") and bool(cfg.get("use_from_config", False)):
        return env_cls.from_config(kwargs)
    return env_cls(**kwargs)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_action_chunk(value: Any, *, action_dim: int | None = None) -> np.ndarray:
    actions = _as_numpy(value).astype(np.float32, copy=False)
    if actions.ndim == 0:
        raise ValueError("rollout actions must not be scalar")
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim == 3 and int(actions.shape[0]) == 1:
        actions = actions[0]
    if actions.ndim != 2:
        raise ValueError("rollout actions must have shape [chunk, action_dim]")
    if int(actions.shape[0]) <= 0:
        raise ValueError("rollout action chunk must be non-empty")
    if action_dim is not None and int(actions.shape[-1]) != int(action_dim):
        raise ValueError(
            "rollout action dim mismatch: "
            f"got {int(actions.shape[-1])}, expected {int(action_dim)}"
        )
    return np.asarray(actions, dtype=np.float32)


def _one_chunk_batch(
    value: Any,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = as_tensor(value, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1).detach().cpu()
    if tensor.ndim == 1:
        return tensor.reshape(1, -1)[:, :1].detach().cpu()
    if int(tensor.shape[0]) == 1:
        return tensor.reshape(1, 1, *tensor.shape[1:]).detach().cpu()
    return tensor.reshape(1, 1, *tensor.shape).detach().cpu()


def _transition_value(value: Any) -> Any:
    tensor = as_tensor(value).detach().cpu()
    if tensor.ndim > 0 and int(tensor.shape[0]) == 1:
        tensor = tensor.squeeze(0)
    return tensor.numpy()


def _transition_version(value: Any) -> int:
    tensor = as_tensor(value).detach().cpu()
    if tensor.numel() == 0:
        return 0
    return int(tensor.reshape(-1)[0].item())


def _transition_sidecars_from_rollout(result: RolloutResultMsg) -> dict[str, Any]:
    sidecars: dict[str, Any] = {}
    forward_inputs = dict(result.forward_inputs)
    if "hidden" in forward_inputs:
        sidecars["obs_embedding"] = _transition_value(forward_inputs["hidden"])
    if "lang_emb" in forward_inputs:
        sidecars["lang_emb"] = _transition_value(forward_inputs["lang_emb"])
    for name, value in dict(result.versions).items():
        key = "policy_version" if str(name) == "policy" else f"{name}_version"
        sidecars[key] = _transition_version(value)
    return sidecars


def _merge_transition_sidecars(
    transition: dict[str, Any],
    obs: dict[str, Any],
) -> dict[str, Any]:
    for key, value in obs.items():
        if key == "obs_embedding":
            transition.setdefault(key, np.asarray(value, dtype=np.float32))
        elif key == "lang_emb":
            transition.setdefault(key, np.asarray(value, dtype=np.float32))
        elif key == "policy_version" or key.endswith("_version"):
            transition.setdefault(key, int(value))
    return transition


def _episode_policy_version(episode: list[dict[str, Any]]) -> int | None:
    versions = [
        int(step["policy_version"])
        for step in episode
        if "policy_version" in step
    ]
    if not versions:
        return None
    return int(versions[-1])


def _call_maybe_remote(method: Any, *args: Any, **kwargs: Any) -> Any:
    remote = getattr(method, "remote", None)
    if remote is not None:
        import ray

        return ray.get(remote(*args, **kwargs))
    return method(*args, **kwargs)


class BaseTrajectoryEnvWorker(Worker):
    """EnvWorker that turns action chunks into step-major trajectory shards."""

    role_name = "env"

    def __init__(
        self,
        role: str,
        env_cfg: Mapping[str, Any],
        num_slots: int,
        rollout_epoch: int,
        max_steps_per_rollout_epoch: int,
        num_action_chunks: int,
        task_id: int = 0,
        replay: Any | None = None,
        dump: Any | None = None,
        rank_offset: int = 0,
    ) -> None:
        super().__init__()
        self.role = str(role)
        self.env_cfg = _plain_dict(env_cfg)
        self.action_postprocess = str(
            self.env_cfg.get("action_postprocess", "none")
        ).strip().lower()
        self.num_slots = int(num_slots)
        self.rollout_epoch = int(rollout_epoch)
        self.max_steps_per_rollout_epoch = int(max_steps_per_rollout_epoch)
        self.num_action_chunks = int(num_action_chunks)
        self.task_id = int(task_id)
        self.replay = replay
        self.dump = dump
        self.rank_offset = int(rank_offset)
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.rollout_epoch <= 0:
            raise ValueError("rollout_epoch must be positive")
        if self.max_steps_per_rollout_epoch <= 0:
            raise ValueError("max_steps_per_rollout_epoch must be positive")
        if self.num_action_chunks <= 0:
            raise ValueError("num_action_chunks must be positive")

        self.envs: list[Any] = []
        self._batched_env = False
        self._obs_by_slot: list[dict[str, Any] | None] = [
            None for _ in range(self.num_slots)
        ]
        self._episodes_by_slot: list[list[dict[str, Any]]] = [
            [] for _ in range(self.num_slots)
        ]
        self._episode_ids_by_slot: list[int] = [0 for _ in range(self.num_slots)]
        self._task_ids_by_slot: list[int] = [
            self.task_id for _ in range(self.num_slots)
        ]
        self._model_versions: dict[str, int] = {}
        self._pending_component_states: dict[str, tuple[dict[str, Any], int]] = {}
        self._last_apply_completed_episodes = 0
        self._last_apply_physical_steps = 0

    def init(self) -> None:
        """Build all local env slots."""

        if self.envs:
            return
        first_env = _build_env_from_cfg(self.env_cfg)
        if hasattr(first_env, "reset_slot") and hasattr(first_env, "step_slot"):
            env_count = getattr(first_env, "num_envs", None)
            if env_count is not None and int(env_count) < self.num_slots:
                raise ValueError(
                    f"batched env supports {int(env_count)} slots; "
                    f"worker needs {self.num_slots}"
                )
            self.envs = [first_env]
            self._batched_env = True
            self._bootstrap_wm_initial_latents_from_replay()
            self._apply_pending_component_states()
            return

        self.envs = [first_env] + [
            _build_env_from_cfg(self.env_cfg) for _ in range(self.num_slots - 1)
        ]
        self._batched_env = False
        for slot_id, env in enumerate(self.envs):
            task_id = int(self._task_ids_by_slot[slot_id])
            if hasattr(env, "set_task"):
                env.set_task(task_id)
        self._bootstrap_wm_initial_latents_from_replay()
        self._apply_pending_component_states()

    def bootstrap_obs(self) -> list[ObservationMsg]:
        """Reset each slot and emit the first observation for rollout inference."""

        self._ensure_initialized()
        messages: list[ObservationMsg] = []
        for slot_id in range(self.num_slots):
            obs = self._reset_slot(slot_id)
            messages.append(self._observation_msg(slot_id, obs))
        return messages

    def apply_rollout_result(self, result: RolloutResultMsg) -> TrajectoryShard:
        """Step one slot through a rollout action chunk and return a shard."""

        self._ensure_initialized()
        slot_id = int(result.slot_id)
        self._validate_slot(slot_id)
        actions_np = _as_action_chunk(result.actions)
        if int(actions_np.shape[0]) > self.num_action_chunks:
            raise ValueError(
                "rollout action chunk length must be <= num_action_chunks "
                f"({self.num_action_chunks})"
            )

        chunk_len = int(actions_np.shape[0])
        action_dim = int(actions_np.shape[-1])
        rewards = np.zeros((self.num_action_chunks,), dtype=np.float32)
        dones = np.zeros((self.num_action_chunks,), dtype=np.bool_)
        completed = 0
        physical_steps = 0
        transition_sidecars = _transition_sidecars_from_rollout(result)
        for index, action in enumerate(actions_np):
            _, reward, done, _ = self._step_slot(
                slot_id,
                action,
                transition_sidecars=transition_sidecars,
            )
            rewards[index] = float(reward)
            dones[index] = bool(done)
            physical_steps += 1
            if done:
                completed = 1
                if index + 1 < self.num_action_chunks:
                    dones[index + 1 :] = True
                break

        self._last_apply_completed_episodes = int(completed)
        self._last_apply_physical_steps = int(physical_steps)
        action_pad = np.zeros(
            (self.num_action_chunks, action_dim),
            dtype=np.float32,
        )
        action_pad[:chunk_len] = actions_np
        action_tensor = torch.as_tensor(action_pad, dtype=torch.float32).view(
            1,
            1,
            self.num_action_chunks,
            action_dim,
        )
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32).view(
            1,
            1,
            self.num_action_chunks,
        )
        done_tensor = torch.as_tensor(dones, dtype=torch.bool).view(
            1,
            1,
            self.num_action_chunks,
        )
        prev_values = None
        if result.prev_values is not None:
            prev_values = _one_chunk_batch(result.prev_values, dtype=torch.float32)

        return TrajectoryShard(
            env_rank=int(result.env_rank),
            slot_id=slot_id,
            task_id=int(result.task_id),
            episode_ids=[int(result.episode_id)],
            actions=action_tensor,
            rewards=reward_tensor,
            dones=done_tensor,
            prev_logprobs=_one_chunk_batch(
                result.prev_logprobs,
                dtype=torch.float32,
            ),
            prev_values=prev_values,
            forward_inputs={
                str(key): _one_chunk_batch(value)
                for key, value in dict(result.forward_inputs).items()
            },
            versions={
                str(key): _one_chunk_batch(value, dtype=torch.long)
                for key, value in dict(result.versions).items()
            },
        )

    def interact(
        self,
        env_channel_name: str,
        rollout_channel_name: str,
        actor_channel_name: str,
    ) -> dict[str, float]:
        """Run a local EnvGroup interaction loop over named cotrain channels."""

        env_channel = Channel.connect(env_channel_name)
        rollout_channel = Channel.connect(rollout_channel_name)
        actor_channel = Channel.connect(actor_channel_name)
        metrics = {
            "env/chunk_steps": 0.0,
            "env/physical_steps": 0.0,
            "env/steps": 0.0,
            "env/trajectory_shards": 0.0,
            "env/episodes_completed": 0.0,
            "env/episodes_flushed": 0.0,
            "env/final_bootstrap_requests": 0.0,
        }

        for _ in range(self.rollout_epoch):
            for message in self.bootstrap_obs():
                env_channel.put(message)

            target_chunk_steps = self._chunk_steps_per_rollout_epoch()
            chunk_steps_by_slot = [0 for _ in range(self.num_slots)]
            while any(steps < target_chunk_steps for steps in chunk_steps_by_slot):
                for slot_id in range(self.num_slots):
                    if chunk_steps_by_slot[slot_id] >= target_chunk_steps:
                        continue
                    key = self._slot_key(slot_id)
                    result = rollout_channel.get(key=key)
                    shard = self.apply_rollout_result(result)
                    actor_channel.put(shard)
                    chunk_steps_by_slot[slot_id] += 1
                    metrics["env/chunk_steps"] += 1.0
                    metrics["env/physical_steps"] += float(
                        self._last_apply_physical_steps
                    )
                    metrics["env/steps"] += float(self._last_apply_physical_steps)
                    metrics["env/trajectory_shards"] += 1.0
                    metrics["env/episodes_completed"] += float(
                        self._last_apply_completed_episodes
                    )
                    if chunk_steps_by_slot[slot_id] < target_chunk_steps:
                        obs = self._obs_by_slot[slot_id]
                        if obs is None:
                            raise RuntimeError("slot has no current observation")
                        message = self._observation_msg(slot_id, obs)
                        env_channel.put(message)
            for slot_id in range(self.num_slots):
                obs = self._obs_by_slot[slot_id]
                if obs is None:
                    continue
                message = self._observation_msg(slot_id, obs)
                message.obs["_final_bootstrap"] = True
                env_channel.put(message)
                rollout_channel.get(key=self._slot_key(slot_id))
                metrics["env/final_bootstrap_requests"] += 1.0
                metrics["env/episodes_flushed"] += float(
                    self._flush_partial_episode(slot_id)
                )
        return metrics

    def load_world_model_state(self, state_dict: dict[str, Any], version: int) -> None:
        """Delegate world-model state sync to env slots that support it."""

        self._model_versions["world_model"] = int(version)
        self._pending_component_states["world_model"] = (state_dict, int(version))
        self._apply_component_state("world_model", state_dict, int(version))

    def load_classifier_state(self, state_dict: dict[str, Any], version: int) -> None:
        """Delegate classifier/reward state sync to env slots that support it."""

        self._model_versions["classifier"] = int(version)
        self._pending_component_states["classifier"] = (state_dict, int(version))
        self._apply_component_state("classifier", state_dict, int(version))

    def _apply_pending_component_states(self) -> None:
        for component, (state_dict, version) in self._pending_component_states.items():
            self._apply_component_state(component, state_dict, version)

    def _apply_component_state(
        self,
        component: str,
        state_dict: dict[str, Any],
        version: int,
    ) -> None:
        loader_name = {
            "world_model": "load_world_model_state",
            "classifier": "load_classifier_state",
        }.get(str(component))
        if loader_name is None:
            raise ValueError(f"unknown component state {component!r}")
        for env in self.envs:
            loader = getattr(env, loader_name, None)
            if loader is not None:
                loader(state_dict, int(version))
                continue
            if self.role == "wm_env":
                raise TypeError(
                    f"WMEnvWorker env {type(env).__name__} must expose {loader_name}()"
                )

    def close(self) -> None:
        """Close all env slots."""

        for env in self.envs:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        self.envs = []
        self._batched_env = False
        self._obs_by_slot = [None for _ in range(self.num_slots)]
        self._episodes_by_slot = [[] for _ in range(self.num_slots)]

    def _reset_slot(self, slot_id: int) -> dict[str, Any]:
        self._validate_slot(slot_id)
        env = self._env_for_slot(slot_id)
        task_id = int(self._task_ids_by_slot[slot_id])
        episode_id = int(self._episode_ids_by_slot[slot_id])
        if self._batched_env:
            obs, _ = env.reset_slot(slot_id, task_id=task_id, episode_id=episode_id)
        else:
            if hasattr(env, "set_task"):
                env.set_task(task_id)
            obs, _ = env.reset(task_id=task_id, episode_id=episode_id)
        self._obs_by_slot[slot_id] = dict(obs)
        self._episodes_by_slot[slot_id] = []
        return dict(obs)

    def _step_slot(
        self,
        slot_id: int,
        action: Any,
        *,
        transition_sidecars: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._validate_slot(slot_id)
        env = self._env_for_slot(slot_id)
        obs = self._obs_by_slot[slot_id]
        if obs is None:
            raise RuntimeError("bootstrap_obs() must be called before stepping")
        policy_action = np.asarray(action, dtype=np.float32).reshape(-1)
        env_action = self._env_action_from_policy_action(policy_action)
        if self._batched_env:
            step_out = env.step_slot(slot_id, env_action)
        else:
            step_out = env.step(env_action)
        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
        elif len(step_out) == 4:
            next_obs, reward, terminated, info = step_out
            truncated = False
        else:
            raise ValueError("env.step(action) must return 4 or 5 values")
        info = dict(info or {})
        info.setdefault("wm_action", np.asarray(env_action, dtype=np.float32).reshape(-1))
        done = bool(terminated or truncated)
        transition_obs = dict(obs)
        transition_obs.update(self._model_version_sidecars())
        if transition_sidecars:
            transition_obs.update(transition_sidecars)
        transition = self._make_transition(
            env,
            transition_obs,
            next_obs,
            policy_action,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )
        self._episodes_by_slot[slot_id].append(transition)
        if done:
            self._push_episode(self.replay, self._episodes_by_slot[slot_id])
            self._push_episode(self.dump, self._episodes_by_slot[slot_id])
            self._episodes_by_slot[slot_id] = []
            self._episode_ids_by_slot[slot_id] += 1
            next_obs = self._reset_slot(slot_id)
        else:
            self._obs_by_slot[slot_id] = dict(next_obs)
        return dict(next_obs), float(reward), done, info

    def _env_for_slot(self, slot_id: int) -> Any:
        self._validate_slot(slot_id)
        if self._batched_env:
            return self.envs[0]
        return self.envs[slot_id]

    def _chunk_steps_per_rollout_epoch(self) -> int:
        if self.num_action_chunks <= 0:
            raise ValueError("num_action_chunks must be positive")
        if self.max_steps_per_rollout_epoch % self.num_action_chunks != 0:
            raise ValueError(
                "max_steps_per_rollout_epoch must be divisible by num_action_chunks "
                f"({self.max_steps_per_rollout_epoch} % {self.num_action_chunks})"
            )
        return self.max_steps_per_rollout_epoch // self.num_action_chunks

    def _env_action_from_policy_action(self, action: Any) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_postprocess in {"", "none", "false"}:
            return action_arr
        if self.action_postprocess in {"openvla_oft", "oft"}:
            from dreamervla.runners.oft_collect_common import process_action

            return process_action(action_arr)
        raise ValueError(f"unknown env action_postprocess: {self.action_postprocess!r}")

    def _model_version_sidecars(self) -> dict[str, int]:
        return {
            f"{str(name)}_version": int(version)
            for name, version in self._model_versions.items()
        }

    def _flush_partial_episode(self, slot_id: int) -> int:
        self._validate_slot(slot_id)
        episode = self._episodes_by_slot[slot_id]
        if not episode:
            return 0
        flushed = [dict(step) for step in episode]
        _mark_transition_truncated(flushed[-1])
        self._push_episode(self.replay, flushed)
        self._push_episode(self.dump, flushed)
        self._episodes_by_slot[slot_id] = []
        self._episode_ids_by_slot[slot_id] += 1
        return 1

    def _bootstrap_wm_initial_latents_from_replay(self) -> None:
        if self.role != "wm_env" or self.replay is None:
            return
        size_method = getattr(self.replay, "size", None)
        if size_method is not None:
            size = _call_maybe_remote(size_method)
            if int(size) <= 0:
                return
        sampler = getattr(self.replay, "sample_initial_obs_embeddings", None)
        if sampler is None:
            return
        try:
            raw = _call_maybe_remote(
                sampler,
                self.num_slots,
                task_id=self.task_id,
                key="obs_embedding",
            )
        except RuntimeError:
            return
        latents = np.asarray(raw, dtype=np.float32)
        for env in self.envs:
            setter = getattr(env, "set_initial_latents", None)
            if setter is not None:
                setter(latents)

    @staticmethod
    def _make_transition(
        env: Any,
        obs: dict[str, Any],
        next_obs: dict[str, Any],
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        if hasattr(env, "make_transition"):
            transition = env.make_transition(
                obs, action, reward, terminated, truncated, info
            )
            return _merge_transition_sidecars(dict(transition), obs)
        return _merge_transition_sidecars({
            "obs": dict(obs),
            "next_obs": dict(next_obs),
            "action": np.asarray(action, dtype=np.float32),
            "reward": float(reward),
            "done": bool(terminated or truncated),
            "is_terminal": bool(terminated),
            "is_last": bool(terminated or truncated),
            "info": dict(info or {}),
        }, obs)

    @staticmethod
    def _push_episode(target: Any | None, episode: list[dict[str, Any]]) -> None:
        if target is None or not episode:
            return
        policy_version = _episode_policy_version(episode)
        if policy_version is not None:
            set_policy_version = getattr(target, "set_policy_version", None)
            if set_policy_version is not None:
                remote = getattr(set_policy_version, "remote", None)
                if remote is not None:
                    import ray

                    ray.get(remote(int(policy_version)))
                else:
                    set_policy_version(int(policy_version))
        add_episode = target.add_episode
        remote = getattr(add_episode, "remote", None)
        if remote is not None:
            import ray

            ray.get(remote(list(episode)))
        else:
            add_episode(list(episode))

    def _observation_msg(self, slot_id: int, obs: dict[str, Any]) -> ObservationMsg:
        return ObservationMsg(
            env_rank=int(self.rank) + int(self.rank_offset),
            slot_id=int(slot_id),
            task_id=int(self._task_ids_by_slot[slot_id]),
            episode_id=int(self._episode_ids_by_slot[slot_id]),
            step=int(obs.get("step", 0)),
            obs=dict(obs),
            versions=dict(self._model_versions),
        )

    def _slot_key(self, slot_id: int) -> str:
        return f"{int(self.rank) + int(self.rank_offset)}:{int(slot_id)}"

    def _ensure_initialized(self) -> None:
        expected_envs = 1 if self._batched_env else self.num_slots
        if len(self.envs) != expected_envs:
            raise RuntimeError("BaseTrajectoryEnvWorker.init() has not been called")

    def _validate_slot(self, slot_id: int) -> None:
        if not 0 <= int(slot_id) < self.num_slots:
            raise ValueError(f"slot_id {slot_id} is out of range")


class RealEnvWorker(BaseTrajectoryEnvWorker):
    """Trajectory worker for real environment rollout slots."""

    role_name = "real_env"

    def __init__(
        self,
        env_cfg: Mapping[str, Any],
        num_slots: int,
        rollout_epoch: int,
        max_steps_per_rollout_epoch: int,
        num_action_chunks: int,
        task_id: int = 0,
        replay: Any | None = None,
        dump: Any | None = None,
        rank_offset: int = 0,
    ) -> None:
        super().__init__(
            self.role_name,
            env_cfg,
            num_slots,
            rollout_epoch,
            max_steps_per_rollout_epoch,
            num_action_chunks,
            task_id=task_id,
            replay=replay,
            dump=dump,
            rank_offset=rank_offset,
        )


class WMEnvWorker(BaseTrajectoryEnvWorker):
    """Trajectory worker for world-model imagined environment slots."""

    role_name = "wm_env"

    def __init__(
        self,
        env_cfg: Mapping[str, Any],
        num_slots: int,
        rollout_epoch: int,
        max_steps_per_rollout_epoch: int,
        num_action_chunks: int,
        task_id: int = 0,
        replay: Any | None = None,
        dump: Any | None = None,
        rank_offset: int = 0,
    ) -> None:
        super().__init__(
            self.role_name,
            env_cfg,
            num_slots,
            rollout_epoch,
            max_steps_per_rollout_epoch,
            num_action_chunks,
            task_id=task_id,
            replay=replay,
            dump=dump,
            rank_offset=rank_offset,
        )


def _same_scalar_kind(value: Any, replacement: bool | float) -> Any:
    if isinstance(value, np.ndarray):
        return np.asarray(replacement, dtype=value.dtype)
    if isinstance(value, np.generic):
        return type(value)(replacement)
    if isinstance(value, bool):
        return bool(replacement)
    if isinstance(value, int):
        return int(replacement)
    if isinstance(value, float):
        return float(replacement)
    return replacement


def _mark_transition_truncated(step: dict[str, Any]) -> None:
    if "done" in step:
        step["done"] = _same_scalar_kind(step["done"], True)
    if "dones" in step:
        step["dones"] = _same_scalar_kind(step["dones"], True)
    if "is_last" in step:
        step["is_last"] = _same_scalar_kind(step["is_last"], True)
    else:
        step["is_last"] = True
    if "is_terminal" in step:
        step["is_terminal"] = _same_scalar_kind(step["is_terminal"], False)
    else:
        step["is_terminal"] = False
    if "discount" in step:
        step["discount"] = _same_scalar_kind(step["discount"], 1.0)
