"""Trajectory-oriented EnvWorkers for the target cotrain channel topology."""

from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.worker import Worker
from dreamervla.workers.cotrain.handshake_trace import trace as _hs_trace
from dreamervla.workers.cotrain.messages import (
    ObservationBatchMsg,
    ObservationMsg,
    RolloutResultBatchMsg,
    RolloutResultMsg,
    TrajectoryShard,
    _shard_loss_mask,
    as_tensor,
)

_ACTOR_PUT_FLUSH_EVERY = 64


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


def _state_dict_nbytes(state_dict: Mapping[str, Any]) -> int:
    total = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel() * value.element_size())
        else:
            tensor = torch.as_tensor(value)
            total += int(tensor.numel() * tensor.element_size())
    return total


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


def _concat_trajectory_shards(shards: list[TrajectoryShard]) -> TrajectoryShard:
    if not shards:
        raise ValueError("cannot concatenate an empty trajectory shard buffer")
    first = shards[0]
    forward_keys = set(first.forward_inputs)
    version_keys = set(first.versions)
    has_prev_values = first.prev_values is not None
    loss_masks: list[torch.Tensor] = []
    for shard in shards:
        if int(shard.env_rank) != int(first.env_rank):
            raise ValueError("buffered trajectory shards must share env_rank")
        if int(shard.slot_id) != int(first.slot_id):
            raise ValueError("buffered trajectory shards must share slot_id")
        if int(shard.task_id) != int(first.task_id):
            raise ValueError("buffered trajectory shards must share task_id")
        if set(shard.forward_inputs) != forward_keys:
            raise ValueError("buffered trajectory shards must share forward_input keys")
        if set(shard.versions) != version_keys:
            raise ValueError("buffered trajectory shards must share version keys")
        if (shard.prev_values is not None) != has_prev_values:
            raise ValueError("buffered trajectory shards must consistently include values")
        batch_size = int(as_tensor(shard.actions).shape[1])
        loss_masks.append(
            _shard_loss_mask(
                shard,
                int(as_tensor(shard.actions).shape[0]),
                batch_size,
            )
        )

    prev_values = None
    if has_prev_values:
        prev_values = torch.cat(
            [as_tensor(shard.prev_values).detach().cpu() for shard in shards],
            dim=0,
        )

    return TrajectoryShard(
        env_rank=int(first.env_rank),
        slot_id=int(first.slot_id),
        task_id=int(first.task_id),
        episode_ids=list(first.episode_ids),
        actions=torch.cat(
            [as_tensor(shard.actions).detach().cpu() for shard in shards],
            dim=0,
        ),
        rewards=torch.cat(
            [as_tensor(shard.rewards).detach().cpu() for shard in shards],
            dim=0,
        ),
        dones=torch.cat(
            [as_tensor(shard.dones).detach().cpu() for shard in shards],
            dim=0,
        ),
        prev_logprobs=torch.cat(
            [as_tensor(shard.prev_logprobs).detach().cpu() for shard in shards],
            dim=0,
        ),
        prev_values=prev_values,
        forward_inputs={
            key: torch.cat(
                [as_tensor(shard.forward_inputs[key]).detach().cpu() for shard in shards],
                dim=0,
            )
            for key in sorted(forward_keys)
        },
        versions={
            key: torch.cat(
                [as_tensor(shard.versions[key]).detach().cpu() for shard in shards],
                dim=0,
            )
            for key in sorted(version_keys)
        },
        loss_mask=torch.cat(loss_masks, dim=0),
    )


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


def _version_sidecar_key(name: str) -> str:
    if str(name) == "policy":
        return "policy_version"
    if str(name) == "global_step" or str(name).endswith("_version"):
        return str(name)
    return f"{str(name)}_version"


def _transition_sidecars_from_rollout(result: RolloutResultMsg) -> dict[str, Any]:
    sidecars: dict[str, Any] = {}
    forward_inputs = dict(result.forward_inputs)
    if "hidden" in forward_inputs:
        sidecars["obs_embedding"] = _transition_value(forward_inputs["hidden"])
    if "lang_emb" in forward_inputs:
        sidecars["lang_emb"] = _transition_value(forward_inputs["lang_emb"])
    for name, value in dict(result.versions).items():
        sidecars[_version_sidecar_key(str(name))] = _transition_version(value)
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
        elif key == "proprio":
            transition.setdefault(key, np.asarray(value, dtype=np.float32).reshape(-1))
        elif key == "global_step" or key == "policy_version" or key.endswith("_version"):
            transition.setdefault(key, int(value))
    if "proprio" not in transition:
        if "proprio" in obs:
            transition["proprio"] = np.asarray(obs["proprio"], dtype=np.float32).reshape(-1)
        elif "state" in obs:
            transition["proprio"] = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        elif "state" in transition:
            transition["proprio"] = np.asarray(
                transition["state"],
                dtype=np.float32,
            ).reshape(-1)
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


def _make_env_transition(
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
        request_final_bootstrap: bool = True,
        replay_write_enabled: bool = True,
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
        self.request_final_bootstrap = bool(request_final_bootstrap)
        self.replay_write_enabled = bool(replay_write_enabled)
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
        self._actor_shards_by_slot: list[list[TrajectoryShard]] = [
            [] for _ in range(self.num_slots)
        ]
        self._episode_ids_by_slot: list[int] = [0 for _ in range(self.num_slots)]
        self._task_ids_by_slot: list[int] = [
            self.task_id for _ in range(self.num_slots)
        ]
        self._model_versions: dict[str, int] = {}
        self.global_step = 0
        self._pending_component_states: dict[str, tuple[dict[str, Any], int]] = {}
        self._last_apply_completed_episodes = 0
        self._last_apply_successful_episodes = 0
        self._last_apply_physical_steps = 0
        self._last_apply_env_crashes = 0
        self._last_apply_env_respawns = 0
        self._progress_path: Path | None = None
        self._progress_min_interval_s = 5.0
        self._progress_last_write_t: float | None = None

    def set_global_step(self, global_step: int) -> None:
        """Set runner-visible progress metadata for observations and replay."""

        self.global_step = int(global_step)

    def configure_progress(
        self,
        progress_dir: str | os.PathLike[str] | None,
        min_interval_s: float = 5.0,
    ) -> dict[str, float]:
        """Configure runner-visible manual-cotrain env progress reporting."""

        if progress_dir in (None, ""):
            self._progress_path = None
            self._progress_last_write_t = None
            return {"env/progress_configured": 0.0}
        path = Path(progress_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self._progress_path = path / f"{self.role}_{self._rank_key()}.json"
        self._progress_min_interval_s = max(0.0, float(min_interval_s))
        self._progress_last_write_t = None
        self._write_interact_progress(
            done=0,
            total=self._interact_progress_total(),
            active=False,
            finished=False,
            force=True,
        )
        return {"env/progress_configured": 1.0}

    def configure_rollout_epoch(self, rollout_epoch: int) -> dict[str, float]:
        """Update this worker's per-slot trajectory count for the next interaction."""

        value = int(rollout_epoch)
        if value <= 0:
            raise ValueError(f"rollout_epoch must be positive, got {value}")
        self.rollout_epoch = value
        return {
            "env/rollout_epoch": float(value),
            f"env/{self.role}/rollout_epoch": float(value),
        }

    def init(self) -> None:
        """Build all local env slots."""

        if self.envs:
            return
        self._reject_legacy_spawn_config()
        self._pin_inproc_render_backend()
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

    def _pin_inproc_render_backend(self) -> None:
        if self.role != "real_env":
            return
        render_backend = str(self.env_cfg.get("render_backend", "")).strip().lower()
        if render_backend == "egl":
            return
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    def _reject_legacy_spawn_config(self) -> None:
        raw = self.env_cfg.get("spawn_env_slots", False)
        enabled = str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}
        if enabled:
            raise ValueError(
                "env_cfg.spawn_env_slots has been removed from manual cotrain; "
                "EnvWorker slots now run in the Ray actor process."
            )

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

        action_dim = int(actions_np.shape[-1])
        rewards = np.zeros((self.num_action_chunks,), dtype=np.float32)
        dones = np.zeros((self.num_action_chunks,), dtype=np.bool_)
        completed = 0
        successful = 0
        physical_steps = 0
        env_crashes = 0
        env_respawns = 0
        transition_sidecars = _transition_sidecars_from_rollout(result)
        for index, action in enumerate(actions_np):
            _, reward, done, info = self._step_slot(
                slot_id,
                action,
                transition_sidecars=transition_sidecars,
            )
            rewards[index] = float(reward)
            dones[index] = bool(done)
            physical_steps += 1
            env_crashes += int(self._last_apply_env_crashes)
            env_respawns += int(self._last_apply_env_respawns)
            if done:
                completed = 1
                successful = int(bool(info.get("success", False)))
                if index + 1 < self.num_action_chunks:
                    dones[index + 1 :] = True
                break

        self._last_apply_completed_episodes = int(completed)
        self._last_apply_successful_episodes = int(successful)
        self._last_apply_physical_steps = int(physical_steps)
        self._last_apply_env_crashes = int(env_crashes)
        self._last_apply_env_respawns = int(env_respawns)
        return self._build_trajectory_shard(
            result,
            actions_np=actions_np,
            rewards=rewards,
            dones=dones,
            action_dim=action_dim,
        )

    def _build_trajectory_shard(
        self,
        result: RolloutResultMsg,
        *,
        actions_np: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        action_dim: int,
    ) -> TrajectoryShard:
        action_pad = np.zeros(
            (self.num_action_chunks, action_dim),
            dtype=np.float32,
        )
        chunk_len = int(actions_np.shape[0])
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
            slot_id=int(result.slot_id),
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
        metrics = self._new_interact_metrics()
        interact_start = time.perf_counter()
        progress_total = self._interact_progress_total()
        self._write_interact_progress(
            done=0,
            total=progress_total,
            active=True,
            finished=False,
            force=True,
        )
        if self._can_batch_wm_slots():
            out = self._interact_batched_wm_slots(
                env_channel,
                rollout_channel,
                actor_channel,
                metrics,
            )
            out["env/interact_loop_s"] = float(
                time.perf_counter() - interact_start
            )
            out[f"env/{self.role}/interact_loop_s"] = out["env/interact_loop_s"]
            self._write_interact_progress(
                done=int(out.get("env/chunk_steps", 0.0)),
                total=progress_total,
                active=False,
                finished=True,
                force=True,
            )
            return out

        pending_actor_puts: list[Any] = []
        for _ in range(self.rollout_epoch):
            self._reset_actor_shard_buffers()
            _hs_trace(
                f"[env rank={int(self.rank)} role={self.role}] "
                f"reset start num_slots={int(self.num_slots)}"
            )
            self._put_observation_batch(
                env_channel,
                self.bootstrap_obs(),
                metrics,
                phase="bootstrap",
            )
            _hs_trace(f"[env rank={int(self.rank)} role={self.role}] reset done")

            target_chunk_steps = self._chunk_steps_per_rollout_epoch()
            chunk_steps_by_slot = [0 for _ in range(self.num_slots)]
            while any(steps < target_chunk_steps for steps in chunk_steps_by_slot):
                active_slot_ids = [
                    slot_id
                    for slot_id in range(self.num_slots)
                    if chunk_steps_by_slot[slot_id] < target_chunk_steps
                ]
                results = self._get_rollout_result_batch(
                    rollout_channel,
                    active_slot_ids,
                    metrics,
                )
                keys_csv = ",".join(result.key for result in results)
                first_step = (
                    min(
                        int(chunk_steps_by_slot[int(result.slot_id)])
                        for result in results
                    )
                    if results
                    else 0
                )
                _hs_trace(
                    f"[env rank={int(self.rank)} role={self.role}] "
                    f"step {first_step} start batch_size={len(results)} keys={keys_csv}"
                )
                next_messages: list[ObservationMsg] = []
                for result in results:
                    slot_id = int(result.slot_id)
                    apply_start = time.perf_counter()
                    shard = self.apply_rollout_result(result)
                    metrics["env/apply_step_s"] += time.perf_counter() - apply_start
                    self._buffer_actor_shard(shard)
                    chunk_steps_by_slot[slot_id] += 1
                    metrics["env/chunk_steps"] += 1.0
                    metrics["env/trajectory_chunks"] += 1.0
                    metrics["env/physical_steps"] += float(
                        self._last_apply_physical_steps
                    )
                    metrics["env/steps"] += float(self._last_apply_physical_steps)
                    metrics["env/episodes_completed"] += float(
                        self._last_apply_completed_episodes
                    )
                    metrics["env/episodes_successful"] += float(
                        self._last_apply_successful_episodes
                    )
                    metrics["env/env_crashes"] += float(
                        self._last_apply_env_crashes
                    )
                    metrics["env/env_respawns"] += float(
                        self._last_apply_env_respawns
                    )
                    self._write_interact_progress(
                        done=int(metrics["env/chunk_steps"]),
                        total=progress_total,
                        active=True,
                        finished=False,
                    )
                    if chunk_steps_by_slot[slot_id] >= target_chunk_steps:
                        put_s, emitted = self._flush_buffered_actor_shard(
                            slot_id,
                            actor_channel,
                            pending_actor_puts,
                        )
                        metrics["env/actor_put_s"] += put_s
                        metrics["env/trajectory_shards"] += float(emitted)
                        if len(pending_actor_puts) >= _ACTOR_PUT_FLUSH_EVERY:
                            metrics["env/actor_put_flush_s"] += self._flush_actor_puts(
                                pending_actor_puts
                            )
                    if chunk_steps_by_slot[slot_id] < target_chunk_steps:
                        obs = self._obs_by_slot[slot_id]
                        if obs is None:
                            raise RuntimeError("slot has no current observation")
                        next_messages.append(self._observation_msg(slot_id, obs))
                _hs_trace(
                    f"[env rank={int(self.rank)} role={self.role}] "
                    f"step {first_step} done batch_size={len(results)} keys={keys_csv}"
                )
                if next_messages:
                    self._put_observation_batch(env_channel, next_messages, metrics)
            final_bootstrap_messages: list[ObservationMsg] = []
            final_bootstrap_slot_ids: list[int] = []
            for slot_id in range(self.num_slots):
                put_s, emitted = self._flush_buffered_actor_shard(
                    slot_id,
                    actor_channel,
                    pending_actor_puts,
                )
                metrics["env/actor_put_s"] += put_s
                metrics["env/trajectory_shards"] += float(emitted)
                obs = self._obs_by_slot[slot_id]
                if obs is None:
                    continue
                if self.request_final_bootstrap:
                    message = self._observation_msg(slot_id, obs)
                    message.obs["_final_bootstrap"] = True
                    final_bootstrap_messages.append(message)
                    final_bootstrap_slot_ids.append(slot_id)
                metrics["env/episodes_flushed"] += float(
                    self._flush_partial_episode(slot_id)
                )
            if final_bootstrap_messages:
                self._put_observation_batch(
                    env_channel,
                    final_bootstrap_messages,
                    metrics,
                    phase="final_bootstrap",
                )
                self._get_rollout_result_batch(
                    rollout_channel,
                    final_bootstrap_slot_ids,
                    metrics,
                    phase="final_bootstrap",
                )
                metrics["env/final_bootstrap_requests"] += float(
                    len(final_bootstrap_messages)
                )
        metrics["env/actor_put_flush_s"] += self._flush_actor_puts(
            pending_actor_puts
        )
        metrics["env/interact_loop_s"] += time.perf_counter() - interact_start
        _hs_trace(
            f"[env rank={int(self.rank)} role={self.role}] interact done "
            f"chunk_steps={int(metrics['env/chunk_steps'])}"
        )
        self._write_interact_progress(
            done=int(metrics["env/chunk_steps"]),
            total=progress_total,
            active=False,
            finished=True,
            force=True,
        )
        return self._finalize_interact_metrics(metrics)

    def _new_interact_metrics(self) -> dict[str, float]:
        return {
            "env/chunk_steps": 0.0,
            "env/physical_steps": 0.0,
            "env/steps": 0.0,
            "env/trajectory_chunks": 0.0,
            "env/trajectory_shards": 0.0,
            "env/episodes_completed": 0.0,
            "env/episodes_successful": 0.0,
            "env/episodes_flushed": 0.0,
            "env/env_crashes": 0.0,
            "env/env_respawns": 0.0,
            "env/final_bootstrap_requests": 0.0,
            "env/channel_put_obs_s": 0.0,
            "env/rollout_get_s": 0.0,
            "env/apply_step_s": 0.0,
            "env/actor_put_s": 0.0,
            "env/actor_put_flush_s": 0.0,
            "env/interact_loop_s": 0.0,
        }

    def _finalize_interact_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        prefix = f"env/{self.role}/"
        for key, value in list(metrics.items()):
            if not key.startswith("env/"):
                continue
            short = key[len("env/") :]
            if "/" in short:
                continue
            metrics[f"{prefix}{short}"] = float(value)
        for key, value in self._collect_backend_metrics(reset=True).items():
            metrics[f"{prefix}{key}"] = float(value)
        return metrics

    def _collect_backend_metrics(self, *, reset: bool) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for env in self.envs:
            getter = getattr(env, "get_metrics", None)
            if getter is None:
                continue
            try:
                raw = getter(reset=bool(reset))
            except TypeError:
                raw = getter()
            if not isinstance(raw, dict):
                continue
            for key, value in raw.items():
                if isinstance(value, (int, float, np.number)):
                    metrics[str(key)] = metrics.get(str(key), 0.0) + float(value)
        return metrics

    def _queue_actor_shard(
        self,
        actor_channel: Channel,
        shard: TrajectoryShard,
        pending: list[Any],
    ) -> float:
        put_start = time.perf_counter()
        put_no_wait = getattr(actor_channel, "put_no_wait", None)
        if callable(put_no_wait):
            pending.append(put_no_wait(shard))
        else:
            actor_channel.put(shard)
        return float(time.perf_counter() - put_start)

    def _reset_actor_shard_buffers(self) -> None:
        self._actor_shards_by_slot = [[] for _ in range(self.num_slots)]

    def _buffer_actor_shard(self, shard: TrajectoryShard) -> None:
        slot_id = int(shard.slot_id)
        self._validate_slot(slot_id)
        self._actor_shards_by_slot[slot_id].append(shard)

    def _flush_buffered_actor_shard(
        self,
        slot_id: int,
        actor_channel: Channel,
        pending: list[Any],
    ) -> tuple[float, int]:
        self._validate_slot(slot_id)
        shards = self._actor_shards_by_slot[slot_id]
        if not shards:
            return 0.0, 0
        shard = _concat_trajectory_shards(shards)
        self._actor_shards_by_slot[slot_id] = []
        put_s = self._queue_actor_shard(actor_channel, shard, pending)
        return put_s, 1

    def _flush_actor_puts(self, pending: list[Any]) -> float:
        if not pending:
            return 0.0
        flush_start = time.perf_counter()
        works = list(pending)
        pending.clear()
        for work in works:
            wait = getattr(work, "wait", None)
            if callable(wait):
                wait()
        return float(time.perf_counter() - flush_start)

    def _can_batch_wm_slots(self) -> bool:
        if self.role != "wm_env":
            return False
        self._ensure_initialized()
        if not self._batched_env or not self.envs:
            return False
        env = self.envs[0]
        return callable(getattr(env, "step_batch", None))

    def _interact_batched_wm_slots(
        self,
        env_channel: Channel,
        rollout_channel: Channel,
        actor_channel: Channel,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        pending_actor_puts: list[Any] = []
        progress_total = self._interact_progress_total()
        for _ in range(self.rollout_epoch):
            self._reset_actor_shard_buffers()
            _hs_trace(
                f"[env rank={int(self.rank)} role={self.role}] "
                f"reset start num_slots={int(self.num_slots)}"
            )
            self._put_observation_batch(
                env_channel,
                self.bootstrap_obs(),
                metrics,
                phase="bootstrap",
            )
            _hs_trace(f"[env rank={int(self.rank)} role={self.role}] reset done")

            target_chunk_steps = self._chunk_steps_per_rollout_epoch()
            chunk_steps_by_slot = [0 for _ in range(self.num_slots)]
            while any(steps < target_chunk_steps for steps in chunk_steps_by_slot):
                active_slot_ids = [
                    slot_id
                    for slot_id in range(self.num_slots)
                    if chunk_steps_by_slot[slot_id] < target_chunk_steps
                ]
                results = self._get_rollout_result_batch(
                    rollout_channel,
                    active_slot_ids,
                    metrics,
                )
                apply_start = time.perf_counter()
                keys_csv = ",".join(result.key for result in results)
                first_step = (
                    min(
                        int(chunk_steps_by_slot[int(result.slot_id)])
                        for result in results
                    )
                    if results
                    else 0
                )
                _hs_trace(
                    f"[env rank={int(self.rank)} role={self.role}] "
                    f"step {first_step} start batch_size={len(results)} keys={keys_csv}"
                )
                applied = self._apply_wm_rollout_results_batch(results)
                metrics["env/apply_step_s"] += time.perf_counter() - apply_start
                _hs_trace(
                    f"[env rank={int(self.rank)} role={self.role}] "
                    f"step {first_step} done batch_size={len(results)} keys={keys_csv}"
                )
                for shard, shard_metrics in applied:
                    self._buffer_actor_shard(shard)
                    slot_id = int(shard.slot_id)
                    chunk_steps_by_slot[slot_id] += 1
                    metrics["env/chunk_steps"] += 1.0
                    metrics["env/trajectory_chunks"] += 1.0
                    metrics["env/physical_steps"] += float(
                        shard_metrics["physical_steps"]
                    )
                    metrics["env/steps"] += float(shard_metrics["physical_steps"])
                    metrics["env/episodes_completed"] += float(
                        shard_metrics["completed_episodes"]
                    )
                    metrics["env/episodes_successful"] += float(
                        shard_metrics["successful_episodes"]
                    )
                    self._write_interact_progress(
                        done=int(metrics["env/chunk_steps"]),
                        total=progress_total,
                        active=True,
                        finished=False,
                    )
                next_messages: list[ObservationMsg] = []
                for shard, _shard_metrics in applied:
                    slot_id = int(shard.slot_id)
                    if chunk_steps_by_slot[slot_id] >= target_chunk_steps:
                        put_s, emitted = self._flush_buffered_actor_shard(
                            slot_id,
                            actor_channel,
                            pending_actor_puts,
                        )
                        metrics["env/actor_put_s"] += put_s
                        metrics["env/trajectory_shards"] += float(emitted)
                        if len(pending_actor_puts) >= _ACTOR_PUT_FLUSH_EVERY:
                            metrics["env/actor_put_flush_s"] += self._flush_actor_puts(
                                pending_actor_puts
                            )
                    if chunk_steps_by_slot[slot_id] < target_chunk_steps:
                        obs = self._obs_by_slot[slot_id]
                        if obs is None:
                            raise RuntimeError("slot has no current observation")
                        next_messages.append(self._observation_msg(slot_id, obs))
                if next_messages:
                    self._put_observation_batch(env_channel, next_messages, metrics)
            final_bootstrap_messages: list[ObservationMsg] = []
            final_bootstrap_slot_ids: list[int] = []
            for slot_id in range(self.num_slots):
                put_s, emitted = self._flush_buffered_actor_shard(
                    slot_id,
                    actor_channel,
                    pending_actor_puts,
                )
                metrics["env/actor_put_s"] += put_s
                metrics["env/trajectory_shards"] += float(emitted)
                obs = self._obs_by_slot[slot_id]
                if obs is None:
                    continue
                if self.request_final_bootstrap:
                    message = self._observation_msg(slot_id, obs)
                    message.obs["_final_bootstrap"] = True
                    final_bootstrap_messages.append(message)
                    final_bootstrap_slot_ids.append(slot_id)
                metrics["env/episodes_flushed"] += float(
                    self._flush_partial_episode(slot_id)
                )
            if final_bootstrap_messages:
                self._put_observation_batch(
                    env_channel,
                    final_bootstrap_messages,
                    metrics,
                    phase="final_bootstrap",
                )
                self._get_rollout_result_batch(
                    rollout_channel,
                    final_bootstrap_slot_ids,
                    metrics,
                    phase="final_bootstrap",
                )
                metrics["env/final_bootstrap_requests"] += float(
                    len(final_bootstrap_messages)
                )
        metrics["env/actor_put_flush_s"] += self._flush_actor_puts(
            pending_actor_puts
        )
        _hs_trace(
            f"[env rank={int(self.rank)} role={self.role}] interact done "
            f"chunk_steps={int(metrics['env/chunk_steps'])}"
        )
        return self._finalize_interact_metrics(metrics)

    def _apply_wm_rollout_results_batch(
        self,
        results: list[RolloutResultMsg],
    ) -> list[tuple[TrajectoryShard, dict[str, float]]]:
        if not results:
            return []
        env = self._env_for_slot(int(results[0].slot_id))
        parsed: list[dict[str, Any]] = []
        action_dim: int | None = None
        for result in results:
            slot_id = int(result.slot_id)
            self._validate_slot(slot_id)
            actions_np = _as_action_chunk(result.actions)
            if int(actions_np.shape[0]) > self.num_action_chunks:
                raise ValueError(
                    "rollout action chunk length must be <= num_action_chunks "
                    f"({self.num_action_chunks})"
                )
            if action_dim is None:
                action_dim = int(actions_np.shape[-1])
            elif int(actions_np.shape[-1]) != action_dim:
                raise ValueError("all batched WM rollout chunks must share action_dim")
            parsed.append(
                {
                    "result": result,
                    "slot_id": slot_id,
                    "actions_np": actions_np,
                    "chunk_len": int(actions_np.shape[0]),
                    "rewards": np.zeros((self.num_action_chunks,), dtype=np.float32),
                    "dones": np.zeros((self.num_action_chunks,), dtype=np.bool_),
                    "completed": 0,
                    "successful": 0,
                    "physical_steps": 0,
                    "active": True,
                    "sidecars": _transition_sidecars_from_rollout(result),
                }
            )
        if action_dim is None:
            raise ValueError("cannot batch empty WM rollout results")

        for action_index in range(self.num_action_chunks):
            active = [
                item
                for item in parsed
                if bool(item["active"]) and action_index < int(item["chunk_len"])
            ]
            if not active:
                continue
            slots = [int(item["slot_id"]) for item in active]
            policy_actions = [
                np.asarray(item["actions_np"][action_index], dtype=np.float32).reshape(-1)
                for item in active
            ]
            env_actions = np.stack(
                [self._env_action_from_policy_action(action) for action in policy_actions],
                axis=0,
            ).astype(np.float32, copy=False)
            step_out = env.step_batch(env_actions, env_ids=slots)
            if len(step_out) != 5:
                raise ValueError("env.step_batch(actions, env_ids=...) must return 5 values")
            next_obs_list, rewards, terminations, truncations, infos = step_out
            for batch_index, item in enumerate(active):
                slot_id = int(item["slot_id"])
                obs = self._obs_by_slot[slot_id]
                if obs is None:
                    raise RuntimeError("bootstrap_obs() must be called before stepping")
                next_obs = dict(next_obs_list[batch_index])
                reward = float(rewards[batch_index])
                terminated = bool(terminations[batch_index])
                truncated = bool(truncations[batch_index])
                info = dict(infos[batch_index] or {})
                info.setdefault(
                    "wm_action",
                    np.asarray(env_actions[batch_index], dtype=np.float32).reshape(-1),
                )
                done = bool(terminated or truncated)
                success = bool(info.get("success", False))
                transition_obs = dict(obs)
                transition_obs.update(self._model_version_sidecars())
                transition_obs.update(dict(item["sidecars"]))
                transition = self._make_transition(
                    env,
                    transition_obs,
                    next_obs,
                    policy_actions[batch_index],
                    reward,
                    terminated,
                    truncated,
                    info,
                )
                self._episodes_by_slot[slot_id].append(transition)
                item["rewards"][action_index] = reward
                item["dones"][action_index] = done
                item["physical_steps"] = int(item["physical_steps"]) + 1
                if done:
                    item["completed"] = 1
                    item["successful"] = int(success)
                    if action_index + 1 < self.num_action_chunks:
                        item["dones"][action_index + 1 :] = True
                    self._push_replay_episode(self._episodes_by_slot[slot_id])
                    self._push_episode(self.dump, self._episodes_by_slot[slot_id])
                    self._episodes_by_slot[slot_id] = []
                    self._episode_ids_by_slot[slot_id] += 1
                    self._reset_slot(slot_id)
                    item["active"] = False
                else:
                    self._obs_by_slot[slot_id] = next_obs

        shards: list[tuple[TrajectoryShard, dict[str, float]]] = []
        for item in parsed:
            shard = self._build_trajectory_shard(
                item["result"],
                actions_np=item["actions_np"],
                rewards=item["rewards"],
                dones=item["dones"],
                action_dim=int(action_dim),
            )
            shards.append(
                (
                    shard,
                    {
                        "physical_steps": float(item["physical_steps"]),
                        "completed_episodes": float(item["completed"]),
                        "successful_episodes": float(item["successful"]),
                    },
                )
            )
        return shards

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

    def load_component_states(
        self,
        state_dicts: Mapping[str, dict[str, Any]],
        version: int,
    ) -> dict[str, float]:
        """Load multiple learner-owned component states with one worker RPC."""

        metrics: dict[str, float] = {}
        for component in ("world_model", "classifier"):
            if component not in state_dicts:
                continue
            state = dict(state_dicts.get(component, {}))
            metrics[f"sync/{component}_tensors"] = float(len(state))
            metrics[f"sync/{component}_bytes"] = float(_state_dict_nbytes(state))
            load_start = time.perf_counter()
            if component == "world_model":
                self.load_world_model_state(state, int(version))
            else:
                self.load_classifier_state(state, int(version))
            metrics[f"sync/{component}_load_s"] = float(
                time.perf_counter() - load_start
            )
        return metrics

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
        self._last_apply_env_crashes = 0
        self._last_apply_env_respawns = 0
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
            self._push_replay_episode(self._episodes_by_slot[slot_id])
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

    def _interact_progress_total(self) -> int:
        return int(self.rollout_epoch) * int(self.num_slots) * int(
            self._chunk_steps_per_rollout_epoch()
        )

    def _write_interact_progress(
        self,
        *,
        done: int,
        total: int,
        active: bool,
        finished: bool,
        force: bool = False,
    ) -> None:
        path = self._progress_path
        if path is None:
            return
        now = time.monotonic()
        if (
            not force
            and self._progress_last_write_t is not None
            and (now - self._progress_last_write_t) < self._progress_min_interval_s
        ):
            return
        payload = {
            "role": str(self.role),
            "rank": int(self.rank),
            "env_rank": int(self._rank_key()),
            "global_step": int(self.global_step),
            "done": max(0, int(done)),
            "total": max(0, int(total)),
            "active": bool(active),
            "finished": bool(finished),
            "time": float(time.time()),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, path)
        self._progress_last_write_t = now

    def _env_action_from_policy_action(self, action: Any) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_postprocess in {"", "none", "false"}:
            return action_arr
        if self.action_postprocess in {"openvla_oft", "oft"}:
            from dreamervla.runners.oft_collect_common import process_action

            return process_action(action_arr)
        raise ValueError(f"unknown env action_postprocess: {self.action_postprocess!r}")

    def _model_version_sidecars(self) -> dict[str, int]:
        world_model_version = int(self._model_versions.get("world_model", 0))
        classifier_version = int(self._model_versions.get("classifier", 0))
        return {
            "world_model_version": world_model_version,
            "wm_version": world_model_version,
            "classifier_version": classifier_version,
            "reward_or_classifier_version": classifier_version,
            "global_step": int(self.global_step),
        }

    def _flush_partial_episode(self, slot_id: int) -> int:
        self._validate_slot(slot_id)
        episode = self._episodes_by_slot[slot_id]
        if not episode:
            return 0
        flushed = [dict(step) for step in episode]
        _mark_transition_truncated(flushed[-1])
        self._push_replay_episode(flushed)
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
        except (KeyError, RuntimeError):
            return
        latents = np.asarray(raw, dtype=np.float32)
        for env in self.envs:
            setter = getattr(env, "set_initial_latents", None)
            if setter is not None:
                setter(latents)
        self._bootstrap_wm_initial_lang_embs_from_replay(sampler)
        self._bootstrap_wm_initial_proprios_from_replay(sampler)

    def _bootstrap_wm_initial_lang_embs_from_replay(self, sampler: Any) -> None:
        try:
            raw = _call_maybe_remote(
                sampler,
                self.num_slots,
                task_id=self.task_id,
                key="lang_emb",
            )
        except (KeyError, RuntimeError):
            return
        lang_embs = np.asarray(raw, dtype=np.float32)
        for env in self.envs:
            setter = getattr(env, "set_initial_lang_embs", None)
            if setter is not None:
                setter(lang_embs)

    def _bootstrap_wm_initial_proprios_from_replay(self, sampler: Any) -> None:
        try:
            raw = _call_maybe_remote(
                sampler,
                self.num_slots,
                task_id=self.task_id,
                key="proprio",
            )
        except (KeyError, RuntimeError):
            return
        proprios = np.asarray(raw, dtype=np.float32)
        for env in self.envs:
            setter = getattr(env, "set_initial_proprios", None)
            if setter is not None:
                setter(proprios)

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
        return _make_env_transition(
            env,
            obs,
            next_obs,
            action,
            reward,
            terminated,
            truncated,
            info,
        )

    @staticmethod
    def _push_episode(
        target: Any | None,
        episode: list[dict[str, Any]],
        *,
        source: str | None = None,
    ) -> None:
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

            if source is None:
                ray.get(remote(list(episode)))
            else:
                ray.get(remote(list(episode), str(source)))
        else:
            if source is None:
                add_episode(list(episode))
            else:
                add_episode(list(episode), source=str(source))

    def _push_replay_episode(self, episode: list[dict[str, Any]]) -> None:
        if not self.replay_write_enabled:
            return
        if self.role == "wm_env":
            self._push_episode(self.replay, episode, source="imagined")
            return
        self._push_episode(self.replay, episode)

    def _observation_msg(self, slot_id: int, obs: dict[str, Any]) -> ObservationMsg:
        return ObservationMsg(
            env_rank=int(self.rank) + int(self.rank_offset),
            slot_id=int(slot_id),
            task_id=int(self._task_ids_by_slot[slot_id]),
            episode_id=int(self._episode_ids_by_slot[slot_id]),
            step=int(obs.get("step", 0)),
            obs=dict(obs),
            versions=self._model_version_sidecars(),
        )

    def _rank_key(self) -> str:
        return str(int(self.rank) + int(self.rank_offset))

    def _observation_batch_msg(
        self,
        messages: list[ObservationMsg],
    ) -> ObservationBatchMsg:
        if not messages:
            raise ValueError("cannot send an empty observation batch")
        env_rank = int(self._rank_key())
        slots_seen: set[int] = set()
        for message in messages:
            if not isinstance(message, ObservationMsg):
                raise TypeError(
                    "observation batches must contain ObservationMsg items, "
                    f"got {type(message).__name__}"
                )
            if int(message.env_rank) != env_rank:
                raise ValueError(
                    "observation batch env_rank mismatch: "
                    f"got {int(message.env_rank)}, expected {env_rank}"
                )
            slot_id = int(message.slot_id)
            self._validate_slot(slot_id)
            if slot_id in slots_seen:
                raise ValueError(f"duplicate observation for slot_id {slot_id}")
            slots_seen.add(slot_id)
        return ObservationBatchMsg(env_rank=env_rank, observations=list(messages))

    def _put_observation_batch(
        self,
        env_channel: Channel,
        messages: list[ObservationMsg],
        metrics: dict[str, float],
        *,
        phase: str | None = None,
    ) -> None:
        if not messages:
            return
        batch = self._observation_batch_msg(messages)
        put_start = time.perf_counter()
        env_channel.put(batch, key=batch.key)
        metrics["env/channel_put_obs_s"] += time.perf_counter() - put_start
        keys_csv = ",".join(message.key for message in messages)
        phase_suffix = f" phase={str(phase)}" if phase is not None else ""
        _hs_trace(
            f"[env rank={int(self.rank)} role={self.role}] "
            f"send action request batch_size={len(messages)} "
            f"key={batch.key} keys={keys_csv}{phase_suffix}"
        )

    def _get_rollout_result_batch(
        self,
        rollout_channel: Channel,
        expected_slot_ids: list[int],
        metrics: dict[str, float],
        *,
        phase: str | None = None,
    ) -> list[RolloutResultMsg]:
        if not expected_slot_ids:
            return []
        expected_rank = int(self._rank_key())
        expected_slots = [int(slot_id) for slot_id in expected_slot_ids]
        expected_set = set(expected_slots)
        if len(expected_set) != len(expected_slots):
            raise ValueError("expected_slot_ids must not contain duplicates")
        for slot_id in expected_slots:
            self._validate_slot(slot_id)

        key = str(expected_rank)
        phase_suffix = f" phase={str(phase)}" if phase is not None else ""
        _hs_trace(
            f"[env rank={int(self.rank)} role={self.role}] "
            f"recv action response WAIT key={key}{phase_suffix}"
        )
        get_start = time.perf_counter()
        msg = rollout_channel.get(key=key)
        metrics["env/rollout_get_s"] += time.perf_counter() - get_start
        if not isinstance(msg, RolloutResultBatchMsg):
            raise TypeError(
                "EnvWorker expected RolloutResultBatchMsg from rollout channel, "
                f"got {type(msg).__name__}"
            )
        if int(msg.env_rank) != expected_rank:
            raise ValueError(
                "rollout result batch env_rank mismatch: "
                f"got {int(msg.env_rank)}, expected {expected_rank}"
            )

        by_slot: dict[int, RolloutResultMsg] = {}
        for result in msg.results:
            if not isinstance(result, RolloutResultMsg):
                raise TypeError(
                    "rollout result batches must contain RolloutResultMsg items, "
                    f"got {type(result).__name__}"
                )
            if int(result.env_rank) != expected_rank:
                raise ValueError(
                    "rollout result env_rank mismatch: "
                    f"got {int(result.env_rank)}, expected {expected_rank}"
                )
            slot_id = int(result.slot_id)
            self._validate_slot(slot_id)
            if slot_id in by_slot:
                raise ValueError(f"duplicate rollout result for slot_id {slot_id}")
            by_slot[slot_id] = result

        actual_set = set(by_slot)
        if actual_set != expected_set:
            raise ValueError(
                "rollout result batch slot mismatch: "
                f"got {sorted(actual_set)}, expected {sorted(expected_set)}"
            )

        results = [by_slot[slot_id] for slot_id in expected_slots]
        keys_csv = ",".join(result.key for result in results)
        _hs_trace(
            f"[env rank={int(self.rank)} role={self.role}] "
            f"recv action response batch_size={len(results)} "
            f"key={key} keys={keys_csv}{phase_suffix}"
        )
        return results

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
        request_final_bootstrap: bool = True,
        replay_write_enabled: bool = True,
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
            request_final_bootstrap=request_final_bootstrap,
            replay_write_enabled=replay_write_enabled,
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
        request_final_bootstrap: bool = True,
        replay_write_enabled: bool = False,
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
            request_final_bootstrap=request_final_bootstrap,
            replay_write_enabled=replay_write_enabled,
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
