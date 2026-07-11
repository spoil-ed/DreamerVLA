"""Trajectory-oriented EnvWorkers for the target cotrain channel topology."""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.worker import Worker
from dreamervla.utils.egl_device import (
    apply_libero_render_regime,
)
from dreamervla.workers.cotrain.handshake_trace import trace as _hs_trace
from dreamervla.workers.cotrain.messages import (
    ObservationBatchMsg,
    ObservationMsg,
    RolloutResultBatchMsg,
    RolloutResultMsg,
    TrajectoryShard,
    _pad_step_batch,
    _shard_loss_mask,
    as_tensor,
    rollout_result_batch_to_messages,
)

_ACTOR_PUT_FLUSH_EVERY = 64
_DIRECT_HIDDEN_OBS_KEYS = ("obs_embedding", "hidden", "latent")
_BATCHED_OBS_SIDECAR_KEYS = ("lang_emb",)


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


def _libero_render_gpu_pool(env_cfg: Mapping[str, Any]) -> list[int]:
    from dreamervla.runners.render_device_config import parse_device_ids

    for key in ("gpu_pool", "render_devices", "egl_device_pool"):
        devices = parse_device_ids(_plain_dict(env_cfg).get(key))
        if devices:
            return devices
    return []


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


def _trajectory_env_subprocess_main(  # noqa: ANN001
    conn,
    env_cfg: Mapping[str, Any],
    shard_id: int,
) -> None:
    """Build and step one real LIBERO env in a fresh spawned process."""

    cfg = _plain_dict(env_cfg)
    render_backend = str(cfg.get("render_backend", "osmesa")).strip().lower()
    apply_libero_render_regime(render_backend, int(shard_id), _libero_render_gpu_pool(cfg))
    env: Any | None = None
    try:
        env = _build_env_from_cfg(cfg)
        conn.send(("ready", None))
    except Exception as exc:  # noqa: BLE001 - surface child init failures
        conn.send(("error", repr(exc)))
        conn.close()
        return

    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                break
            if cmd == "reset":
                task_id, episode_id = payload
                if hasattr(env, "set_task"):
                    env.set_task(int(task_id))
                conn.send(("ok", env.reset(task_id=int(task_id), episode_id=int(episode_id))))
            elif cmd == "step":
                conn.send(("ok", env.step(payload)))
            else:
                conn.send(("error", f"unknown cmd {cmd!r}"))
    except EOFError:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            conn.send(("error", repr(exc)))
        except Exception:  # noqa: BLE001
            pass
    finally:
        if env is not None:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        conn.close()


class _SpawnedTrajectoryEnvSlot:
    """Small reset/step proxy for one spawned real-env slot."""

    def __init__(
        self,
        env_cfg: Mapping[str, Any],
        *,
        shard_id: int,
        start_timeout_s: float,
    ) -> None:
        self.env_cfg = _plain_dict(env_cfg)
        self.task_id = int(self.env_cfg.get("kwargs", {}).get("task_id", 0))
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=_trajectory_env_subprocess_main,
            args=(child_conn, self.env_cfg, int(shard_id)),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        if not parent_conn.poll(float(start_timeout_s)):
            self.close()
            raise RuntimeError(
                "spawned trajectory env timed out during init "
                f"(shard_id={int(shard_id)}, timeout={float(start_timeout_s):.0f}s)"
            )
        status, payload = parent_conn.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(f"spawned trajectory env init failed: {payload}")

    def set_task(self, task_id: int) -> None:
        self.task_id = int(task_id)

    def reset(
        self,
        *,
        task_id: int | None = None,
        episode_id: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        selected_task_id = self.task_id if task_id is None else int(task_id)
        self.task_id = int(selected_task_id)
        return self._rpc("reset", (int(selected_task_id), int(episode_id or 0)))

    def step(self, action: Any) -> tuple[Any, ...]:
        self.send_step(action)
        return self.recv_step()

    def send_step(self, action: Any) -> None:
        """Dispatch one step RPC without blocking on the result.

        Splitting ``step`` into ``send_step``/``recv_step`` lets a caller keep
        multiple slots' steps in flight at once (scatter the sends to every
        subprocess, then gather the recvs) instead of blocking on each slot in
        turn.
        """

        conn = self._conn
        if conn is None:
            raise RuntimeError("spawned trajectory env is closed")
        try:
            conn.send(("step", action))
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError("spawned trajectory env exited unexpectedly") from exc

    def recv_step(self) -> tuple[Any, ...]:
        """Block on the result of a prior ``send_step`` on this slot."""

        conn = self._conn
        if conn is None:
            raise RuntimeError("spawned trajectory env is closed")
        try:
            status, value = conn.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError("spawned trajectory env exited unexpectedly") from exc
        if status == "error":
            raise RuntimeError(f"spawned trajectory env error: {value}")
        return value

    def make_transition(
        self,
        obs: dict[str, Any],
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        done = bool(terminated or truncated)
        if "wm_action" in info:
            wm_action = np.asarray(info["wm_action"], dtype=np.float32).reshape(-1)[:7]
        else:
            wm_action = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
        policy_action = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
        if {"image", "state", "task_id", "step", "task_description"}.issubset(obs):
            return {
                "image": np.asarray(obs["image"], dtype=np.uint8),
                "state": np.asarray(obs["state"], dtype=np.float32),
                "action": wm_action.astype(np.float32, copy=False),
                "wm_action": wm_action.astype(np.float32, copy=False),
                "policy_action": policy_action.astype(np.float32, copy=False),
                "reward": np.float32(reward),
                "done": np.float32(done),
                "discount": np.float32(0.0 if terminated else 1.0),
                "is_first": bool(obs.get("is_first", False)),
                "is_terminal": bool(terminated),
                "is_last": bool(done),
                "task_id": int(obs["task_id"]),
                "step": int(obs["step"]),
                "task_description": str(obs["task_description"]),
            }
        return {
            "obs": dict(obs),
            "action": np.asarray(action, dtype=np.float32),
            "reward": float(reward),
            "done": bool(done),
            "is_terminal": bool(terminated),
            "is_last": bool(done),
            "info": dict(info or {}),
        }

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        proc = getattr(self, "_proc", None)
        if conn is not None:
            try:
                if proc is not None and proc.is_alive():
                    conn.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                conn.close()
            except OSError:
                pass
            self._conn = None
        if proc is not None:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
            self._proc = None

    def _rpc(self, cmd: str, payload: Any) -> Any:
        conn = self._conn
        if conn is None:
            raise RuntimeError("spawned trajectory env is closed")
        try:
            conn.send((cmd, payload))
            status, value = conn.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError("spawned trajectory env exited unexpectedly") from exc
        if status == "error":
            raise RuntimeError(f"spawned trajectory env error: {value}")
        return value


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


def _batch_action_chunks(action_chunks: list[np.ndarray]) -> np.ndarray:
    if not action_chunks:
        raise ValueError("cannot batch an empty action chunk list")
    first = np.asarray(action_chunks[0], dtype=np.float32)
    batch = np.empty((len(action_chunks), *first.shape), dtype=np.float32)
    batch[0] = first
    for index, action_chunk in enumerate(action_chunks[1:], start=1):
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if tuple(int(v) for v in chunk.shape) != tuple(int(v) for v in first.shape):
            raise ValueError("rollout action chunks must share shape for batching")
        batch[index] = chunk
    return batch


def _batched_hidden_payload_from_messages(
    messages: list[ObservationMsg],
) -> tuple[dict[str, Any] | None, list[ObservationMsg]]:
    if not messages:
        return None, []
    for key in _DIRECT_HIDDEN_OBS_KEYS:
        if not all(key in message.obs for message in messages):
            continue
        values = [message.obs[key] for message in messages]
        batched_value = _batch_same_shape_obs_values(values)
        if batched_value is None:
            continue
        batched_obs = {key: batched_value}
        stripped_keys = {key}
        for sidecar_key in _BATCHED_OBS_SIDECAR_KEYS:
            if not all(sidecar_key in message.obs for message in messages):
                continue
            sidecar_values = [message.obs[sidecar_key] for message in messages]
            sidecar_batch = _batch_same_shape_obs_values(sidecar_values)
            if sidecar_batch is None:
                continue
            batched_obs[sidecar_key] = sidecar_batch
            stripped_keys.add(sidecar_key)
        stripped = []
        for message in messages:
            obs = dict(message.obs)
            for stripped_key in stripped_keys:
                obs.pop(stripped_key, None)
            stripped.append(replace(message, obs=obs))
        return batched_obs, stripped
    return None, list(messages)


def _batch_same_shape_obs_values(values: list[Any]) -> Any | None:
    shapes = []
    for value in values:
        if isinstance(value, torch.Tensor):
            shapes.append(tuple(int(dim) for dim in value.shape))
        else:
            shapes.append(tuple(int(dim) for dim in np.asarray(_as_numpy(value)).shape))
    if len(set(shapes)) != 1:
        return None
    shape = shapes[0]
    if all(isinstance(value, torch.Tensor) for value in values):
        return torch.cat(
            [value.detach().cpu().reshape(1, *shape) for value in values],
            dim=0,
        )
    return np.concatenate(
        [
            np.asarray(_as_numpy(value), dtype=np.float32).reshape(1, *shape)
            for value in values
        ],
        axis=0,
    )


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


def _one_forward_input_chunk_batch(value: Any) -> torch.Tensor:
    tensor = as_tensor(value)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1, 1).detach().cpu()
    if tensor.ndim == 1:
        return tensor.reshape(1, 1, *tuple(tensor.shape)).detach().cpu()
    if int(tensor.shape[0]) == 1:
        return tensor.reshape(1, 1, *tuple(tensor.shape[1:])).detach().cpu()
    return tensor.reshape(1, 1, *tuple(tensor.shape)).detach().cpu()


def _numpy_dtype_for_torch_dtype(dtype: torch.dtype | None) -> np.dtype | None:
    if dtype is None:
        return None
    if dtype == torch.float32:
        return np.dtype(np.float32)
    if dtype == torch.float64:
        return np.dtype(np.float64)
    if dtype == torch.float16:
        return np.dtype(np.float16)
    if dtype == torch.long:
        return np.dtype(np.int64)
    if dtype == torch.int32:
        return np.dtype(np.int32)
    if dtype == torch.bool:
        return np.dtype(np.bool_)
    return None


def _chunk_value_array(
    value: Any,
    *,
    dtype: torch.dtype | None = None,
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if dtype is not None and dtype != torch.bfloat16:
            tensor = tensor.to(dtype=dtype)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        array = tensor.cpu().numpy()
    else:
        array = np.asarray(value)
    np_dtype = _numpy_dtype_for_torch_dtype(dtype)
    if np_dtype is not None:
        array = array.astype(np_dtype, copy=False)
    if array.ndim > 0 and int(array.shape[0]) == 1:
        array = np.squeeze(array, axis=0)
    return np.asarray(array)


def _cpu_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value.detach() if isinstance(value, torch.Tensor) else as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.cpu()


def _loss_mask_from_dones(dones: torch.Tensor) -> torch.Tensor:
    steps = int(dones.shape[0])
    batch_size = int(dones.shape[1])
    if steps <= 0:
        return torch.zeros((0, batch_size), dtype=torch.float32)
    if dones.ndim > 2:
        done_by_step = dones.reshape(steps, batch_size, -1).any(dim=2)
    else:
        done_by_step = dones.reshape(steps, batch_size)
    mask = torch.zeros((steps, batch_size), dtype=torch.float32)
    alive = torch.ones((batch_size,), dtype=torch.bool)
    for step in range(steps):
        mask[step] = alive.to(dtype=torch.float32)
        alive = alive & ~done_by_step[step]
    return mask


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


def _concat_uniform_slot_shards(shards: list[TrajectoryShard]) -> TrajectoryShard:
    if not shards:
        raise ValueError("cannot concatenate an empty trajectory shard buffer")
    if len(shards) == 1:
        return shards[0]
    first = shards[0]
    forward_keys = set(first.forward_inputs)
    version_keys = set(first.versions)
    has_prev_values = first.prev_values is not None
    if any(
        int(shard.env_rank) != int(first.env_rank)
        or int(shard.slot_id) != int(first.slot_id)
        or int(shard.task_id) != int(first.task_id)
        or set(shard.forward_inputs) != forward_keys
        or set(shard.versions) != version_keys
        or (shard.prev_values is not None) != has_prev_values
        or shard.loss_mask is not None
        for shard in shards
    ):
        return _concat_trajectory_shards(shards)

    actions = torch.cat([_cpu_tensor(shard.actions) for shard in shards], dim=0)
    rewards = torch.cat(
        [_cpu_tensor(shard.rewards, dtype=torch.float32) for shard in shards],
        dim=0,
    )
    dones = torch.cat(
        [_cpu_tensor(shard.dones, dtype=torch.bool) for shard in shards],
        dim=0,
    )
    prev_logprobs = torch.cat(
        [_cpu_tensor(shard.prev_logprobs, dtype=torch.float32) for shard in shards],
        dim=0,
    )
    prev_values = None
    if has_prev_values:
        prev_values = torch.cat(
            [
                _cpu_tensor(shard.prev_values, dtype=torch.float32)
                for shard in shards
                if shard.prev_values is not None
            ],
            dim=0,
        )
    return TrajectoryShard(
        env_rank=int(first.env_rank),
        slot_id=int(first.slot_id),
        task_id=int(first.task_id),
        episode_ids=list(first.episode_ids),
        actions=actions,
        rewards=rewards,
        dones=dones,
        prev_logprobs=prev_logprobs,
        prev_values=prev_values,
        forward_inputs={
            key: torch.cat(
                [_cpu_tensor(shard.forward_inputs[key]) for shard in shards],
                dim=0,
            )
            for key in sorted(forward_keys)
        },
        versions={
            key: torch.cat(
                [_cpu_tensor(shard.versions[key], dtype=torch.long) for shard in shards],
                dim=0,
            )
            for key in sorted(version_keys)
        },
        loss_mask=_loss_mask_from_dones(dones),
    )


def _concat_worker_slot_shards(shards: list[TrajectoryShard]) -> TrajectoryShard:
    if not shards:
        raise ValueError("cannot concatenate an empty trajectory shard buffer")
    if len(shards) == 1:
        return shards[0]
    first = shards[0]
    forward_keys = set(first.forward_inputs)
    version_keys = set(first.versions)
    has_prev_values = first.prev_values is not None
    for shard in shards:
        if int(shard.env_rank) != int(first.env_rank):
            raise ValueError("worker trajectory shards must share env_rank")
        if int(shard.task_id) != int(first.task_id):
            raise ValueError("worker trajectory shards must share task_id")
        if set(shard.forward_inputs) != forward_keys:
            raise ValueError("worker trajectory shards must share forward_input keys")
        if set(shard.versions) != version_keys:
            raise ValueError("worker trajectory shards must share version keys")
        if (shard.prev_values is not None) != has_prev_values:
            raise ValueError("worker trajectory shards must consistently include values")

    steps_by_shard = [int(as_tensor(shard.actions).shape[0]) for shard in shards]
    batch_sizes = [int(as_tensor(shard.actions).shape[1]) for shard in shards]
    max_steps = max(steps_by_shard)
    prev_values = None
    if has_prev_values:
        prev_values = torch.cat(
            [
                _pad_step_batch(shard.prev_values, max_steps).float()
                for shard in shards
                if shard.prev_values is not None
            ],
            dim=1,
        )
    return TrajectoryShard(
        env_rank=int(first.env_rank),
        slot_id=int(first.slot_id),
        task_id=int(first.task_id),
        episode_ids=[int(ep) for shard in shards for ep in shard.episode_ids],
        actions=torch.cat(
            [_pad_step_batch(shard.actions, max_steps) for shard in shards],
            dim=1,
        ),
        rewards=torch.cat(
            [_pad_step_batch(shard.rewards, max_steps).float() for shard in shards],
            dim=1,
        ),
        dones=torch.cat(
            [
                _pad_step_batch(shard.dones, max_steps, pad_value=True).bool()
                for shard in shards
            ],
            dim=1,
        ),
        prev_logprobs=torch.cat(
            [
                _pad_step_batch(shard.prev_logprobs, max_steps).float()
                for shard in shards
            ],
            dim=1,
        ),
        prev_values=prev_values,
        forward_inputs={
            key: torch.cat(
                [
                    _pad_step_batch(shard.forward_inputs[key], max_steps)
                    for shard in shards
                ],
                dim=1,
            )
            for key in sorted(forward_keys)
        },
        versions={
            key: torch.cat(
                [
                    _pad_step_batch(shard.versions[key], max_steps).long()
                    for shard in shards
                ],
                dim=1,
            )
            for key in sorted(version_keys)
        },
        loss_mask=torch.cat(
            [
                _shard_loss_mask(shard, max_steps, batch_size)
                for shard, batch_size in zip(shards, batch_sizes, strict=True)
            ],
            dim=1,
        ),
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


def _wm_classifier_step_success(
    env: Any,
    *,
    reward: float,
    info: dict[str, Any] | None = None,
    terminated: bool | None = None,
) -> bool:
    info = dict(info or {})
    if "success" in info:
        return bool(info["success"])
    threshold = getattr(env, "success_threshold", None)
    if threshold is not None:
        score = float(info.get("success_score", reward))
        return bool(score >= float(threshold))
    return bool(terminated) if terminated is not None else False


def _wm_classifier_success_mask(
    env: Any,
    rewards: np.ndarray,
    terminations: np.ndarray,
) -> np.ndarray:
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    terminations_arr = np.asarray(terminations, dtype=np.bool_)
    threshold = getattr(env, "success_threshold", None)
    if threshold is None:
        return terminations_arr.astype(np.bool_, copy=False)
    return np.logical_or(rewards_arr >= float(threshold), terminations_arr)


def _derive_wm_classifier_success_rates(
    metrics: dict[str, float],
    prefix: str,
) -> None:
    chunk_total = float(metrics.get(f"{prefix}classifier_total_chunks", 0.0))
    if chunk_total > 0.0:
        metrics[f"{prefix}classifier_success_rate"] = float(
            metrics.get(f"{prefix}classifier_success_chunks", 0.0) / chunk_total
        )
    traj_total = float(metrics.get(f"{prefix}classifier_total_trajectories", 0.0))
    if traj_total > 0.0:
        metrics[f"{prefix}classifier_trajectory_success_rate"] = float(
            metrics.get(f"{prefix}classifier_success_trajectories", 0.0)
            / traj_total
        )


@dataclass(frozen=True)
class _TrajectoryChunk:
    """Internal raw trajectory chunk buffered before actor-channel emission."""

    result: RolloutResultMsg
    slot_id: int
    actions_np: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    action_dim: int


@dataclass
class _SlotRollout:
    """Mutable per-slot accumulator while stepping one rollout action chunk.

    Holds everything ``apply_rollout_result`` records for a single slot so the
    physical stepping can be driven either serially (one slot at a time) or in
    lockstep across slots without changing what gets recorded.
    """

    result: RolloutResultMsg
    actions: np.ndarray
    action_dim: int
    rewards: np.ndarray
    dones: np.ndarray
    transition_sidecars: dict[str, Any]
    chunk_len: int
    completed: int = 0
    successful: int = 0
    physical_steps: int = 0
    classifier_success_chunks: int = 0
    classifier_total_chunks: int = 0
    env_crashes: int = 0
    env_respawns: int = 0
    active: bool = True


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
        self._actor_shards_by_slot: list[list[TrajectoryShard | _TrajectoryChunk]] = [
            [] for _ in range(self.num_slots)
        ]
        self._episode_ids_by_slot: list[int] = [0 for _ in range(self.num_slots)]
        self._task_ids_by_slot: list[int] = [
            self.task_id for _ in range(self.num_slots)
        ]
        self._model_versions: dict[str, int] = {}
        self.global_step = 0
        self._pending_component_states: dict[str, tuple[dict[str, Any], int]] = {}
        self._pending_classifier_threshold: float | None = None
        self._last_apply_completed_episodes = 0
        self._last_apply_successful_episodes = 0
        self._last_apply_physical_steps = 0
        self._last_apply_classifier_success_chunks = 0
        self._last_apply_classifier_total_chunks = 0
        self._last_apply_classifier_success_trajectories = 0
        self._last_apply_classifier_total_trajectories = 0
        self._last_apply_env_crashes = 0
        self._last_apply_env_respawns = 0
        self._pending_step: dict[int, tuple[Any, ...]] = {}
        self._progress_path: Path | None = None
        self._progress_min_interval_s = 5.0
        self._progress_last_write_t: float | None = None
        self._progress_last_done = 0
        self._progress_last_total = 0
        self._last_action_diagnostics: dict[str, float | int | str] | None = None
        self._prefetched_bootstrap: list[ObservationMsg] | None = None

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
        if self._use_spawn_env_slots():
            self._init_spawned_env_slots()
            self._bootstrap_wm_initial_latents_from_replay()
            self._apply_pending_component_states()
            return
        self._reject_compat_spawn_config()
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
        render_backend = str(self.env_cfg.get("render_backend", "osmesa")).strip().lower()
        apply_libero_render_regime(
            render_backend,
            int(self.local_rank),
            _libero_render_gpu_pool(self.env_cfg),
        )

    def _use_spawn_env_slots(self) -> bool:
        if self.role != "real_env":
            return False
        raw = self.env_cfg.get("spawn_env_slots", False)
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _init_spawned_env_slots(self) -> None:
        render_backend = str(self.env_cfg.get("render_backend", "osmesa")).strip().lower()
        if render_backend not in {"egl", "osmesa"}:
            raise ValueError(f"render_backend must be 'egl' or 'osmesa', got {render_backend!r}")
        start_timeout_s = float(self.env_cfg.get("spawn_env_init_timeout_s", 900.0))
        slots: list[Any] = []
        base_shard_id = int(self.local_rank) * max(1, int(self.num_slots))
        try:
            for slot_id in range(self.num_slots):
                slots.append(
                    _SpawnedTrajectoryEnvSlot(
                        self.env_cfg,
                        shard_id=base_shard_id + int(slot_id),
                        start_timeout_s=start_timeout_s,
                    )
                )
            self.envs = slots
            self._batched_env = False
        except Exception:
            for env in slots:
                close = getattr(env, "close", None)
                if close is not None:
                    close()
            raise

    def _reject_compat_spawn_config(self) -> None:
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

    def prefetch_bootstrap(self) -> dict[str, float]:
        """Reset all slots and cache the first observation batch (idempotent).

        Lets the runner overlap the next interaction's env reset with actor
        training; ``interact`` then consumes the cache instead of resetting
        inline.
        """

        self._ensure_initialized()
        if self._prefetched_bootstrap is None:
            self._prefetched_bootstrap = self.bootstrap_obs()
        return {
            f"env/{self.role}/prefetched_bootstrap_slots": float(self.num_slots)
        }

    def _consume_bootstrap_obs(self) -> list[ObservationMsg]:
        """Return the prefetched bootstrap batch if present, else reset now."""

        if self._prefetched_bootstrap is not None:
            messages = self._prefetched_bootstrap
            self._prefetched_bootstrap = None
            return messages
        return self.bootstrap_obs()

    def apply_rollout_result(self, result: RolloutResultMsg) -> TrajectoryShard:
        """Step one slot through a rollout action chunk and return a shard."""

        self._ensure_initialized()
        slot_id = int(result.slot_id)
        accum = self._new_slot_accum(result)
        for index, action in enumerate(accum.actions):
            _, reward, done, info = self._step_slot(
                slot_id,
                action,
                transition_sidecars=accum.transition_sidecars,
            )
            if self._accumulate_step(accum, index, reward, done, info, slot_id):
                break
        return self._finalize_accum(accum)

    def _new_slot_accum(self, result: RolloutResultMsg) -> _SlotRollout:
        """Validate one rollout result and build its per-slot accumulator."""

        self._validate_slot(int(result.slot_id))
        actions_np = _as_action_chunk(result.actions)
        if int(actions_np.shape[0]) > self.num_action_chunks:
            raise ValueError(
                "rollout action chunk length must be <= num_action_chunks "
                f"({self.num_action_chunks})"
            )
        return _SlotRollout(
            result=result,
            actions=actions_np,
            action_dim=int(actions_np.shape[-1]),
            rewards=np.zeros((self.num_action_chunks,), dtype=np.float32),
            dones=np.zeros((self.num_action_chunks,), dtype=np.bool_),
            transition_sidecars=_transition_sidecars_from_rollout(result),
            chunk_len=int(actions_np.shape[0]),
        )

    def _accumulate_step(
        self,
        accum: _SlotRollout,
        index: int,
        reward: float,
        done: bool,
        info: dict[str, Any],
        slot_id: int,
    ) -> bool:
        """Record one physical step into ``accum``; return True if it ended.

        Terminating a slot fills the remaining chunk with ``done`` and marks the
        accumulator inactive so lockstep stepping drops it from later steps -
        the same early-stop semantics as the serial per-slot loop.
        """

        accum.rewards[index] = float(reward)
        accum.dones[index] = bool(done)
        accum.physical_steps += 1
        if self.role == "wm_env":
            accum.classifier_total_chunks += 1
            accum.classifier_success_chunks += int(
                _wm_classifier_step_success(
                    self._env_for_slot(slot_id),
                    reward=reward,
                    info=info,
                )
            )
        accum.env_crashes += int(self._last_apply_env_crashes)
        accum.env_respawns += int(self._last_apply_env_respawns)
        if done:
            accum.completed = 1
            accum.successful = int(bool(info.get("success", False)))
            if index + 1 < self.num_action_chunks:
                accum.dones[index + 1 :] = True
            accum.active = False
            return True
        return False

    def _finalize_accum(self, accum: _SlotRollout) -> TrajectoryShard:
        """Publish the accumulated per-slot stats and build the shard."""

        self._last_apply_completed_episodes = int(accum.completed)
        self._last_apply_successful_episodes = int(accum.successful)
        self._last_apply_physical_steps = int(accum.physical_steps)
        self._last_apply_classifier_success_chunks = int(accum.classifier_success_chunks)
        self._last_apply_classifier_total_chunks = int(accum.classifier_total_chunks)
        self._last_apply_classifier_success_trajectories = int(
            accum.classifier_success_chunks > 0
        )
        self._last_apply_classifier_total_trajectories = int(
            self.role == "wm_env" and accum.classifier_total_chunks > 0
        )
        self._last_apply_env_crashes = int(accum.env_crashes)
        self._last_apply_env_respawns = int(accum.env_respawns)
        return self._build_trajectory_shard(
            accum.result,
            actions_np=accum.actions,
            rewards=accum.rewards,
            dones=accum.dones,
            action_dim=accum.action_dim,
        )

    def _should_step_slots_parallel(self, results: list[RolloutResultMsg]) -> bool:
        """Real spawned slots step in parallel; keep every other path serial."""

        if self._batched_env or len(results) <= 1:
            return False
        first_env = self._env_for_slot(int(results[0].slot_id))
        return hasattr(first_env, "send_step") and hasattr(first_env, "recv_step")

    def _step_slots_parallel(
        self,
        results: list[RolloutResultMsg],
    ) -> dict[int, _SlotRollout]:
        """Step every slot's chunk in lockstep, one physical step at a time.

        For each physical step index the sends to all still-active slots are
        scattered first, then the recvs are gathered - so all subprocesses step
        concurrently instead of one slot idling the rest (RLinf's
        ``BaseVectorEnv.step`` scatter/gather). Per-slot recording is identical
        to the serial path; only the order of the pipe RPCs changes.
        """

        accums = {int(r.slot_id): self._new_slot_accum(r) for r in results}
        max_len = max(accum.chunk_len for accum in accums.values())
        for index in range(max_len):
            active = [
                (slot_id, accum)
                for slot_id, accum in accums.items()
                if accum.active and index < accum.chunk_len
            ]
            for slot_id, accum in active:
                self._send_step_slot(
                    slot_id,
                    accum.actions[index],
                    transition_sidecars=accum.transition_sidecars,
                )
            for slot_id, accum in active:
                _, reward, done, info = self._recv_step_slot(slot_id)
                self._accumulate_step(accum, index, reward, done, info, slot_id)
        return accums

    def _build_trajectory_shard(
        self,
        result: RolloutResultMsg,
        *,
        actions_np: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        action_dim: int,
    ) -> TrajectoryShard:
        chunk_len = int(actions_np.shape[0])
        if chunk_len == int(self.num_action_chunks):
            action_pad = np.asarray(actions_np, dtype=np.float32)
        else:
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
                str(key): _one_forward_input_chunk_batch(value)
                for key, value in dict(result.forward_inputs).items()
            },
            versions={
                str(key): _one_chunk_batch(value, dtype=torch.long)
                for key, value in dict(result.versions).items()
            },
        )

    def _build_trajectory_shard_from_chunks(
        self,
        chunks: list[_TrajectoryChunk],
    ) -> TrajectoryShard:
        if not chunks:
            raise ValueError("cannot build a trajectory shard from no chunks")
        first = chunks[0]
        result = first.result
        action_dim = int(first.action_dim)
        action_blocks: list[np.ndarray] = []
        reward_blocks: list[np.ndarray] = []
        done_blocks: list[np.ndarray] = []
        prev_logprobs: list[torch.Tensor] = []
        prev_values: list[torch.Tensor] = []
        has_prev_values = result.prev_values is not None
        forward_keys = set(result.forward_inputs)
        version_keys = set(result.versions)
        for chunk in chunks:
            if (
                int(chunk.result.env_rank) != int(result.env_rank)
                or int(chunk.slot_id) != int(first.slot_id)
                or int(chunk.result.task_id) != int(result.task_id)
                or int(chunk.action_dim) != action_dim
                or set(chunk.result.forward_inputs) != forward_keys
                or set(chunk.result.versions) != version_keys
                or (chunk.result.prev_values is not None) != has_prev_values
            ):
                fallback = [
                    self._build_trajectory_shard(
                        chunk.result,
                        actions_np=chunk.actions_np,
                        rewards=chunk.rewards,
                        dones=chunk.dones,
                        action_dim=int(chunk.action_dim),
                    )
                    for chunk in chunks
                ]
                return _concat_uniform_slot_shards(fallback)
            chunk_len = int(chunk.actions_np.shape[0])
            if chunk_len == int(self.num_action_chunks):
                action_pad = np.asarray(chunk.actions_np, dtype=np.float32)
            else:
                action_pad = np.zeros(
                    (self.num_action_chunks, action_dim),
                    dtype=np.float32,
                )
                action_pad[:chunk_len] = chunk.actions_np
            action_blocks.append(action_pad)
            reward_blocks.append(np.asarray(chunk.rewards, dtype=np.float32))
            done_blocks.append(np.asarray(chunk.dones, dtype=np.bool_))
            prev_logprobs.append(
                _one_chunk_batch(chunk.result.prev_logprobs, dtype=torch.float32)
            )
            if has_prev_values and chunk.result.prev_values is not None:
                prev_values.append(
                    _one_chunk_batch(chunk.result.prev_values, dtype=torch.float32)
                )

        actions = torch.as_tensor(
            np.stack(action_blocks, axis=0),
            dtype=torch.float32,
        ).view(len(chunks), 1, self.num_action_chunks, action_dim)
        rewards = torch.as_tensor(
            np.stack(reward_blocks, axis=0),
            dtype=torch.float32,
        ).view(len(chunks), 1, self.num_action_chunks)
        dones = torch.as_tensor(
            np.stack(done_blocks, axis=0),
            dtype=torch.bool,
        ).view(len(chunks), 1, self.num_action_chunks)
        return TrajectoryShard(
            env_rank=int(result.env_rank),
            slot_id=int(first.slot_id),
            task_id=int(result.task_id),
            episode_ids=[int(result.episode_id)],
            actions=actions,
            rewards=rewards,
            dones=dones,
            prev_logprobs=torch.cat(prev_logprobs, dim=0),
            prev_values=torch.cat(prev_values, dim=0) if has_prev_values else None,
            forward_inputs={
                str(key): torch.cat(
                    [
                        _one_forward_input_chunk_batch(
                            chunk.result.forward_inputs[key]
                        )
                        for chunk in chunks
                    ],
                    dim=0,
                )
                for key in sorted(forward_keys)
            },
            versions={
                str(key): torch.cat(
                    [
                        _one_chunk_batch(
                            chunk.result.versions[key],
                            dtype=torch.long,
                        )
                        for chunk in chunks
                    ],
                    dim=0,
                )
                for key in sorted(version_keys)
            },
            loss_mask=_loss_mask_from_dones(dones),
        )

    def _build_worker_trajectory_shard_from_slot_chunks(
        self,
        slot_chunks: list[list[_TrajectoryChunk]],
    ) -> TrajectoryShard:
        if not slot_chunks:
            raise ValueError("cannot build a worker trajectory shard from no slots")
        first_chunks = slot_chunks[0]
        if not first_chunks:
            raise ValueError("cannot build a worker trajectory shard from empty chunks")
        chunk_count = len(first_chunks)
        first = first_chunks[0]
        result = first.result
        action_dim = int(first.action_dim)
        has_prev_values = result.prev_values is not None
        forward_keys = set(result.forward_inputs)
        version_keys = set(result.versions)
        for chunks in slot_chunks:
            if len(chunks) != chunk_count:
                raise ValueError("worker slot chunk buffers must have matching lengths")
            for chunk in chunks:
                if (
                    int(chunk.result.env_rank) != int(result.env_rank)
                    or int(chunk.result.task_id) != int(result.task_id)
                    or int(chunk.action_dim) != action_dim
                    or set(chunk.result.forward_inputs) != forward_keys
                    or set(chunk.result.versions) != version_keys
                    or (chunk.result.prev_values is not None) != has_prev_values
                ):
                    raise ValueError("worker slot chunk buffers are not batch-compatible")

        batch_size = len(slot_chunks)
        actions_np = np.empty(
            (chunk_count, batch_size, self.num_action_chunks, action_dim),
            dtype=np.float32,
        )
        rewards_np = np.empty(
            (chunk_count, batch_size, self.num_action_chunks),
            dtype=np.float32,
        )
        dones_np = np.empty(
            (chunk_count, batch_size, self.num_action_chunks),
            dtype=np.bool_,
        )
        for step in range(chunk_count):
            for slot_index, chunks in enumerate(slot_chunks):
                chunk = chunks[step]
                chunk_len = int(chunk.actions_np.shape[0])
                if chunk_len == int(self.num_action_chunks):
                    actions_np[step, slot_index] = np.asarray(
                        chunk.actions_np,
                        dtype=np.float32,
                    )
                else:
                    actions_np[step, slot_index].fill(0.0)
                    actions_np[step, slot_index, :chunk_len] = chunk.actions_np
                rewards_np[step, slot_index] = np.asarray(
                    chunk.rewards,
                    dtype=np.float32,
                )
                dones_np[step, slot_index] = np.asarray(chunk.dones, dtype=np.bool_)

        actions = torch.as_tensor(actions_np, dtype=torch.float32)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32)
        dones = torch.as_tensor(dones_np, dtype=torch.bool)

        def stack_result_value(
            getter: Any,
            *,
            dtype: torch.dtype | None = None,
        ) -> torch.Tensor:
            values = [
                _chunk_value_array(getter(chunks[step].result), dtype=dtype)
                for step in range(chunk_count)
                for chunks in slot_chunks
            ]
            value_shape = values[0].shape
            if value_shape:
                stacked = np.concatenate(
                    [value.reshape(1, *value_shape) for value in values],
                    axis=0,
                )
            else:
                stacked = np.asarray(values)
            tensor = torch.as_tensor(
                stacked.reshape(chunk_count, batch_size, *value_shape)
            )
            return tensor.to(dtype=dtype) if dtype is not None else tensor

        prev_values = None
        if has_prev_values:
            prev_values = stack_result_value(
                lambda result: result.prev_values,
                dtype=torch.float32,
            )
        return TrajectoryShard(
            env_rank=int(result.env_rank),
            slot_id=int(first.slot_id),
            task_id=int(result.task_id),
            episode_ids=[int(chunks[0].result.episode_id) for chunks in slot_chunks],
            actions=actions,
            rewards=rewards,
            dones=dones,
            prev_logprobs=stack_result_value(
                lambda result: result.prev_logprobs,
                dtype=torch.float32,
            ),
            prev_values=prev_values,
            forward_inputs={
                str(key): (
                    actions
                    if str(key) == "action"
                    else stack_result_value(
                        lambda result, key=key: result.forward_inputs[key]
                    )
                )
                for key in sorted(forward_keys)
            },
            versions={
                str(key): stack_result_value(
                    lambda result, key=key: result.versions[key],
                    dtype=torch.long,
                )
                for key in sorted(version_keys)
            },
            loss_mask=_loss_mask_from_dones(dones),
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
            metrics=metrics,
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
                metrics=out,
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
                self._consume_bootstrap_obs(),
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
                parallel_accums: dict[int, _SlotRollout] | None = None
                if self._should_step_slots_parallel(results):
                    apply_start = time.perf_counter()
                    self._ensure_initialized()
                    parallel_accums = self._step_slots_parallel(results)
                    metrics["env/apply_step_s"] += time.perf_counter() - apply_start
                for result in results:
                    slot_id = int(result.slot_id)
                    if parallel_accums is not None:
                        shard = self._finalize_accum(parallel_accums[slot_id])
                    else:
                        apply_start = time.perf_counter()
                        shard = self.apply_rollout_result(result)
                        metrics["env/apply_step_s"] += (
                            time.perf_counter() - apply_start
                        )
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
                    self._add_wm_classifier_metrics(
                        metrics,
                        success_chunks=self._last_apply_classifier_success_chunks,
                        total_chunks=self._last_apply_classifier_total_chunks,
                        success_trajectories=(
                            self._last_apply_classifier_success_trajectories
                        ),
                        total_trajectories=(
                            self._last_apply_classifier_total_trajectories
                        ),
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
                        metrics=metrics,
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
            metrics=metrics,
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
        _derive_wm_classifier_success_rates(metrics, "env/")
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

    def _add_wm_classifier_metrics(
        self,
        metrics: dict[str, float],
        *,
        success_chunks: float,
        total_chunks: float,
        success_trajectories: float,
        total_trajectories: float,
    ) -> None:
        if self.role != "wm_env":
            return
        for key, value in {
            "env/classifier_success_chunks": success_chunks,
            "env/classifier_total_chunks": total_chunks,
            "env/classifier_success_trajectories": success_trajectories,
            "env/classifier_total_trajectories": total_trajectories,
        }.items():
            metrics[key] = float(metrics.get(key, 0.0) + float(value))

    def _queue_actor_shard(
        self,
        actor_channel: Channel,
        shard: TrajectoryShard,
        pending: list[Any],
    ) -> float:
        put_start = time.perf_counter()
        put_no_wait = getattr(actor_channel, "put_no_wait", None)
        if callable(put_no_wait):
            pending.append(put_no_wait(shard, key=str(self.role)))
        else:
            actor_channel.put(shard, key=str(self.role))
        return float(time.perf_counter() - put_start)

    def _reset_actor_shard_buffers(self) -> None:
        self._actor_shards_by_slot = [[] for _ in range(self.num_slots)]

    def _buffer_actor_shard(self, shard: TrajectoryShard | _TrajectoryChunk) -> None:
        if not self._emit_actor_trajectories():
            return
        slot_id = int(shard.slot_id)
        self._validate_slot(slot_id)
        self._actor_shards_by_slot[slot_id].append(shard)

    def _materialize_buffered_actor_shard(
        self,
        slot_id: int,
    ) -> TrajectoryShard | None:
        self._validate_slot(slot_id)
        shards = self._actor_shards_by_slot[slot_id]
        if not shards:
            return None
        if all(isinstance(shard, _TrajectoryChunk) for shard in shards):
            shard = self._build_trajectory_shard_from_chunks(
                [shard for shard in shards if isinstance(shard, _TrajectoryChunk)]
            )
        else:
            materialized: list[TrajectoryShard] = []
            for item in shards:
                if isinstance(item, TrajectoryShard):
                    materialized.append(item)
                else:
                    materialized.append(self._build_trajectory_shard_from_chunks([item]))
            shard = _concat_uniform_slot_shards(materialized)
        self._actor_shards_by_slot[slot_id] = []
        return shard

    def _flush_buffered_actor_shard(
        self,
        slot_id: int,
        actor_channel: Channel,
        pending: list[Any],
    ) -> tuple[float, int]:
        shard = self._materialize_buffered_actor_shard(slot_id)
        if shard is None:
            return 0.0, 0
        put_s = self._queue_actor_shard(actor_channel, shard, pending)
        return put_s, 1

    def _flush_buffered_actor_slot_batch(
        self,
        slot_ids: list[int],
        actor_channel: Channel,
        pending: list[Any],
    ) -> tuple[float, int]:
        chunk_buffers: list[list[_TrajectoryChunk]] = []
        chunk_slot_ids: list[int] = []
        can_direct_materialize = True
        for slot_id in slot_ids:
            self._validate_slot(int(slot_id))
            shards = self._actor_shards_by_slot[int(slot_id)]
            if not shards:
                continue
            if not all(isinstance(shard, _TrajectoryChunk) for shard in shards):
                can_direct_materialize = False
                break
            chunk_buffers.append(
                [shard for shard in shards if isinstance(shard, _TrajectoryChunk)]
            )
            chunk_slot_ids.append(int(slot_id))
        if can_direct_materialize and chunk_buffers:
            try:
                shard = self._build_worker_trajectory_shard_from_slot_chunks(chunk_buffers)
            except ValueError:
                pass
            else:
                for slot_id in chunk_slot_ids:
                    self._actor_shards_by_slot[slot_id] = []
                put_s = self._queue_actor_shard(actor_channel, shard, pending)
                return put_s, 1

        materialized: list[TrajectoryShard] = []
        for slot_id in slot_ids:
            shard = self._materialize_buffered_actor_shard(int(slot_id))
            if shard is not None:
                materialized.append(shard)
        if not materialized:
            return 0.0, 0
        shard = _concat_worker_slot_shards(materialized)
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
                self._consume_bootstrap_obs(),
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
                    self._add_wm_classifier_metrics(
                        metrics,
                        success_chunks=shard_metrics.get(
                            "classifier_success_chunks",
                            0.0,
                        ),
                        total_chunks=shard_metrics.get(
                            "classifier_total_chunks",
                            0.0,
                        ),
                        success_trajectories=shard_metrics.get(
                            "classifier_success_trajectories",
                            0.0,
                        ),
                        total_trajectories=shard_metrics.get(
                            "classifier_total_trajectories",
                            0.0,
                        ),
                    )
                    self._write_interact_progress(
                        done=int(metrics["env/chunk_steps"]),
                        total=progress_total,
                        active=True,
                        finished=False,
                        metrics=metrics,
                    )
                next_messages: list[ObservationMsg] = []
                for shard, _shard_metrics in applied:
                    slot_id = int(shard.slot_id)
                    if chunk_steps_by_slot[slot_id] < target_chunk_steps:
                        obs = self._obs_by_slot[slot_id]
                        if obs is None:
                            raise RuntimeError("slot has no current observation")
                        next_messages.append(self._observation_msg(slot_id, obs))
                if next_messages:
                    self._put_observation_batch(env_channel, next_messages, metrics)
            put_s, emitted = self._flush_buffered_actor_slot_batch(
                list(range(self.num_slots)),
                actor_channel,
                pending_actor_puts,
            )
            metrics["env/actor_put_s"] += put_s
            metrics["env/trajectory_shards"] += float(emitted)
            if len(pending_actor_puts) >= _ACTOR_PUT_FLUSH_EVERY:
                metrics["env/actor_put_flush_s"] += self._flush_actor_puts(
                    pending_actor_puts
                )
            final_bootstrap_messages: list[ObservationMsg] = []
            final_bootstrap_slot_ids: list[int] = []
            for slot_id in range(self.num_slots):
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
    ) -> list[tuple[TrajectoryShard | _TrajectoryChunk, dict[str, float]]]:
        if not results:
            return []
        env = self._env_for_slot(int(results[0].slot_id))
        parsed: list[dict[str, Any]] = []
        action_dim: int | None = None
        collect_transitions = self._collect_episode_transitions()
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
                    "classifier_success_chunks": 0,
                    "classifier_total_chunks": 0,
                    "physical_steps": 0,
                    "active": True,
                    "sidecars": (
                        _transition_sidecars_from_rollout(result)
                        if collect_transitions
                        else {}
                    ),
                }
            )
        if action_dim is None:
            raise ValueError("cannot batch empty WM rollout results")
        if (
            not collect_transitions
            and callable(getattr(env, "chunk_step_batch", None))
            and all(int(item["chunk_len"]) == self.num_action_chunks for item in parsed)
        ):
            return self._apply_wm_rollout_results_chunk_batch(
                env,
                parsed,
                action_dim=int(action_dim),
            )

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
                if collect_transitions:
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
                item["classifier_total_chunks"] = (
                    int(item["classifier_total_chunks"]) + 1
                )
                item["classifier_success_chunks"] = int(
                    item["classifier_success_chunks"]
                ) + int(
                    _wm_classifier_step_success(
                        env,
                        reward=reward,
                        info=info,
                        terminated=terminated,
                    )
                )
                if done:
                    item["completed"] = 1
                    item["successful"] = int(success)
                    if action_index + 1 < self.num_action_chunks:
                        item["dones"][action_index + 1 :] = True
                    if collect_transitions:
                        self._push_replay_episode(self._episodes_by_slot[slot_id])
                        self._push_episode(self.dump, self._episodes_by_slot[slot_id])
                    self._episodes_by_slot[slot_id] = []
                    self._episode_ids_by_slot[slot_id] += 1
                    self._reset_slot(slot_id)
                    item["active"] = False
                else:
                    self._obs_by_slot[slot_id] = next_obs

        shards: list[tuple[TrajectoryShard | _TrajectoryChunk, dict[str, float]]] = []
        for item in parsed:
            shard = _TrajectoryChunk(
                result=item["result"],
                slot_id=int(item["slot_id"]),
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
                        "classifier_success_chunks": float(
                            item["classifier_success_chunks"]
                        ),
                        "classifier_total_chunks": float(
                            item["classifier_total_chunks"]
                        ),
                        "classifier_success_trajectories": float(
                            int(item["classifier_success_chunks"]) > 0
                        ),
                        "classifier_total_trajectories": float(
                            int(item["classifier_total_chunks"]) > 0
                        ),
                    },
                )
            )
        return shards

    def _apply_wm_rollout_results_chunk_batch(
        self,
        env: Any,
        parsed: list[dict[str, Any]],
        *,
        action_dim: int,
    ) -> list[tuple[TrajectoryShard | _TrajectoryChunk, dict[str, float]]]:
        slots = [int(item["slot_id"]) for item in parsed]
        policy_action_chunks = [
            np.asarray(item["actions_np"], dtype=np.float32) for item in parsed
        ]
        if self.action_postprocess in {"", "none", "false"}:
            env_actions = _batch_action_chunks(policy_action_chunks)
        elif self.action_postprocess in {"openvla_oft", "oft"}:
            from dreamervla.runners.oft_collect_common import process_action_batch

            env_actions = process_action_batch(_batch_action_chunks(policy_action_chunks))
        else:
            first_chunk = policy_action_chunks[0]
            env_actions = np.empty(
                (len(policy_action_chunks), *first_chunk.shape),
                dtype=np.float32,
            )
            for chunk_index, action_chunk in enumerate(policy_action_chunks):
                if tuple(int(v) for v in action_chunk.shape) != tuple(
                    int(v) for v in first_chunk.shape
                ):
                    raise ValueError(
                        "rollout action chunks must share shape for batching"
                    )
                for action_index, action in enumerate(action_chunk):
                    env_actions[chunk_index, action_index] = (
                        self._env_action_from_policy_action(action)
                    )
        step_out = env.chunk_step_batch(env_actions, env_ids=slots)
        if len(step_out) != 5:
            raise ValueError(
                "env.chunk_step_batch(actions, env_ids=...) must return 5 values"
            )
        next_obs_list, rewards, terminations, truncations, infos = step_out
        rewards_arr = np.asarray(rewards, dtype=np.float32).reshape(
            len(parsed),
            self.num_action_chunks,
        )
        terminations_arr = np.asarray(terminations, dtype=np.bool_).reshape(
            len(parsed),
            self.num_action_chunks,
        )
        truncations_arr = np.asarray(truncations, dtype=np.bool_).reshape(
            len(parsed),
            self.num_action_chunks,
        )
        done_arr = np.logical_or(terminations_arr, truncations_arr)
        has_nonfinal_done = (
            bool(done_arr[:, :-1].any()) if int(self.num_action_chunks) > 1 else False
        )
        shards: list[tuple[TrajectoryShard | _TrajectoryChunk, dict[str, float]]] = []
        for batch_index, item in enumerate(parsed):
            slot_id = int(item["slot_id"])
            next_obs = dict(next_obs_list[batch_index])
            info = dict(infos[batch_index] or {})
            done_values = done_arr[batch_index]
            item["rewards"][:] = rewards_arr[batch_index]
            item["dones"][:] = done_values
            classifier_success_mask = _wm_classifier_success_mask(
                env,
                rewards_arr[batch_index],
                terminations_arr[batch_index],
            )
            if has_nonfinal_done:
                done_indices = np.flatnonzero(done_values)
                physical_steps = (
                    int(done_indices[0]) + 1
                    if len(done_indices) > 0
                    else int(self.num_action_chunks)
                )
                if len(done_indices) > 0:
                    item["dones"][int(done_indices[0]) :] = True
                completed = bool(len(done_indices) > 0)
                successful = bool(info.get("success", False)) or bool(
                    terminations_arr[batch_index].any()
                )
            else:
                physical_steps = int(self.num_action_chunks)
                completed = bool(done_values[-1])
                successful = bool(info.get("success", False)) or bool(
                    terminations_arr[batch_index, -1]
                )
            classifier_total_chunks = int(physical_steps)
            classifier_success_chunks = int(
                classifier_success_mask[:classifier_total_chunks].sum()
            )
            if completed:
                self._episodes_by_slot[slot_id] = []
                self._episode_ids_by_slot[slot_id] += 1
                self._reset_slot(slot_id)
            else:
                self._obs_by_slot[slot_id] = next_obs
            shard = _TrajectoryChunk(
                result=item["result"],
                slot_id=slot_id,
                actions_np=item["actions_np"],
                rewards=item["rewards"],
                dones=item["dones"],
                action_dim=int(action_dim),
            )
            shards.append(
                (
                    shard,
                    {
                        "physical_steps": float(physical_steps),
                        "completed_episodes": float(completed),
                        "successful_episodes": float(successful),
                        "classifier_success_chunks": float(classifier_success_chunks),
                        "classifier_total_chunks": float(classifier_total_chunks),
                        "classifier_success_trajectories": float(
                            classifier_success_chunks > 0
                        ),
                        "classifier_total_trajectories": 1.0,
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
        state_dicts: Mapping[str, Any],
        version: int,
    ) -> dict[str, float]:
        """Load multiple learner-owned component states with one worker RPC."""

        metrics: dict[str, float] = {}
        if "classifier_threshold" in state_dicts:
            threshold = float(state_dicts["classifier_threshold"])
            self.set_classifier_threshold(threshold)
            metrics["sync/classifier_threshold"] = threshold
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
        if self._pending_classifier_threshold is not None:
            self.set_classifier_threshold(float(self._pending_classifier_threshold))

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

    def set_classifier_threshold(self, threshold: float) -> None:
        value = float(threshold)
        self._pending_classifier_threshold = value
        for env in self.envs:
            setter = getattr(env, "set_success_threshold", None)
            if callable(setter):
                setter(value)
                continue
            elif hasattr(env, "success_threshold"):
                env.success_threshold = value
                continue
            if self.role == "wm_env":
                raise TypeError(
                    "WMEnvWorker env "
                    f"{type(env).__name__} must expose set_success_threshold() "
                    "or success_threshold"
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
        env, obs, policy_action, env_action = self._prepare_step(slot_id, action)
        if self._batched_env:
            step_out = env.step_slot(slot_id, env_action)
        else:
            step_out = env.step(env_action)
        return self._finish_step(
            slot_id,
            env,
            obs,
            policy_action,
            env_action,
            step_out,
            transition_sidecars,
        )

    def _prepare_step(
        self,
        slot_id: int,
        action: Any,
    ) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray]:
        self._validate_slot(slot_id)
        self._last_apply_env_crashes = 0
        self._last_apply_env_respawns = 0
        env = self._env_for_slot(slot_id)
        obs = self._obs_by_slot[slot_id]
        if obs is None:
            raise RuntimeError("bootstrap_obs() must be called before stepping")
        policy_action = np.asarray(action, dtype=np.float32).reshape(-1)
        env_action = self._env_action_from_policy_action(policy_action)
        self._record_action_diagnostics(slot_id, policy_action, env_action)
        return env, obs, policy_action, env_action

    def _send_step_slot(
        self,
        slot_id: int,
        action: Any,
        *,
        transition_sidecars: dict[str, Any] | None = None,
    ) -> None:
        """Scatter half of a lockstep step: dispatch the RPC, defer the recv."""

        env, obs, policy_action, env_action = self._prepare_step(slot_id, action)
        env.send_step(env_action)
        self._pending_step[slot_id] = (
            env,
            obs,
            policy_action,
            env_action,
            transition_sidecars,
        )

    def _recv_step_slot(
        self,
        slot_id: int,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Gather half of a lockstep step: block on the deferred RPC and record."""

        env, obs, policy_action, env_action, transition_sidecars = (
            self._pending_step.pop(slot_id)
        )
        step_out = env.recv_step()
        return self._finish_step(
            slot_id,
            env,
            obs,
            policy_action,
            env_action,
            step_out,
            transition_sidecars,
        )

    def _finish_step(
        self,
        slot_id: int,
        env: Any,
        obs: dict[str, Any],
        policy_action: np.ndarray,
        env_action: np.ndarray,
        step_out: tuple[Any, ...],
        transition_sidecars: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
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
        collect_transitions = self._collect_episode_transitions()
        if collect_transitions:
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
            if collect_transitions:
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
        metrics: Mapping[str, float] | None = None,
        force: bool = False,
    ) -> None:
        path = self._progress_path
        if path is None:
            return
        self._progress_last_done = max(0, int(done))
        self._progress_last_total = max(0, int(total))
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
        if self._last_action_diagnostics is not None:
            payload["last_action"] = dict(self._last_action_diagnostics)
        if self.role == "wm_env":
            metric_values = metrics or {}
            for key in (
                "classifier_success_chunks",
                "classifier_total_chunks",
                "classifier_success_trajectories",
                "classifier_total_trajectories",
            ):
                payload[key] = max(
                    0,
                    int(float(metric_values.get(f"env/{key}", 0.0))),
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, path)
        self._progress_last_write_t = now

    def _record_action_diagnostics(
        self,
        slot_id: int,
        policy_action: np.ndarray,
        env_action: np.ndarray,
    ) -> None:
        """Persist the last real-env action range before native rendering code."""

        if self.role != "real_env":
            return
        policy = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        env = np.asarray(env_action, dtype=np.float32).reshape(-1)
        self._last_action_diagnostics = {
            "slot_id": int(slot_id),
            "policy_min": float(np.min(policy)),
            "policy_max": float(np.max(policy)),
            "policy_absmax": float(np.max(np.abs(policy))),
            "policy_dim": int(policy.size),
            "env_min": float(np.min(env)),
            "env_max": float(np.max(env)),
            "env_absmax": float(np.max(np.abs(env))),
            "env_dim": int(env.size),
            "action_postprocess": str(self.action_postprocess),
            "time": float(time.time()),
        }
        if self._progress_path is not None and self._debug_action_diagnostics():
            self._write_interact_progress(
                done=self._progress_last_done,
                total=self._progress_last_total,
                active=True,
                finished=False,
                force=True,
            )

    def _debug_action_diagnostics(self) -> bool:
        raw = self.env_cfg.get("debug_action_diagnostics", False)
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _env_action_from_policy_action(self, action: Any) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if not np.isfinite(action_arr).all():
            raise ValueError(
                "non-finite policy action received by "
                f"{self.role} rank={int(self.local_rank)}"
            )
        if self.action_postprocess in {"", "none", "false"}:
            return action_arr
        if self.action_postprocess in {"openvla_oft", "oft"}:
            from dreamervla.runners.oft_collect_common import process_action

            env_action = process_action(action_arr)
            if not np.isfinite(env_action).all():
                raise ValueError(
                    "non-finite env action after postprocess for "
                    f"{self.role} rank={int(self.local_rank)}"
                )
            return env_action
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

    def _collect_episode_transitions(self) -> bool:
        if self.dump is not None:
            return True
        if self.replay is None:
            return False
        if self.role == "wm_env" and not self.replay_write_enabled:
            return False
        return True

    def _emit_actor_trajectories(self) -> bool:
        override = self.env_cfg.get("emit_actor_trajectories")
        if override is not None:
            return bool(override)
        return self.role == "wm_env"

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
        batched_obs, batch_messages = _batched_hidden_payload_from_messages(messages)
        return ObservationBatchMsg(
            env_rank=env_rank,
            observations=batch_messages,
            batched_obs=batched_obs,
        )

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
        for result in rollout_result_batch_to_messages(msg):
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
        for result in results:
            if "hidden" in result.forward_inputs:
                continue
            obs = self._obs_by_slot[int(result.slot_id)]
            if obs is None:
                continue
            for obs_key in _DIRECT_HIDDEN_OBS_KEYS:
                if obs_key in obs:
                    value = obs[obs_key]
                    if isinstance(value, torch.Tensor):
                        hidden = value.detach().to(dtype=torch.float32, device="cpu")
                    else:
                        hidden = torch.as_tensor(np.asarray(value, dtype=np.float32))
                    result.forward_inputs["hidden"] = hidden.reshape(1, -1)
                    break
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
