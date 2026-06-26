"""Ray InferenceWorker for batched online rollout action selection."""

from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np
import torch

from dreamervla.runners.action_chunk_queue import ActionChunkQueue
from dreamervla.scheduler.worker import Worker
from dreamervla.workers.inference.rollout_contract import RolloutBatchOutput


class InferenceWorker(Worker):
    """Hold encoder, world model, and policy for rollout inference."""

    def __init__(
        self,
        model_cfg: dict[str, Any],
        init_ckpt: dict[str, Any],
        num_envs: int,
    ) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.init_ckpt = dict(init_ckpt)
        self.num_envs = int(num_envs)
        configured_device = str(self.model_cfg.get("device", self.device))
        if configured_device.lower() in {"", "auto"}:
            configured_device = self.device
        self.torch_device = torch.device(configured_device)
        self.encoder: Any | None = None
        self.world_model: torch.nn.Module | None = None
        self.policy: torch.nn.Module | None = None
        self.state: list[dict[str, Any]] = []
        self.action_dim = _cfg_int(
            self.model_cfg,
            (
                ("action_dim",),
                ("policy", "kwargs", "action_dim"),
                ("world_model", "kwargs", "action_dim"),
            ),
            default=7,
        )
        self.action_steps = _cfg_int(
            self.model_cfg,
            (
                ("action_steps",),
                ("policy", "kwargs", "time_horizon"),
                ("world_model", "kwargs", "chunk_size"),
            ),
            default=1,
        )
        self.action_queues = [
            ActionChunkQueue(action_dim=self.action_dim, action_steps=self.action_steps)
            for _ in range(self.num_envs)
        ]

    def init(self) -> None:
        self.encoder = _build_from_cfg(self.model_cfg["encoder"])
        self.world_model = _build_from_cfg(self.model_cfg["world_model"]).to(
            self.torch_device
        )
        self.policy = _build_from_cfg(self.model_cfg["policy"]).to(self.torch_device)
        self._load_initial_weights()
        for module in (self.encoder, self.world_model, self.policy):
            if hasattr(module, "eval"):
                module.eval()
        self.state = [self._empty_state() for _ in range(self.num_envs)]

    def reset_states(self, env_ids: list[int]) -> None:
        for env_id in env_ids:
            self.state[int(env_id)] = self._empty_state()
            self.action_queues[int(env_id)].clear()

    @torch.no_grad()
    def forward_batch(
        self,
        obs_batch: list[dict[str, Any]],
        env_ids: list[int],
    ) -> dict[str, list[Any]]:
        if len(obs_batch) != len(env_ids):
            raise ValueError("obs_batch and env_ids must have the same length")
        encoder = self._encoder()
        world_model = self._world_model()
        policy = self._policy()
        encode_start = time.perf_counter()
        obs_embedding = _encode_batch(encoder, obs_batch).to(self.torch_device).float()
        encode_s = time.perf_counter() - encode_start

        wm_start = time.perf_counter()
        latents: list[torch.Tensor] = []
        for idx, env_id_raw in enumerate(env_ids):
            env_id = int(env_id_raw)
            item = self.state[env_id]
            hidden_i = obs_embedding[idx : idx + 1]
            is_first = (
                bool(obs_batch[idx].get("is_first", False))
                or item["latent"] is None
            )
            if is_first:
                latent = world_model({"mode": "encode_latent", "hidden": hidden_i})
            else:
                latent = world_model(
                    {
                        "mode": "observe_next",
                        "latent": item["latent"],
                        "hidden": hidden_i,
                        "actions": item["prev_action"],
                        "is_first": False,
                    }
                )
            self.state[env_id]["latent"] = _detach_tensor(latent)
            latents.append(latent)

        refill_positions = [
            idx
            for idx, env_id_raw in enumerate(env_ids)
            if not self.action_queues[int(env_id_raw)].has_pending
        ]
        if refill_positions:
            latent_batch = _concat_structures([latents[idx] for idx in refill_positions])
            feat = world_model({"mode": "actor_input", "latent": latent_batch}).float()
        else:
            feat = None
        world_model_s = time.perf_counter() - wm_start
        policy_start = time.perf_counter()
        if feat is not None:
            action_chunk, _log_prob, _extra = policy(
                {
                    "mode": "sample",
                    "hidden": feat,
                    "deterministic": False,
                    "return_chunk": True,
                }
            )
            action_chunks_np = (
                action_chunk.reshape(action_chunk.shape[0], -1, action_chunk.shape[-1])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            for row, idx in enumerate(refill_positions):
                env_id = int(env_ids[idx])
                self.action_queues[env_id].refill(action_chunks_np[row])
        policy_s = time.perf_counter() - policy_start

        actions_np: list[np.ndarray] = []
        for idx, env_id_raw in enumerate(env_ids):
            env_id = int(env_id_raw)
            action_np = self.action_queues[env_id].pop()
            actions_np.append(action_np.astype(np.float32, copy=False))
            prev_action = torch.from_numpy(action_np).to(
                self.torch_device, dtype=obs_embedding.dtype
            )
            self.state[env_id]["prev_action"] = prev_action.unsqueeze(0).detach()
            self.state[env_id]["is_first"] = False

        obs_embedding_np = obs_embedding.detach().cpu().numpy().astype(np.float32)
        out = RolloutBatchOutput(
            actions=actions_np,
            sidecars={
                "obs_embedding": [
                    obs_embedding_np[i] for i in range(obs_embedding_np.shape[0])
                ]
            },
        ).to_legacy_dict()
        out["timing"] = {
            "encode_s": float(encode_s),
            "world_model_s": float(world_model_s),
            "policy_s": float(policy_s),
        }
        return out

    def update_weights(
        self,
        world_model_sd: dict[str, Any] | None = None,
        policy_sd: dict[str, Any] | None = None,
    ) -> None:
        if world_model_sd is not None:
            self._world_model().load_state_dict(_to_device_state(world_model_sd, self.torch_device))
        if policy_sd is not None:
            self._policy().load_state_dict(_to_device_state(policy_sd, self.torch_device))

    def pull_weights(self, store_name: str, key: str, local_version: int) -> int | None:
        from dreamervla.hybrid_engines.weight_syncer.objectstore import ObjectStoreWeightSyncer

        syncer = ObjectStoreWeightSyncer(store_name=str(store_name))
        if str(key) == "policy":
            return syncer.pull("policy", self._policy(), int(local_version))
        if str(key) == "world_model":
            return syncer.pull("world_model", self._world_model(), int(local_version))
        raise ValueError(f"unknown weight key {key!r}")

    def state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "world_model": _cpu_state_dict(self._world_model()),
            "policy": _cpu_state_dict(self._policy()),
        }

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {"latent": None, "prev_action": None, "is_first": True}

    def _load_initial_weights(self) -> None:
        if "world_model" in self.init_ckpt:
            self._world_model().load_state_dict(
                _to_device_state(self.init_ckpt["world_model"], self.torch_device)
            )
        if "policy" in self.init_ckpt:
            self._policy().load_state_dict(
                _to_device_state(self.init_ckpt["policy"], self.torch_device)
            )

    def _encoder(self) -> Any:
        if self.encoder is None:
            raise RuntimeError("InferenceWorker.init() has not been called")
        return self.encoder

    def _world_model(self) -> torch.nn.Module:
        if self.world_model is None:
            raise RuntimeError("InferenceWorker.init() has not been called")
        return self.world_model

    def _policy(self) -> torch.nn.Module:
        if self.policy is None:
            raise RuntimeError("InferenceWorker.init() has not been called")
        return self.policy


def _build_from_cfg(cfg: dict[str, Any]) -> Any:
    target = cfg.get("target") or cfg.get("_target_") or cfg.get("class_path")
    if not target:
        raise ValueError("component config must include target/_target_/class_path")
    kwargs = dict(cfg.get("kwargs", {}))
    if ":" in str(target):
        module_name, class_name = str(target).split(":", 1)
    else:
        module_name, class_name = str(target).rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)


def _cfg_int(
    cfg: dict[str, Any],
    paths: tuple[tuple[str, ...], ...],
    *,
    default: int,
) -> int:
    for path in paths:
        cur: Any = cfg
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if cur is not None:
            return max(1, int(cur))
    return max(1, int(default))


def _encode_batch(encoder: Any, obs_batch: list[dict[str, Any]]) -> torch.Tensor:
    if hasattr(encoder, "encode_obs_batch"):
        return encoder.encode_obs_batch(obs_batch)
    if hasattr(encoder, "encode"):
        encoded_rows = [encoder.encode(obs) for obs in obs_batch]
        encoded_tensors = [
            torch.from_numpy(row) if isinstance(row, np.ndarray) else row
            for row in encoded_rows
        ]
        return torch.stack(
            [
                _single_encoded_row(
                    row if isinstance(row, torch.Tensor) else torch.as_tensor(row)
                )
                for row in encoded_tensors
            ],
            dim=0,
        )
    encoded = encoder(obs_batch)
    if isinstance(encoded, np.ndarray):
        return torch.from_numpy(encoded)
    return encoded


def _single_encoded_row(row: torch.Tensor) -> torch.Tensor:
    if row.ndim == 2 and row.shape[0] == 1:
        return row.squeeze(0)
    return row


def _concat_structures(values: list[Any]) -> Any:
    if not values:
        raise ValueError("cannot concatenate an empty latent list")
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat([_require_type(value, torch.Tensor) for value in values], dim=0)
    if isinstance(first, dict):
        return {
            key: _concat_structures([_require_type(value, dict)[key] for value in values])
            for key in first
        }
    if isinstance(first, tuple) and hasattr(first, "_fields"):
        return type(first)(
            *[
                _concat_structures([getattr(value, field) for value in values])
                for field in first._fields
            ]
        )
    raise TypeError(f"unsupported latent structure {type(first).__name__}")


def _detach_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_tensor(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*[_detach_tensor(getattr(value, field)) for field in value._fields])
    return value


def _require_type(value: Any, expected: type) -> Any:
    if not isinstance(value, expected):
        raise TypeError(f"expected {expected.__name__}, got {type(value).__name__}")
    return value


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    from dreamervla.hybrid_engines.weight_syncer.objectstore import _independent_cpu

    return {key: _independent_cpu(value) for key, value in module.state_dict().items()}


def _to_device_state(state_dict: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: (value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)).to(device)
        for key, value in state_dict.items()
    }
