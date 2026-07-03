"""FSDP-capable ActorGroup worker that owns VLA PPO updates."""

from __future__ import annotations

import importlib
import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from dreamervla.hybrid_engines.fsdp import FSDPModelManager
from dreamervla.hybrid_engines.weight_syncer import PatchWeightSyncer
from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.worker import Worker
from dreamervla.workers.cotrain.messages import (
    StopMsg,
    TrajectoryBatch,
    TrajectoryShard,
    collate_trajectory_shards,
)

_DEFAULT_PATCH_STORE = "DreamerVLAActorRolloutPatchStore"
_EXTRA_FORWARD_KEYS = (
    "action_token_ids",
    "input_ids",
    "attention_mask",
    "hidden_states",
)


class EmbodiedFSDPActor(Worker):
    """Trainable VLA actor used by the target manual-cotrain ActorGroup route."""

    def __init__(
        self,
        policy_cfg: Any,
        init_ckpt: Any,
        train_cfg: Any,
    ) -> None:
        super().__init__()
        self.policy_cfg = _as_plain_dict(policy_cfg)
        self.init_ckpt = _as_plain_dict(init_ckpt)
        self.train_cfg = _as_plain_dict(train_cfg)
        configured_device = str(self.train_cfg.get("device", self.device))
        if configured_device == "auto":
            configured_device = self.device
        self.torch_device = torch.device(configured_device)

        self.global_step = 0
        self.policy: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.fsdp_manager: FSDPModelManager | None = None
        self.syncer: PatchWeightSyncer | None = None
        self.trajectory_shards: list[TrajectoryShard] = []
        self.batch: TrajectoryBatch | None = None
        self.returns: torch.Tensor | None = None
        self.advantages: torch.Tensor | None = None
        self.group_variance_mask: torch.Tensor | None = None
        self._advantage_metrics: dict[str, float] = {}

    def init(self) -> None:
        """Build policy, optional FSDP wrapper, and optimizer."""

        policy = _build_from_cfg(self.policy_cfg)
        if not isinstance(policy, nn.Module):
            raise TypeError("EmbodiedFSDPActor policy must be a torch.nn.Module")
        policy.to(self.torch_device)
        if "policy" in self.init_ckpt:
            policy.load_state_dict(
                _to_device_state(self.init_ckpt["policy"], self.torch_device)
            )

        fsdp_cfg = self.train_cfg.get("fsdp")
        if fsdp_cfg is not None:
            self.fsdp_manager = FSDPModelManager(**_as_plain_dict(fsdp_cfg))
            self.fsdp_manager.ensure_process_group()
            policy = self.fsdp_manager.prepare_model(policy)

        self.policy = policy
        self.optimizer = self._build_optimizer()

    def set_global_step(self, global_step: int) -> None:
        """Set runner-visible global progress used by weight-sync versions."""

        self.global_step = int(global_step)

    def load_trajectory_shards(self, shards: list[TrajectoryShard]) -> None:
        """Store and collate trajectory shards for the next PPO update."""

        self.trajectory_shards = list(shards)
        self.batch = collate_trajectory_shards(self.trajectory_shards)
        self.returns = None
        self.advantages = None
        self._advantage_metrics = {}

    def recv_rollout_trajectories(
        self,
        actor_channel_name: str,
        expected_shards: int | None = None,
        keyed_counts: list[tuple[str, int]] | None = None,
    ) -> dict[str, float]:
        """Receive trajectory shards from a named ActorGroup channel."""

        channel = Channel.connect(actor_channel_name)
        get_start = time.perf_counter()
        messages: list[Any] = []
        if keyed_counts is not None:
            for key, key_count in keyed_counts:
                key_count = max(0, int(key_count))
                if key_count <= 0:
                    continue
                messages.extend(channel.get(key=str(key)) for _ in range(key_count))
        else:
            count = 1 if expected_shards is None else max(0, int(expected_shards))
            if count > 0:
                messages = [channel.get() for _ in range(count)]
        channel_get_s = float(time.perf_counter() - get_start)
        shards: list[TrajectoryShard] = []
        for msg in messages:
            if isinstance(msg, StopMsg):
                break
            if not isinstance(msg, TrajectoryShard):
                raise TypeError(
                    "EmbodiedFSDPActor expected TrajectoryShard or StopMsg, "
                    f"got {type(msg).__name__}"
                )
            shards.append(msg)
        load_start = time.perf_counter()
        self.load_trajectory_shards(shards)
        load_s = float(time.perf_counter() - load_start)
        return {
            "actor/received_shards": float(len(shards)),
            "actor/channel_get_batch_s": channel_get_s,
            "actor/load_trajectory_shards_s": load_s,
        }

    def compute_advantages_and_returns(self) -> dict[str, float]:
        """Compute trajectory returns and group-relative advantages."""

        batch = self._batch()
        algorithm_cfg = _as_plain_dict(self.train_cfg.get("algorithm_cfg", {}))
        group_size = int(algorithm_cfg.get("group_size", 1))
        if group_size <= 0:
            raise ValueError("algorithm_cfg.group_size must be positive")

        loss_mask = batch.loss_mask.to(self.torch_device, dtype=torch.float32)
        returns = _trajectory_returns_from_rewards(
            batch.rewards.to(self.torch_device, dtype=torch.float32),
            loss_mask=loss_mask,
        )
        trajectory_count = int(returns.numel())
        if trajectory_count <= 0:
            raise ValueError("trajectory batch is empty")
        if trajectory_count % group_size != 0:
            raise ValueError(
                "trajectory count must be divisible by algorithm_cfg.group_size "
                f"for EmbodiedFSDPActor; got {trajectory_count} and {group_size}"
            )

        advantages = _group_advantage(returns, group_size, eps=1e-6)
        if bool(algorithm_cfg.get("filter_zero_variance_groups", False)):
            var_mask = _group_variance_mask(returns, group_size, eps=1e-6)
        else:
            var_mask = torch.ones_like(returns)
        self.group_variance_mask = var_mask.detach()
        self.returns = returns.detach()
        self.advantages = (advantages * var_mask).detach()
        self._advantage_metrics = {
            "actor/trajectory_count": float(trajectory_count),
            "actor/loss_mask_sum": float(loss_mask.detach().sum().cpu().item()),
            "actor/return_mean": float(returns.detach().mean().cpu().item()),
            "actor/advantage_std": float(
                advantages.detach().std(unbiased=False).cpu().item()
            ),
            "actor/zero_variance_masked_rollouts": float(
                (var_mask <= 0.0).sum().cpu().item()
            ),
        }
        return dict(self._advantage_metrics)

    def run_training(self) -> dict[str, float]:
        """Run PPO updates over the loaded rollout trajectories."""

        batch = self._batch()
        if batch.actions.ndim != 4:
            raise ValueError(
                "manual cotrain actor training expects chunk-level actions with shape "
                "[time, batch, chunk, action_dim]"
            )
        advantages = self._advantages()
        loss_mask = batch.loss_mask.to(self.torch_device, dtype=torch.bool)
        if self.group_variance_mask is not None:
            keep = self.group_variance_mask.to(loss_mask.device) > 0.0
            loss_mask = loss_mask & keep.reshape((1, -1) + (1,) * (loss_mask.ndim - 2))
        policy = self._policy()
        optimizer = self._optimizer()
        algorithm_cfg = _as_plain_dict(self.train_cfg.get("algorithm_cfg", {}))
        optim_cfg = _as_plain_dict(
            _as_plain_dict(self.train_cfg.get("optimizers", {})).get("policy", {})
        )

        update_epochs = max(1, int(algorithm_cfg.get("ppo_update_epochs", 1)))
        clip_low = float(algorithm_cfg.get("clip_ratio_low", 0.2))
        clip_high = float(algorithm_cfg.get("clip_ratio_high", 0.28))
        clip_ratio_c_value = algorithm_cfg.get("clip_ratio_c", None)
        clip_ratio_c = (
            None if clip_ratio_c_value is None else float(clip_ratio_c_value)
        )
        clip_log_ratio = float(algorithm_cfg.get("clip_log_ratio", 20.0))
        entropy_coef = _entropy_coef(algorithm_cfg)
        zero_grad_set_to_none = bool(optim_cfg.get("zero_grad_set_to_none", True))

        policy.train()
        losses: list[float] = []
        ratio_means: list[float] = []
        entropy_means: list[float] = []
        grad_norms: list[float] = []
        ppo_updates = 0
        local_time_steps = int(batch.rewards.shape[0])
        global_time_steps = _distributed_max_int(local_time_steps, self.torch_device)
        local_valid_count = int(loss_mask.sum().detach().cpu().item())
        valid_count = _distributed_sum_int(local_valid_count, self.torch_device)
        if valid_count <= 0:
            raise ValueError("trajectory loss_mask has no valid training steps")
        # RLinf masked_mean_ratio: weight every rollout equally regardless of its
        # valid-step count, instead of the global-valid-count sum (which
        # over-weights long/failed rollouts). Default preserves current behavior.
        loss_norm = str(algorithm_cfg.get("loss_normalization", "global_valid_count"))
        per_rollout_count = (
            loss_mask.to(torch.float32)
            .reshape(int(loss_mask.shape[0]), int(loss_mask.shape[1]), -1)
            .sum(dim=(0, 2))
            .clamp(min=1.0)
        )
        num_rollouts = _distributed_sum_int(
            int(loss_mask.shape[1]), self.torch_device
        )
        zero_loss_steps = 0
        for _ in range(update_epochs):
            optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
            epoch_loss = 0.0
            ratio_sum = 0.0
            entropy_sum = 0.0
            for step in range(global_time_steps):
                has_local_step = step < local_time_steps
                if has_local_step:
                    step_mask = _as_vector(loss_mask[step]).to(
                        self.torch_device,
                        dtype=torch.bool,
                    )
                    eval_batch = self._eval_inputs_for_step(batch, step)
                else:
                    step_mask = torch.zeros(
                        int(batch.rewards.shape[1]),
                        device=self.torch_device,
                        dtype=torch.bool,
                    )
                    eval_batch = self._eval_inputs_for_step(batch, 0)

                new_logprob, entropy, _ = policy(eval_batch)
                new_logprob = _as_vector(new_logprob)
                entropy = _as_vector(entropy)
                if not has_local_step:
                    loss = _zero_loss_from_policy_outputs(new_logprob, entropy)
                    loss.backward()
                    zero_loss_steps += 1
                    continue

                old_logprob = _as_vector(
                    batch.prev_logprobs[step].to(
                        self.torch_device,
                        dtype=torch.float32,
                    )
                )
                advantage = _as_vector(advantages).to(self.torch_device)
                if new_logprob.shape != old_logprob.shape:
                    raise ValueError(
                        "policy evaluate log_prob shape must match prev_logprobs; "
                        f"got {tuple(new_logprob.shape)} and {tuple(old_logprob.shape)}"
                    )
                if advantage.shape != old_logprob.shape:
                    raise ValueError(
                        "advantage shape must match prev_logprobs; "
                        f"got {tuple(advantage.shape)} and {tuple(old_logprob.shape)}"
                    )
                if not bool(step_mask.any().item()):
                    loss = _zero_loss_from_policy_outputs(new_logprob, entropy)
                    loss.backward()
                    zero_loss_steps += 1
                    continue
                old_logprob = old_logprob[step_mask]
                advantage = advantage[step_mask]
                new_logprob = new_logprob[step_mask]
                entropy = entropy[step_mask]

                ratio = _ppo_ratio(
                    new_logprob,
                    old_logprob,
                    clip_log_ratio=clip_log_ratio,
                )
                ppo_clip = _ppo_clip_term(
                    ratio,
                    advantage,
                    clip_low,
                    clip_high,
                    clip_ratio_c=clip_ratio_c,
                )
                if loss_norm == "per_rollout":
                    step_weight = (
                        1.0 / per_rollout_count.to(self.torch_device)
                    )[step_mask]
                    loss = (
                        (ppo_clip * step_weight).sum()
                        - float(entropy_coef) * (entropy * step_weight).sum()
                    ) / float(num_rollouts)
                else:
                    loss = (
                        ppo_clip.sum() - float(entropy_coef) * entropy.sum()
                    ) / float(valid_count)
                loss.backward()

                epoch_loss += float(loss.detach().cpu().item())
                ratio_sum += float(ratio.detach().sum().cpu().item())
                entropy_sum += float(entropy.detach().sum().cpu().item())
            grad_norm = self._clip_or_measure_grad_norm(optim_cfg)
            optimizer.step()

            ppo_updates += 1
            losses.append(epoch_loss)
            ratio_means.append(ratio_sum / float(valid_count))
            entropy_means.append(entropy_sum / float(valid_count))
            grad_norms.append(float(grad_norm))

        policy.train(False)
        return {
            "actor/ppo_updates": float(ppo_updates),
            "actor/loss": _mean(losses),
            "actor/ratio_mean": _mean(ratio_means),
            "actor/entropy_mean": _mean(entropy_means),
            "actor/policy_grad_norm": _mean(grad_norms),
            "actor/local_time_steps": float(local_time_steps),
            "actor/global_time_steps": float(global_time_steps),
            "actor/local_loss_mask_sum": float(local_valid_count),
            "actor/global_loss_mask_sum": float(valid_count),
            "actor/zero_loss_steps": float(zero_loss_steps),
            "actor/loss_normalization_per_rollout": (
                1.0 if loss_norm == "per_rollout" else 0.0
            ),
        }

    def sync_model_to_rollout(
        self,
        key: str = "policy",
        version: int | None = None,
    ) -> dict[str, float]:
        """Push the current policy state to RolloutGroup through patch sync."""

        resolved_version = self.global_step if version is None else int(version)
        export_start = time.perf_counter()
        state = self.state_dict()
        export_s = float(time.perf_counter() - export_start)
        push_s = 0.0
        if int(self.rank) == 0 and state:
            push_start = time.perf_counter()
            syncer = self._syncer()
            syncer.push(str(key), state, int(resolved_version))
            push_s = float(time.perf_counter() - push_start)
            syncer_metrics = dict(getattr(syncer, "last_push_metrics", {}) or {})
        else:
            syncer_metrics = {}
        num_tensors = float(len(state))
        num_bytes = float(
            sum(
                value.numel() * value.element_size()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
        )
        metrics = {
            f"sync/{key}_version": float(resolved_version),
            f"sync/{key}_export_s": export_s,
            f"sync/{key}_push_s": push_s,
            f"sync/{key}_tensors": num_tensors,
            f"sync/{key}_bytes": num_bytes,
        }
        metrics.update(syncer_metrics)
        return metrics

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached CPU copy of the policy state."""

        state = _export_policy_state_dict(self._policy())
        return {
            name: value.detach().cpu().clone()
            for name, value in state.items()
        }

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optimizer_cfgs = _as_plain_dict(self.train_cfg.get("optimizers", {}))
        policy_optim_cfg = _as_plain_dict(optimizer_cfgs.get("policy", {}))
        lr = float(policy_optim_cfg.get("lr", self.train_cfg.get("lr", 1e-4)))
        weight_decay = float(policy_optim_cfg.get("weight_decay", 0.0))
        params = [param for param in self._policy().parameters() if param.requires_grad]
        if not params:
            raise ValueError("EmbodiedFSDPActor policy has no trainable parameters")
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def _eval_inputs_for_step(
        self,
        batch: TrajectoryBatch,
        step: int,
    ) -> dict[str, torch.Tensor | str]:
        forward_inputs = batch.forward_inputs
        if "hidden" not in forward_inputs:
            raise ValueError("trajectory forward_inputs must include 'hidden'")
        eval_batch: dict[str, torch.Tensor | str] = {
            "mode": "evaluate",
            "hidden": forward_inputs["hidden"][step].to(
                self.torch_device,
                dtype=torch.float32,
            ),
            "action": batch.actions[step].to(
                self.torch_device,
                dtype=torch.float32,
            ),
        }
        for key in _EXTRA_FORWARD_KEYS:
            if key in forward_inputs:
                eval_batch[key] = forward_inputs[key][step].to(self.torch_device)
        return eval_batch

    def _clip_or_measure_grad_norm(self, optim_cfg: dict[str, Any]) -> float:
        params = [param for param in self._policy().parameters() if param.requires_grad]
        grad_clip_norm = optim_cfg.get("grad_clip_norm", None)
        if grad_clip_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(params, float(grad_clip_norm))
            return float(_to_float(norm))
        return _grad_norm(params)

    def _policy(self) -> nn.Module:
        if self.policy is None:
            raise RuntimeError("EmbodiedFSDPActor.init() has not been called")
        return self.policy

    def _optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            raise RuntimeError("EmbodiedFSDPActor.init() has not been called")
        return self.optimizer

    def _batch(self) -> TrajectoryBatch:
        if self.batch is None:
            raise RuntimeError("load_trajectory_shards() must be called first")
        return self.batch

    def _advantages(self) -> torch.Tensor:
        if self.advantages is None:
            raise RuntimeError("compute_advantages_and_returns() must be called first")
        return self.advantages

    def _syncer(self) -> PatchWeightSyncer:
        if self.syncer is None:
            syncer_cfg = _as_plain_dict(self.train_cfg.get("syncer", {}))
            store_name = str(syncer_cfg.get("store_name", _DEFAULT_PATCH_STORE))
            self.syncer = PatchWeightSyncer(store_name=store_name)
        return self.syncer


def _build_from_cfg(cfg: dict[str, Any]) -> Any:
    target = cfg.get("target") or cfg.get("_target_") or cfg.get("class_path")
    if not target:
        raise ValueError("component config must include target/_target_/class_path")

    kwargs = {
        key: value
        for key, value in cfg.items()
        if key not in {"target", "_target_", "class_path", "kwargs", "init_args"}
    }
    kwargs.update(_as_plain_dict(cfg.get("init_args", {})))
    kwargs.update(_as_plain_dict(cfg.get("kwargs", {})))

    if ":" in str(target):
        module_name, class_name = str(target).split(":", 1)
    else:
        module_name, class_name = str(target).rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)


def _load_grpo_helpers() -> Any:
    path = Path(__file__).resolve().parents[2] / "algorithms" / "ppo" / "grpo.py"
    spec = importlib.util.spec_from_file_location(
        "_dreamervla_embodied_actor_grpo_helpers",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load GRPO helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return dict(OmegaConf.to_container(value, resolve=True) or {})
    return dict(value)


def _to_device_state(value: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        str(name): torch.as_tensor(tensor).to(device)
        for name, tensor in dict(value).items()
    }


def _as_vector(value: Any) -> torch.Tensor:
    tensor = _as_tensor(value)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor
    return tensor.reshape(int(tensor.shape[0]), -1).sum(dim=1)


def _trajectory_returns_from_rewards(
    rewards: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if rewards.ndim < 2:
        raise ValueError("trajectory rewards must have at least [time, batch] dimensions")
    rewards_f = rewards.to(dtype=torch.float32)
    if rewards_f.ndim == 2:
        reward_by_chunk = rewards_f
    else:
        trailing = tuple(range(2, rewards_f.ndim))
        reward_by_chunk = rewards_f.sum(dim=trailing)
    if loss_mask is not None:
        reward_by_chunk = reward_by_chunk * loss_mask.to(
            device=reward_by_chunk.device,
            dtype=reward_by_chunk.dtype,
        )
    return reward_by_chunk.sum(dim=0)


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return torch.as_tensor(value)


def _grad_norm(params: list[torch.nn.Parameter]) -> float:
    norms = [
        param.grad.detach().norm(2)
        for param in params
        if param.grad is not None
    ]
    if not norms:
        return 0.0
    total = torch.norm(torch.stack(norms), 2)
    return float(total.detach().cpu().item())


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _distributed_max_int(value: int, device: torch.device) -> int:
    return _distributed_reduce_int(value, device, torch.distributed.ReduceOp.MAX)


def _distributed_sum_int(value: int, device: torch.device) -> int:
    return _distributed_reduce_int(value, device, torch.distributed.ReduceOp.SUM)


def _distributed_reduce_int(
    value: int,
    device: torch.device,
    op: Any,
) -> int:
    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return int(value)
    backend = str(torch.distributed.get_backend()).lower()
    tensor_device = device if backend == "nccl" else torch.device("cpu")
    tensor = torch.tensor([int(value)], dtype=torch.long, device=tensor_device)
    torch.distributed.all_reduce(tensor, op=op)
    return int(tensor.detach().cpu().item())


def _zero_loss_from_policy_outputs(
    new_logprob: torch.Tensor,
    entropy: torch.Tensor,
) -> torch.Tensor:
    return new_logprob.sum() * 0.0 + entropy.sum() * 0.0


def _export_policy_state_dict(policy: nn.Module) -> dict[str, torch.Tensor]:
    if _is_fsdp_module(policy):
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel,
            StateDictType,
        )

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(
            policy,
            StateDictType.FULL_STATE_DICT,
            cfg,
        ):
            state = policy.state_dict()
    else:
        state = policy.state_dict()
    return {
        str(name): torch.as_tensor(value)
        for name, value in dict(state).items()
    }


def _is_fsdp_module(policy: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except Exception:
        return False
    return isinstance(policy, FullyShardedDataParallel)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


_GRPO_HELPERS = _load_grpo_helpers()
_entropy_coef = _GRPO_HELPERS._entropy_coef
_group_advantage = _GRPO_HELPERS._group_advantage
_group_variance_mask = _GRPO_HELPERS.group_variance_mask
_ppo_clip_term = _GRPO_HELPERS._ppo_clip_term
_ppo_ratio = _GRPO_HELPERS._ppo_ratio

__all__ = ["EmbodiedFSDPActor"]
