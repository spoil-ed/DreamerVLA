"""FSDP-capable ActorGroup worker that owns VLA PPO updates."""

from __future__ import annotations

import importlib
import importlib.util
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from dreamervla.hybrid_engines.fsdp import FSDPModelManager
from dreamervla.hybrid_engines.fsdp.strategy import dtype_from_precision
from dreamervla.hybrid_engines.weight_syncer import BucketWeightSyncer
from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.worker import Worker
from dreamervla.utils.progress import ProgressReporter
from dreamervla.utils.seed import capture_rng_state, restore_rng_state, select_rank_rng_state
from dreamervla.workers.cotrain.messages import (
    RealTrajectoryBatch,
    StopMsg,
    TrajectoryBatch,
    TrajectoryShard,
    collate_trajectory_shards,
)

_DEFAULT_WEIGHT_STORE = "DreamerVLAActorRolloutWeightStore"
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
        fsdp_cfg = _as_plain_dict(self.train_cfg.get("fsdp", {}))
        self.policy_input_dtype = dtype_from_precision(
            fsdp_cfg.get(
                "precision",
                self.train_cfg.get("precision", "fp32"),
            )
        )
        self.enable_offload = bool(self.train_cfg.get("enable_offload", False))
        self.is_weight_offloaded = False
        self.is_optimizer_offloaded = False
        self.is_encoder_optimizer_offloaded = False
        self.memory_phase = "uninitialized"

        self.global_step = 0
        self.policy: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.encoder_optimizer: torch.optim.Optimizer | None = None
        self._encoder_parameter_ids: set[int] = set()
        self.fsdp_manager: FSDPModelManager | None = None
        self.syncer: BucketWeightSyncer | None = None
        self.trajectory_shards: list[TrajectoryShard] = []
        self.batch: TrajectoryBatch | None = None
        self.returns: torch.Tensor | None = None
        self.advantages: torch.Tensor | None = None
        self.rollout_filter_mask: torch.Tensor | None = None
        self._advantage_metrics: dict[str, float] = {}
        self._transaction_policy_state: dict[str, torch.Tensor] | None = None
        self._transaction_policy_optimizer: dict[str, Any] | None = None
        self._transaction_encoder_optimizer: dict[str, Any] | None = None
        self._transaction_uses_local_state = False

    def init(self) -> None:
        """Build policy, optional FSDP wrapper, and optimizer."""

        policy = _build_from_cfg(self.policy_cfg)
        if not isinstance(policy, nn.Module):
            raise TypeError("EmbodiedFSDPActor policy must be a torch.nn.Module")
        policy.to(self.torch_device)
        if "policy" in self.init_ckpt:
            policy.load_state_dict(_to_device_state(self.init_ckpt["policy"], self.torch_device))

        fsdp_cfg = self.train_cfg.get("fsdp")
        if fsdp_cfg is not None:
            self.fsdp_manager = FSDPModelManager(**_as_plain_dict(fsdp_cfg))
            self.fsdp_manager.ensure_process_group()
            policy = self.fsdp_manager.prepare_model(policy)

        self.policy = policy
        encoder_params, actor_params = self._partition_trainable_parameters()
        self._encoder_parameter_ids = {id(parameter) for parameter in encoder_params}
        self.optimizer = self._build_optimizer("policy", actor_params)
        if encoder_params:
            self.encoder_optimizer = self._build_optimizer("encoder", encoder_params)
        optimizer_state = self.init_ckpt.get("policy_optimizer")
        if isinstance(optimizer_state, dict) and optimizer_state:
            self._load_optimizer_state_dict(optimizer_state)
        encoder_optimizer_state = self.init_ckpt.get("encoder_optimizer")
        if (
            self.encoder_optimizer is not None
            and isinstance(encoder_optimizer_state, dict)
            and encoder_optimizer_state
        ):
            self._load_optimizer_state_dict(
                encoder_optimizer_state,
                optimizer=self.encoder_optimizer,
            )
        # Encoder parameters are opt-in only during encoder_sft(). Imagined PPO
        # always updates the native action decoder/actor partition.
        for parameter in self.policy.parameters():
            if id(parameter) in self._encoder_parameter_ids:
                parameter.requires_grad_(False)
        self.memory_phase = "training_loaded"
        if self.enable_offload:
            self._offload_optimizer()
            if self.encoder_optimizer is not None:
                self._offload_optimizer(self.encoder_optimizer)
            self._offload_parameters(offload_grad=True)

    def set_global_step(self, global_step: int) -> None:
        """Set runner-visible global progress used by weight-sync versions."""

        self.global_step = int(global_step)

    def encoder_sft(self, batch: RealTrajectoryBatch) -> dict[str, float]:
        """Run encoder SFT with an explicit Actor residency lease."""

        self._load_for_training()
        try:
            return self._encoder_sft_loaded(batch)
        finally:
            self._release_after_training()

    def _encoder_sft_loaded(self, batch: RealTrajectoryBatch) -> dict[str, float]:
        """SFT only the raw-input encoder on successful current-step episodes."""

        successful = [trajectory for trajectory in batch.trajectories if trajectory.success]
        decisions = [
            dict(transition)
            for trajectory in successful
            for transition in trajectory.transitions
            if bool(transition.get("policy_decision", False))
            and "action_token_ids_chunk" in transition
        ]
        base_metrics = {
            "actor/encoder_sft_trajectories": float(len(successful)),
            "actor/encoder_sft_decisions": float(len(decisions)),
        }
        if not decisions:
            return {
                **base_metrics,
                "actor/encoder_sft_skipped": 1.0,
                "actor/encoder_sft_optimizer_steps": 0.0,
                "actor/encoder_sft_loss": 0.0,
                "actor/encoder_sft_kl": 0.0,
            }
        optimizer = self.encoder_optimizer
        if optimizer is None:
            raise RuntimeError(
                "encoder SFT requires policy.encoder_parameter_names() and an encoder optimizer"
            )
        config = _as_plain_dict(self.train_cfg.get("encoder_sft", {}))
        epochs = max(1, int(config.get("epochs", 1)))
        batch_size = max(1, int(config.get("batch_size", 1)))
        total_optimizer_steps = epochs * ((len(decisions) + batch_size - 1) // batch_size)
        progress = ProgressReporter(
            total_optimizer_steps,
            f"cotrain-vla-real-sft/{int(self.global_step):08d}",
            enabled=int(self.rank) == 0,
            min_interval_s=float(self.train_cfg.get("progress_every_s", 5.0)),
            unit="update",
        )
        progress.set_status(f"phase=encoder_sft epoch=1/{epochs} update=0/{total_optimizer_steps}")
        progress.set(0, force=True)
        zero_grad_set_to_none = bool(config.get("zero_grad_set_to_none", True))
        policy = self._policy()
        original_training = bool(policy.training)
        old_action_logprobs = self._encoder_sft_action_logprobs(
            policy,
            decisions,
            batch_size=batch_size,
        )
        original_requires_grad = {
            id(parameter): bool(parameter.requires_grad) for parameter in policy.parameters()
        }
        for parameter in policy.parameters():
            parameter.requires_grad_(id(parameter) in self._encoder_parameter_ids)
        policy.train()
        loss_sum = 0.0
        optimizer_steps = 0
        try:
            for _epoch in range(epochs):
                for start in range(0, len(decisions), batch_size):
                    records = decisions[start : start + batch_size]
                    prepared = self._prepare_raw_batch(records)
                    labels = torch.as_tensor(
                        np.stack(
                            [np.asarray(record["action_token_ids_chunk"]) for record in records],
                            axis=0,
                        ),
                        dtype=torch.long,
                        device=self.torch_device,
                    ).reshape(len(records), -1)
                    optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
                    loss, _unused, _extras = policy(
                        {
                            "mode": "encoder_sft",
                            **prepared,
                            "action_token_ids": labels,
                        }
                    )
                    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                        raise TypeError("encoder_sft policy forward must return scalar loss")
                    loss.backward()
                    grad_clip = config.get("grad_clip_norm")
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [
                                parameter
                                for parameter in policy.parameters()
                                if id(parameter) in self._encoder_parameter_ids
                            ],
                            float(grad_clip),
                        )
                    optimizer.step()
                    loss_sum += float(loss.detach().cpu().item())
                    optimizer_steps += 1
                    progress.set_status(
                        f"phase=encoder_sft epoch={_epoch + 1}/{epochs} "
                        f"update={optimizer_steps}/{total_optimizer_steps} "
                        f"loss={float(loss.detach().cpu().item()):.4g}"
                    )
                    progress.set(
                        optimizer_steps,
                        force=optimizer_steps == total_optimizer_steps,
                    )
        finally:
            for parameter in policy.parameters():
                parameter.requires_grad_(original_requires_grad[id(parameter)])
            policy.train(original_training)
        new_action_logprobs = self._encoder_sft_action_logprobs(
            policy,
            decisions,
            batch_size=batch_size,
        )
        encoder_kl = float(
            F.kl_div(
                new_action_logprobs,
                old_action_logprobs.exp(),
                reduction="batchmean",
            )
            .clamp_min(0.0)
            .item()
        )
        return {
            **base_metrics,
            "actor/encoder_sft_skipped": 0.0,
            "actor/encoder_sft_optimizer_steps": float(optimizer_steps),
            "actor/encoder_sft_loss": loss_sum / float(max(1, optimizer_steps)),
            "actor/encoder_sft_kl": encoder_kl,
        }

    def _encoder_sft_action_logprobs(
        self,
        policy: nn.Module,
        decisions: list[dict[str, Any]],
        *,
        batch_size: int,
    ) -> torch.Tensor:
        original_training = bool(policy.training)
        policy.eval()
        values: list[torch.Tensor] = []
        try:
            with torch.no_grad():
                for start in range(0, len(decisions), int(batch_size)):
                    records = decisions[start : start + int(batch_size)]
                    prepared = self._prepare_raw_batch(records)
                    labels = torch.as_tensor(
                        np.stack(
                            [np.asarray(record["action_token_ids_chunk"]) for record in records],
                            axis=0,
                        ),
                        dtype=torch.long,
                        device=self.torch_device,
                    ).reshape(len(records), -1)
                    _loss, _unused, extras = policy(
                        {
                            "mode": "encoder_sft",
                            **prepared,
                            "action_token_ids": labels,
                        }
                    )
                    logits = extras.get("action_logits")
                    if not isinstance(logits, torch.Tensor):
                        raise TypeError("encoder_sft policy must return action_logits")
                    values.append(
                        torch.log_softmax(logits.detach().float(), dim=-1)
                        .reshape(-1, int(logits.shape[-1]))
                        .cpu()
                    )
        finally:
            policy.train(original_training)
        return torch.cat(values, dim=0)

    def begin_policy_transaction(self) -> dict[str, float]:
        """Snapshot policy and optimizer state for a bounded-KL sub-transaction."""

        if self._transaction_policy_state is not None:
            raise RuntimeError("a policy KL transaction is already active")
        self._transaction_uses_local_state = _is_fsdp_module(self._policy())
        if self._transaction_uses_local_state:
            restore_weights = bool(self.enable_offload and self.is_weight_offloaded)
            self._load_parameters(onload_grad=False)
            try:
                self._transaction_policy_state = _export_local_policy_state(self._policy())
                self._transaction_policy_optimizer = _to_cpu_tree(self._optimizer().state_dict())
                self._transaction_encoder_optimizer = (
                    None
                    if self.encoder_optimizer is None
                    else _to_cpu_tree(self.encoder_optimizer.state_dict())
                )
            finally:
                if restore_weights:
                    self._offload_parameters(offload_grad=True)
        else:
            self._transaction_policy_state = self.state_dict()
            self._transaction_policy_optimizer = self.optimizer_state_dict()
            self._transaction_encoder_optimizer = self.encoder_optimizer_state_dict()
        return {"actor/kl_transaction_started": 1.0}

    def finalize_policy_transaction(
        self,
        observed_kl: float,
        max_kl: float,
    ) -> dict[str, float]:
        """Accept a sub-update or restore its exact pre-update state."""

        if self._transaction_policy_state is None:
            raise RuntimeError("no active policy KL transaction")
        observed = max(0.0, float(observed_kl))
        budget = float(max_kl)
        accepted = observed <= max(0.0, budget)
        if not accepted:
            restore_weights = bool(self.enable_offload and self.is_weight_offloaded)
            restore_policy_optimizer = bool(self.enable_offload and self.is_optimizer_offloaded)
            restore_encoder_optimizer = bool(
                self.enable_offload and self.is_encoder_optimizer_offloaded
            )
            try:
                self._load_parameters(onload_grad=False)
                self._load_optimizer()
                if self.encoder_optimizer is not None:
                    self._load_optimizer(self.encoder_optimizer)
                if self._transaction_uses_local_state:
                    _load_local_policy_state(
                        self._policy(),
                        self._transaction_policy_state,
                    )
                else:
                    _load_policy_state_dict(
                        self._policy(),
                        self._transaction_policy_state,
                        rank=int(self.rank),
                    )
                if self._transaction_policy_optimizer is not None:
                    if self._transaction_uses_local_state:
                        self._optimizer().load_state_dict(self._transaction_policy_optimizer)
                    else:
                        self._load_optimizer_state_dict(
                            self._transaction_policy_optimizer,
                        )
                if (
                    self.encoder_optimizer is not None
                    and self._transaction_encoder_optimizer is not None
                ):
                    if self._transaction_uses_local_state:
                        self.encoder_optimizer.load_state_dict(self._transaction_encoder_optimizer)
                    else:
                        self._load_optimizer_state_dict(
                            self._transaction_encoder_optimizer,
                            optimizer=self.encoder_optimizer,
                        )
            finally:
                if restore_policy_optimizer:
                    self._offload_optimizer()
                if restore_encoder_optimizer and self.encoder_optimizer is not None:
                    self._offload_optimizer(self.encoder_optimizer)
                if restore_weights:
                    self._offload_parameters(offload_grad=True)
        self._transaction_policy_state = None
        self._transaction_policy_optimizer = None
        self._transaction_encoder_optimizer = None
        self._transaction_uses_local_state = False
        return {
            "actor/kl_transaction_observed": observed,
            "actor/kl_transaction_budget": budget,
            "actor/kl_transaction_accepted": float(accepted),
            "actor/kl_transaction_rolled_back": float(not accepted),
        }

    def reencode_real_trajectories(
        self,
        batch: RealTrajectoryBatch,
    ) -> RealTrajectoryBatch:
        """Re-encode raw trajectories while the Actor parameters are resident."""

        self._load_parameters(onload_grad=False)
        try:
            return self._reencode_real_trajectories_loaded(batch)
        finally:
            if self.enable_offload:
                self._offload_parameters(offload_grad=True)

    def _reencode_real_trajectories_loaded(
        self,
        batch: RealTrajectoryBatch,
    ) -> RealTrajectoryBatch:
        """Recompute current encoder tokens for every transition in ``batch``."""

        config = _as_plain_dict(self.train_cfg.get("encoder_sft", {}))
        batch_size = max(1, int(config.get("reencode_batch_size", config.get("batch_size", 1))))
        flat: list[tuple[int, int, dict[str, Any]]] = [
            (trajectory_index, transition_index, dict(transition))
            for trajectory_index, trajectory in enumerate(batch.trajectories)
            for transition_index, transition in enumerate(trajectory.transitions)
        ]
        replacements: dict[tuple[int, int], dict[str, Any]] = {}
        policy = self._policy()
        original_training = bool(policy.training)
        policy.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(flat), batch_size):
                    rows = flat[start : start + batch_size]
                    prepared = self._prepare_raw_batch([row[2] for row in rows])
                    _hidden, _unused, extras = policy({"mode": "encode_raw", **prepared})
                    hidden = torch.as_tensor(extras["hidden"]).detach().float().cpu()
                    lang = torch.as_tensor(extras["lang_emb"]).detach().float().cpu()
                    if int(hidden.shape[0]) != len(rows) or int(lang.shape[0]) != len(rows):
                        raise ValueError("raw re-encoding output batch size mismatch")
                    for row_index, (trajectory_index, transition_index, transition) in enumerate(
                        rows
                    ):
                        updated = dict(transition)
                        updated["obs_embedding"] = hidden[row_index].numpy()
                        updated["lang_emb"] = lang[row_index].numpy()
                        updated["encoder_version"] = int(batch.global_step)
                        replacements[(trajectory_index, transition_index)] = updated
        finally:
            policy.train(original_training)
        trajectories = tuple(
            replace(
                trajectory,
                transitions=tuple(
                    replacements[(trajectory_index, transition_index)]
                    for transition_index, _transition in enumerate(trajectory.transitions)
                ),
            )
            for trajectory_index, trajectory in enumerate(batch.trajectories)
        )
        if int(self.rank) != 0:
            # Every FSDP rank participates in the forwards, but only rank zero
            # returns the very large sidecar batch to Ray's object store.
            return RealTrajectoryBatch(
                global_step=batch.global_step,
                trajectories=(),
            )
        return RealTrajectoryBatch(global_step=batch.global_step, trajectories=trajectories)

    def _prepare_raw_batch(self, transitions: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        module = _unwrap_policy_module(self._policy())
        prepare = getattr(module, "prepare_raw_batch", None)
        if not callable(prepare):
            raise TypeError("staged VLA policy must implement prepare_raw_batch")
        prepared = dict(prepare(transitions))
        for key in ("input_ids", "attention_mask", "pixel_values"):
            if key not in prepared:
                raise KeyError(f"prepare_raw_batch must return {key!r}")
            prepared[key] = torch.as_tensor(prepared[key]).to(self.torch_device)
        return prepared

    def _partition_trainable_parameters(
        self,
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        policy = self._policy()
        module = _unwrap_policy_module(policy)
        name_getter = getattr(module, "encoder_parameter_names", None)
        declared = set(name_getter()) if callable(name_getter) else set()
        encoder: list[nn.Parameter] = []
        actor: list[nn.Parameter] = []
        for name, parameter in policy.named_parameters():
            if not parameter.requires_grad:
                continue
            normalized = str(name).removeprefix("_fsdp_wrapped_module.")
            is_encoder = any(
                normalized == declared_name or normalized.endswith(f".{declared_name}")
                for declared_name in declared
            )
            (encoder if is_encoder else actor).append(parameter)
        if not actor:
            raise ValueError("EmbodiedFSDPActor policy has no trainable actor parameters")
        return encoder, actor

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
        filter_mask, n_filtered = _rollout_filter_mask(returns, algorithm_cfg, group_size)
        self.rollout_filter_mask = filter_mask.detach()
        self.returns = returns.detach()
        # RLinf applies the reward filter to loss_mask (not the advantage); a
        # filtered rollout's loss_mask is zeroed so its advantage never enters.
        self.advantages = advantages.detach()
        self._advantage_metrics = {
            "actor/trajectory_count": float(trajectory_count),
            "actor/loss_mask_sum": float(loss_mask.detach().sum().cpu().item()),
            "actor/return_mean": float(returns.detach().mean().cpu().item()),
            "actor/advantage_std": float(advantages.detach().std(unbiased=False).cpu().item()),
            "actor/reward_filtered_rollouts": float(n_filtered),
        }
        return dict(self._advantage_metrics)

    def run_training(self) -> dict[str, float]:
        """Run PPO under the Actor's exclusive training residency phase."""

        self._load_for_training()
        try:
            metrics = self._run_training_loaded()
        finally:
            self._release_after_training()
        metrics.update(self._memory_metrics("train_end"))
        return metrics

    def _run_training_loaded(self) -> dict[str, float]:
        """Run PPO with the same batch hierarchy used by RLinf embodied actors.

        Trajectory chunks are flattened and deterministically shuffled, split
        into per-rank global batches, then split again into micro batches.
        Gradients accumulate inside one global batch and every global batch owns
        exactly one optimizer step.
        """

        batch = self._batch()
        if batch.actions.ndim != 4:
            raise ValueError(
                "manual cotrain actor training expects chunk-level actions with shape "
                "[time, batch, chunk, action_dim]"
            )
        policy = self._policy()
        optimizer = self._optimizer()
        algorithm_cfg = _as_plain_dict(self.train_cfg.get("algorithm_cfg", {}))
        optim_cfg = _as_plain_dict(
            _as_plain_dict(self.train_cfg.get("optimizers", {})).get("policy", {})
        )
        update_epochs = max(1, int(algorithm_cfg.get("ppo_update_epochs", 1)))
        clip_low = float(algorithm_cfg.get("clip_ratio_low", 0.2))
        clip_high = float(algorithm_cfg.get("clip_ratio_high", 0.28))
        clip_ratio_c_value = algorithm_cfg.get("clip_ratio_c")
        clip_ratio_c = None if clip_ratio_c_value is None else float(clip_ratio_c_value)
        clip_log_ratio_value = algorithm_cfg.get("clip_log_ratio", 20.0)
        clip_log_ratio = None if clip_log_ratio_value is None else float(clip_log_ratio_value)
        entropy_coef = float(_entropy_coef(algorithm_cfg))
        kl_coef = float(algorithm_cfg.get("kl_coef", algorithm_cfg.get("kl_beta", 0.0)))
        zero_grad_set_to_none = bool(optim_cfg.get("zero_grad_set_to_none", True))
        logprob_type = str(algorithm_cfg.get("logprob_type", "")).lower()
        token_level_logprob = logprob_type == "token_level"
        loss_norm = str(algorithm_cfg.get("loss_normalization", "token_mean"))

        local_time_steps = int(batch.rewards.shape[0])
        local_rollouts = int(batch.rewards.shape[1])
        global_time_steps = _distributed_max_int(local_time_steps, self.torch_device)
        global_rollouts = _distributed_sum_int(local_rollouts, self.torch_device)
        min_rollouts = _distributed_min_int(local_rollouts, self.torch_device)
        max_rollouts = _distributed_max_int(local_rollouts, self.torch_device)
        if min_rollouts != max_rollouts and self.train_cfg.get("global_batch_size"):
            raise ValueError(
                "RLinf-style FSDP PPO requires equal trajectory counts on every "
                f"Actor rank; observed min={min_rollouts}, max={max_rollouts}"
            )

        local_sample_count = global_time_steps * local_rollouts
        global_sample_count = _distributed_sum_int(
            local_sample_count,
            self.torch_device,
        )
        world_size = max(1, int(self.world_size))
        configured_global_batch = self.train_cfg.get("global_batch_size")
        global_batch_size = (
            global_sample_count if configured_global_batch is None else int(configured_global_batch)
        )
        if global_batch_size <= 0 or global_batch_size % world_size != 0:
            raise ValueError(
                "actor.train_cfg.global_batch_size must be positive and divisible "
                f"by Actor world_size; got {global_batch_size} and {world_size}"
            )
        local_global_batch_size = global_batch_size // world_size
        if local_sample_count % local_global_batch_size != 0:
            raise ValueError(
                "flattened per-rank PPO samples must be divisible by the per-rank "
                "global batch: "
                f"{local_sample_count} % {local_global_batch_size} != 0"
            )
        configured_micro_batch = self.train_cfg.get("micro_batch_size")
        micro_batch_size = (
            max(1, local_rollouts)
            if configured_micro_batch is None
            else int(configured_micro_batch)
        )
        if micro_batch_size <= 0 or local_global_batch_size % micro_batch_size != 0:
            raise ValueError(
                "actor.train_cfg.micro_batch_size must be positive and divide the "
                "per-rank global batch: "
                f"{local_global_batch_size} % {micro_batch_size} != 0"
            )
        global_batches_per_epoch = local_sample_count // local_global_batch_size
        micro_batches_per_global = local_global_batch_size // micro_batch_size
        optimizer_steps_total = update_epochs * global_batches_per_epoch
        forward_backward_steps_total = optimizer_steps_total * micro_batches_per_global
        progress_ops_total = optimizer_steps_total + forward_backward_steps_total

        loss_mask_cpu = batch.loss_mask.to(device="cpu", dtype=torch.bool)
        if self.rollout_filter_mask is not None:
            keep = self.rollout_filter_mask.detach().to(device="cpu") > 0.0
            loss_mask_cpu = loss_mask_cpu & keep.reshape((1, -1) + (1,) * (loss_mask_cpu.ndim - 2))
        local_flat_mask = _as_vector(_flatten_time_batch(loss_mask_cpu)).to(dtype=torch.bool)
        padded_flat_mask = torch.zeros(local_sample_count, dtype=torch.bool)
        padded_flat_mask[: int(local_flat_mask.numel())] = local_flat_mask
        local_valid_count = int(padded_flat_mask.sum().item())
        valid_count = _distributed_sum_int(local_valid_count, self.torch_device)
        logprob_tokens_per_sample = (
            _trailing_numel(batch.prev_logprobs.shape[2:]) if token_level_logprob else 1
        )
        local_logprob_token_count = local_valid_count * logprob_tokens_per_sample
        logprob_token_count = _distributed_sum_int(
            local_logprob_token_count,
            self.torch_device,
        )

        progress = ProgressReporter(
            progress_ops_total,
            f"cotrain-vla-ppo/{int(self.global_step):08d}",
            enabled=int(self.rank) == 0,
            min_interval_s=float(self.train_cfg.get("progress_every_s", 5.0)),
            unit="op",
        )
        progress.set_status(
            "phase=forward_backward epoch=1/"
            f"{update_epochs} global_batch=1/{global_batches_per_epoch} "
            f"micro_batch=0/{micro_batches_per_global} optimizer=0/"
            f"{optimizer_steps_total}"
        )
        progress.set(0, force=True)

        common_metrics = {
            "actor/local_time_steps": float(local_time_steps),
            "actor/global_time_steps": float(global_time_steps),
            "actor/local_rollout_trajectories": float(local_rollouts),
            "actor/global_rollout_trajectories": float(global_rollouts),
            "actor/local_ppo_samples": float(local_sample_count),
            "actor/global_ppo_samples": float(global_sample_count),
            "actor/global_batch_size": float(global_batch_size),
            "actor/per_rank_global_batch_size": float(local_global_batch_size),
            "actor/micro_batch_size": float(micro_batch_size),
            "actor/local_loss_mask_sum": float(local_valid_count),
            "actor/global_loss_mask_sum": float(valid_count),
            "actor/local_logprob_token_count": float(local_logprob_token_count),
            "actor/global_logprob_token_count": float(logprob_token_count),
            "actor/valid_sample_fraction": (
                float(valid_count) / float(max(1, global_sample_count))
            ),
            "actor/kl_coef": float(kl_coef),
            "actor/loss_normalization_per_rollout": (1.0 if loss_norm == "per_rollout" else 0.0),
            "actor/logprob_type_token_level": (1.0 if token_level_logprob else 0.0),
        }
        if valid_count <= 0:
            policy.train(False)
            progress.set_status("phase=skipped reason=zero_valid_samples optimizer=0/0")
            progress.set(0, force=True)
            return {
                **common_metrics,
                "actor/ppo_updates": 0.0,
                "actor/ppo_optimizer_steps": 0.0,
                "actor/ppo_forward_backward_steps": 0.0,
                "actor/ppo_progress_ops": 0.0,
                "actor/policy_loss": 0.0,
                "actor/loss": 0.0,
                "actor/total_loss": 0.0,
                "actor/ratio": 0.0,
                "actor/ratio_mean": 0.0,
                "actor/ratio_abs": 0.0,
                "actor/clipped_ratio": 0.0,
                "actor/approx_kl": 0.0,
                "actor/clip_fraction": 0.0,
                "actor/dual_clip_fraction": 0.0,
                "actor/entropy_mean": 0.0,
                "actor/behavior_kl_mean": 0.0,
                "actor/policy_grad_norm": 0.0,
                "actor/grad_norm": 0.0,
                "actor/lr": _optimizer_lr(optimizer),
                "actor/zero_loss_steps": 0.0,
                "actor/zero_loss_micro_batches": 0.0,
                "actor/skipped_zero_valid_update": 1.0,
            }

        per_rollout_count = (
            loss_mask_cpu.to(torch.float32)
            .reshape(local_time_steps, local_rollouts, -1)
            .sum(dim=(0, 2))
            .clamp(min=1.0)
        )
        inverse_per_rollout_count = per_rollout_count.reciprocal().to(self.torch_device)
        advantage_vector = _as_vector(self._advantages()).to(self.torch_device)
        flat_old_logprob = _flatten_time_batch(batch.prev_logprobs)

        shuffle_ids = torch.arange(local_sample_count, dtype=torch.long)
        if "seed" in self.train_cfg:
            generator = torch.Generator()
            generator.manual_seed(int(self.train_cfg["seed"]) + int(self.rank))
            shuffle_ids = torch.randperm(local_sample_count, generator=generator)

        metric_names = (
            "policy_loss",
            "total_loss",
            "ratio",
            "ratio_abs",
            "clipped_ratio",
            "approx_kl",
            "clip_fraction",
            "dual_clip_fraction",
            "entropy",
            "behavior_kl",
        )
        metric_sums = {name: torch.zeros((), device=self.torch_device) for name in metric_names}
        grad_norms: list[float] = []
        optimizer_steps = 0
        forward_backward_steps = 0
        progress_ops = 0
        zero_loss_micro_batches = 0
        fsdp_policy = _is_fsdp_module(policy)
        fsdp_gradient_accumulation = bool(
            fsdp_policy
            and self.fsdp_manager is not None
            and self.fsdp_manager.enable_gradient_accumulation
        )
        policy.train()

        for epoch in range(update_epochs):
            for global_batch_index in range(global_batches_per_epoch):
                batch_lo = global_batch_index * local_global_batch_size
                batch_hi = batch_lo + local_global_batch_size
                global_batch_indices = shuffle_ids[batch_lo:batch_hi]
                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
                for micro_index in range(micro_batches_per_global):
                    micro_lo = micro_index * micro_batch_size
                    micro_hi = micro_lo + micro_batch_size
                    padded_indices = global_batch_indices[micro_lo:micro_hi]
                    source_indices = _local_source_indices(
                        padded_indices,
                        local_time_steps=local_time_steps,
                        local_rollouts=local_rollouts,
                    )
                    trajectory_indices = padded_indices.remainder(local_rollouts)
                    sample_mask_cpu = padded_flat_mask.index_select(
                        0,
                        padded_indices,
                    )
                    sample_mask = sample_mask_cpu.to(
                        self.torch_device,
                        dtype=torch.bool,
                    )
                    eval_batch = self._eval_inputs_for_flat_indices(
                        batch,
                        source_indices,
                    )
                    if logprob_type:
                        eval_batch["logprob_type"] = logprob_type
                    old_logprob_raw = flat_old_logprob.index_select(
                        0,
                        source_indices,
                    ).to(self.torch_device, dtype=torch.float32)
                    old_logprob = (
                        old_logprob_raw if token_level_logprob else _as_vector(old_logprob_raw)
                    )
                    advantage = advantage_vector.index_select(
                        0,
                        trajectory_indices.to(self.torch_device),
                    )
                    is_last_micro = micro_index == micro_batches_per_global - 1
                    backward_context = (
                        policy.no_sync()
                        if fsdp_gradient_accumulation and not is_last_micro
                        else nullcontext()
                    )
                    with backward_context:
                        new_logprob, entropy, _ = policy(eval_batch)
                        if token_level_logprob:
                            new_logprob = _as_tensor(new_logprob).to(
                                self.torch_device,
                                dtype=torch.float32,
                            )
                            entropy = _match_logprob_shape(
                                _as_tensor(entropy).to(
                                    self.torch_device,
                                    dtype=torch.float32,
                                ),
                                new_logprob,
                                name="entropy",
                            )
                        else:
                            new_logprob = _as_vector(new_logprob)
                            entropy = _as_vector(entropy)
                        if new_logprob.shape != old_logprob.shape:
                            raise ValueError(
                                "policy evaluate log_prob shape must match "
                                "prev_logprobs; got "
                                f"{tuple(new_logprob.shape)} and "
                                f"{tuple(old_logprob.shape)}"
                            )
                        if not bool(sample_mask_cpu.any()):
                            zero_loss = _zero_loss_from_policy_outputs(
                                new_logprob,
                                entropy,
                            )
                            (zero_loss / float(micro_batches_per_global)).backward()
                            zero_loss_micro_batches += 1
                        else:
                            old_logprob = old_logprob[sample_mask]
                            advantage = advantage[sample_mask]
                            new_logprob = new_logprob[sample_mask]
                            entropy = entropy[sample_mask]
                            if token_level_logprob:
                                advantage = _expand_batch_vector_as(
                                    advantage,
                                    new_logprob,
                                )
                            raw_log_ratio = new_logprob - old_logprob.detach()
                            ratio = _ppo_ratio(
                                new_logprob,
                                old_logprob,
                                clip_log_ratio=clip_log_ratio,
                            )
                            clipped_ratio = ratio.clamp(
                                1.0 - clip_low,
                                1.0 + clip_high,
                            )
                            policy_loss1 = -advantage * ratio
                            policy_loss2 = -advantage * clipped_ratio
                            clip_mask = policy_loss1.detach() < policy_loss2.detach()
                            unclipped_policy_loss = torch.maximum(
                                policy_loss1,
                                policy_loss2,
                            )
                            if clip_ratio_c is None:
                                dual_clip_mask = torch.zeros_like(
                                    clip_mask,
                                    dtype=torch.bool,
                                )
                            else:
                                dual_bound = float(clip_ratio_c) * advantage.abs()
                                dual_clip_mask = (
                                    dual_bound.detach() < unclipped_policy_loss.detach()
                                )
                            ppo_clip = _ppo_clip_term(
                                ratio,
                                advantage,
                                clip_low,
                                clip_high,
                                clip_ratio_c=clip_ratio_c,
                            )
                            behavior_kl = _approx_behavior_kl(
                                new_logprob,
                                old_logprob,
                                clip_log_ratio=clip_log_ratio,
                            )
                            if loss_norm == "per_rollout" and not token_level_logprob:
                                weights = inverse_per_rollout_count.index_select(
                                    0,
                                    trajectory_indices.to(self.torch_device),
                                )[sample_mask]
                                weight_denom = weights.sum().clamp_min(1.0e-8)
                                policy_loss = (ppo_clip * weights).sum() / weight_denom
                                behavior_kl_mean = (behavior_kl * weights).sum() / weight_denom
                                entropy_mean = (entropy * weights).sum() / weight_denom
                            else:
                                policy_loss = ppo_clip.mean()
                                behavior_kl_mean = behavior_kl.mean()
                                entropy_mean = entropy.mean()
                            total_loss = (
                                policy_loss
                                + kl_coef * behavior_kl_mean
                                - entropy_coef * entropy_mean
                            )
                            (total_loss / float(micro_batches_per_global)).backward()
                            metric_sums["policy_loss"] += policy_loss.detach()
                            metric_sums["total_loss"] += total_loss.detach()
                            metric_sums["ratio"] += ratio.detach().mean()
                            metric_sums["ratio_abs"] += (ratio.detach() - 1.0).abs().mean()
                            metric_sums["clipped_ratio"] += clipped_ratio.detach().mean()
                            metric_sums["approx_kl"] += -raw_log_ratio.detach().mean()
                            metric_sums["clip_fraction"] += clip_mask.float().mean()
                            metric_sums["dual_clip_fraction"] += dual_clip_mask.float().mean()
                            metric_sums["entropy"] += entropy_mean.detach()
                            metric_sums["behavior_kl"] += behavior_kl_mean.detach()

                    forward_backward_steps += 1
                    progress_ops += 1
                    progress.set_status(
                        "phase=forward_backward "
                        f"epoch={epoch + 1}/{update_epochs} "
                        f"global_batch={global_batch_index + 1}/"
                        f"{global_batches_per_epoch} "
                        f"micro_batch={micro_index + 1}/"
                        f"{micro_batches_per_global} "
                        f"optimizer={optimizer_steps}/"
                        f"{optimizer_steps_total}"
                    )
                    progress.set(progress_ops)

                grad_norm = self._clip_or_measure_grad_norm(optim_cfg)
                optimizer.step()
                optimizer_steps += 1
                progress_ops += 1
                grad_norms.append(float(grad_norm))
                progress.set_status(
                    f"phase=optimizer epoch={epoch + 1}/{update_epochs} "
                    f"global_batch={global_batch_index + 1}/"
                    f"{global_batches_per_epoch} "
                    f"micro_batch={micro_batches_per_global}/"
                    f"{micro_batches_per_global} "
                    f"optimizer={optimizer_steps}/{optimizer_steps_total}"
                )
                progress.set(
                    progress_ops,
                    force=optimizer_steps == optimizer_steps_total,
                )

        policy.train(False)
        metric_divisor = float(max(1, forward_backward_steps))
        scalar_values = (
            torch.stack(tuple(metric_sums[name] / metric_divisor for name in metric_names))
            .detach()
            .float()
            .cpu()
            .tolist()
        )
        scalars = dict(zip(metric_names, scalar_values, strict=True))
        mean_grad_norm = _mean(grad_norms)
        return {
            **common_metrics,
            "actor/ppo_updates": float(optimizer_steps),
            "actor/ppo_optimizer_steps": float(optimizer_steps),
            "actor/ppo_forward_backward_steps": float(forward_backward_steps),
            "actor/ppo_progress_ops": float(progress_ops),
            "actor/policy_loss": float(scalars["policy_loss"]),
            "actor/loss": float(scalars["total_loss"]),
            "actor/total_loss": float(scalars["total_loss"]),
            "actor/ratio": float(scalars["ratio"]),
            "actor/ratio_mean": float(scalars["ratio"]),
            "actor/ratio_abs": float(scalars["ratio_abs"]),
            "actor/clipped_ratio": float(scalars["clipped_ratio"]),
            "actor/approx_kl": float(scalars["approx_kl"]),
            "actor/clip_fraction": float(scalars["clip_fraction"]),
            "actor/dual_clip_fraction": float(scalars["dual_clip_fraction"]),
            "actor/entropy_mean": float(scalars["entropy"]),
            "actor/behavior_kl_mean": float(scalars["behavior_kl"]),
            "actor/policy_grad_norm": mean_grad_norm,
            "actor/grad_norm": mean_grad_norm,
            "actor/lr": _optimizer_lr(optimizer),
            "actor/zero_loss_steps": float(zero_loss_micro_batches),
            "actor/zero_loss_micro_batches": float(zero_loss_micro_batches),
            "actor/skipped_zero_valid_update": 0.0,
        }

    def sync_model_to_rollout(
        self,
        key: str = "policy",
        version: int | None = None,
    ) -> dict[str, float]:
        """Push the current policy state to RolloutGroup through patch sync."""

        resolved_version = self.global_step if version is None else int(version)
        syncer_cfg = _as_plain_dict(self.train_cfg.get("syncer", {}))
        sync_dtype = dtype_from_precision(syncer_cfg.get("precision", "fp32"))
        export_start = time.perf_counter()
        if _is_fsdp_module(self._policy()):
            restore_offloaded = bool(self.enable_offload and self.is_weight_offloaded)
            self._load_parameters(onload_grad=False)
            try:
                sync_metrics = self._push_fsdp_sharded_buckets(
                    str(key),
                    int(resolved_version),
                    sync_dtype=sync_dtype,
                    bucket_bytes=int(syncer_cfg.get("bucket_bytes", 128 * 1024 * 1024)),
                )
            finally:
                if restore_offloaded:
                    self._offload_parameters(offload_grad=True)
            export_s = float(time.perf_counter() - export_start)
            push_s = float(sync_metrics.pop("push_s", 0.0))
            num_tensors = float(sync_metrics.pop("tensors", 0.0))
            num_bytes = float(sync_metrics.pop("bytes", 0.0))
            syncer_metrics = sync_metrics
        else:
            state = self.state_dict()
            export_s = float(time.perf_counter() - export_start)
            if sync_dtype is not torch.float32:
                state = {
                    name: (value.to(dtype=sync_dtype) if value.is_floating_point() else value)
                    for name, value in state.items()
                }
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
        metrics.update(self._memory_metrics("sync_end"))
        return metrics

    def _push_fsdp_sharded_buckets(
        self,
        key: str,
        version: int,
        *,
        sync_dtype: torch.dtype,
        bucket_bytes: int,
    ) -> dict[str, float]:
        """Gather and publish one FSDP parameter bucket at a time."""

        if bucket_bytes <= 0:
            raise ValueError("syncer.bucket_bytes must be positive")
        state = _export_policy_sharded_state_dict(self._policy())
        bucket: dict[str, torch.Tensor] = {}
        bucket_size = 0
        bucket_index = 0
        tensor_count = 0
        byte_count = 0
        push_s = 0.0
        syncer = self._syncer() if int(self.rank) == 0 else None

        def flush() -> None:
            nonlocal bucket, bucket_size, bucket_index, push_s
            if not bucket:
                return
            if syncer is not None:
                start = time.perf_counter()
                syncer.push_bucket(
                    key,
                    bucket,
                    version=int(version),
                    index=int(bucket_index),
                )
                push_s += time.perf_counter() - start
            bucket = {}
            bucket_size = 0
            bucket_index += 1

        for name, value in state.items():
            materialized = _materialize_sharded_state_value(
                value,
                rank=int(self.rank),
                device=self.torch_device,
            )
            if materialized is None:
                continue
            if materialized.is_floating_point() and sync_dtype is not torch.float32:
                materialized = materialized.to(dtype=sync_dtype)
            size = int(materialized.numel()) * int(materialized.element_size())
            if bucket and bucket_size + size > bucket_bytes:
                flush()
            bucket[str(name)] = materialized
            bucket_size += size
            tensor_count += 1
            byte_count += size
        flush()
        if syncer is not None:
            start = time.perf_counter()
            syncer.commit(
                key,
                version=int(version),
                num_buckets=int(bucket_index),
            )
            push_s += time.perf_counter() - start
        return {
            "push_s": float(push_s),
            "tensors": float(tensor_count),
            "bytes": float(byte_count),
            "sync/policy_bucket_count": float(bucket_index),
            "sync/policy_sharded_export": 1.0,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached CPU copy of the policy state."""

        restore_offloaded = bool(self.enable_offload and self.is_weight_offloaded)
        self._load_parameters(onload_grad=False)
        try:
            state = _export_policy_state_dict(self._policy())
            return {name: value.detach().cpu().clone() for name, value in state.items()}
        finally:
            if restore_offloaded:
                self._offload_parameters(offload_grad=True)

    def release_synced_model(
        self,
        key: str = "policy",
        version: int | None = None,
    ) -> dict[str, float]:
        """Release object-store buckets after RolloutGroup acknowledged them."""

        resolved_version = self.global_step if version is None else int(version)
        released = False
        if int(self.rank) == 0:
            released = self._syncer().release(
                str(key),
                version=int(resolved_version),
            )
        return {f"sync/{key}_buckets_released": float(released)}

    def rng_state_dict(self) -> dict[str, Any]:
        """Return this actor rank's Python, NumPy, torch, and CUDA RNG state."""

        return capture_rng_state()

    def load_rng_state_dict(self, states: object) -> None:
        """Strictly restore this actor's RNG state from a group-wide rank list."""

        state = select_rank_rng_state(states, int(self.rank))
        if state is None:
            raise RuntimeError(f"ActorGroup RNG checkpoint has no state for rank {self.rank}")
        restore_rng_state(state, strict=True)

    def optimizer_state_dict(self) -> dict[str, Any]:
        """Export the full policy optimizer state on actor rank zero."""

        return self._export_optimizer_state_dict(self._optimizer())

    def encoder_optimizer_state_dict(self) -> dict[str, Any]:
        """Export the encoder-SFT optimizer state on actor rank zero."""

        if self.encoder_optimizer is None:
            return {}
        return self._export_optimizer_state_dict(self.encoder_optimizer)

    def _export_optimizer_state_dict(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        restore_weights = bool(self.enable_offload and self.is_weight_offloaded)
        is_encoder = optimizer is self.encoder_optimizer
        restore_optimizer = bool(
            self.enable_offload
            and (self.is_encoder_optimizer_offloaded if is_encoder else self.is_optimizer_offloaded)
        )
        try:
            self._load_parameters(onload_grad=False)
            self._load_optimizer(optimizer)
            policy = self._policy()
            if _is_fsdp_module(policy):
                from torch.distributed.fsdp import FullyShardedDataParallel

                state = FullyShardedDataParallel.full_optim_state_dict(
                    policy,
                    optimizer,
                    rank0_only=True,
                )
            else:
                state = optimizer.state_dict()
            return _to_cpu_tree(state)
        finally:
            if restore_optimizer:
                self._offload_optimizer(optimizer)
            if restore_weights:
                self._offload_parameters(offload_grad=True)

    def _load_optimizer_state_dict(
        self,
        state: dict[str, Any],
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        policy = self._policy()
        resolved_optimizer = self._optimizer() if optimizer is None else optimizer
        if _is_fsdp_module(policy):
            from torch.distributed.fsdp import FullyShardedDataParallel

            full_state = state if int(self.rank) == 0 else None
            sharded = FullyShardedDataParallel.scatter_full_optim_state_dict(
                full_state,
                policy,
            )
            resolved_optimizer.load_state_dict(sharded)
            return
        resolved_optimizer.load_state_dict(state)

    def _build_optimizer(
        self,
        role: str,
        params: list[nn.Parameter],
    ) -> torch.optim.Optimizer:
        optimizer_cfgs = _as_plain_dict(self.train_cfg.get("optimizers", {}))
        policy_optim_cfg = _as_plain_dict(
            optimizer_cfgs.get(str(role), optimizer_cfgs.get("policy", {}))
        )
        optimizer_name = str(policy_optim_cfg.get("name", "adam")).strip().lower()
        optimizer_cls = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }.get(optimizer_name)
        if optimizer_cls is None:
            raise ValueError(
                f"EmbodiedFSDPActor {role} optimizer must be adam or adamw, got {optimizer_name!r}"
            )
        lr = float(policy_optim_cfg.get("lr", self.train_cfg.get("lr", 1e-4)))
        raw_betas = policy_optim_cfg.get("betas", (0.9, 0.999))
        if not isinstance(raw_betas, (list, tuple)) or len(raw_betas) != 2:
            raise ValueError("policy optimizer betas must contain exactly two values")
        betas = (float(raw_betas[0]), float(raw_betas[1]))
        eps = float(policy_optim_cfg.get("eps", 1e-8))
        weight_decay = float(policy_optim_cfg.get("weight_decay", 0.0))
        if not params:
            raise ValueError(f"EmbodiedFSDPActor {role} has no trainable parameters")
        return optimizer_cls(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def _eval_inputs_for_step(
        self,
        batch: TrajectoryBatch,
        step: int,
        *,
        batch_slice: slice | None = None,
    ) -> dict[str, torch.Tensor | str]:
        forward_inputs = batch.forward_inputs
        if "hidden" not in forward_inputs:
            raise ValueError("trajectory forward_inputs must include 'hidden'")
        selected = slice(None) if batch_slice is None else batch_slice
        eval_batch: dict[str, torch.Tensor | str] = {
            "mode": "evaluate",
            "hidden": forward_inputs["hidden"][step][selected].to(
                self.torch_device,
                dtype=self.policy_input_dtype,
            ),
            "action": batch.actions[step][selected].to(
                self.torch_device,
                dtype=torch.float32,
            ),
        }
        for key in _EXTRA_FORWARD_KEYS:
            if key in forward_inputs:
                eval_batch[key] = forward_inputs[key][step][selected].to(self.torch_device)
        return eval_batch

    def _eval_inputs_for_flat_indices(
        self,
        batch: TrajectoryBatch,
        flat_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor | str]:
        """Gather flattened ``[time, rollout]`` samples before device transfer."""

        forward_inputs = batch.forward_inputs
        if "hidden" not in forward_inputs:
            raise ValueError("trajectory forward_inputs must include 'hidden'")
        eval_batch: dict[str, torch.Tensor | str] = {
            "mode": "evaluate",
            "hidden": _flatten_time_batch(forward_inputs["hidden"])
            .index_select(0, flat_indices)
            .to(self.torch_device, dtype=self.policy_input_dtype),
            "action": _flatten_time_batch(batch.actions)
            .index_select(0, flat_indices)
            .to(self.torch_device, dtype=torch.float32),
        }
        for key in _EXTRA_FORWARD_KEYS:
            if key in forward_inputs:
                eval_batch[key] = (
                    _flatten_time_batch(forward_inputs[key])
                    .index_select(0, flat_indices)
                    .to(self.torch_device)
                )
        return eval_batch

    def _clip_or_measure_grad_norm(self, optim_cfg: dict[str, Any]) -> float:
        policy = self._policy()
        grad_clip_norm = optim_cfg.get("grad_clip_norm", None)
        if _is_fsdp_module(policy):
            max_norm = float("inf") if grad_clip_norm is None else float(grad_clip_norm)
            return float(_to_float(policy.clip_grad_norm_(max_norm)))
        params = [param for param in policy.parameters() if param.requires_grad]
        if grad_clip_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(params, float(grad_clip_norm))
            return float(_to_float(norm))
        return _grad_norm(params)

    def _load_parameters(self, *, onload_grad: bool) -> None:
        if not self.enable_offload or not self.is_weight_offloaded:
            if self.memory_phase == "uninitialized":
                self.memory_phase = "parameters_loaded"
            return
        if self.fsdp_manager is None:
            raise RuntimeError("Actor offload requires an FSDPModelManager")
        # Clear first so an OOM during a partial transfer can be cleaned by the
        # same offload path used after a successful phase.
        self.is_weight_offloaded = False
        self.memory_phase = "parameters_loaded"
        try:
            self.fsdp_manager.onload_param_and_grad(
                self._policy(),
                self.torch_device,
                onload_grad=bool(onload_grad),
            )
        except Exception:
            self._offload_parameters(offload_grad=True)
            raise

    def _offload_parameters(self, *, offload_grad: bool) -> None:
        if not self.enable_offload or self.is_weight_offloaded:
            return
        if self.fsdp_manager is None:
            raise RuntimeError("Actor offload requires an FSDPModelManager")
        self.fsdp_manager.offload_param_and_grad(
            self._policy(),
            offload_grad=bool(offload_grad),
        )
        self.is_weight_offloaded = True
        self.memory_phase = "offloaded"

    def _load_optimizer(
        self,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        resolved = self._optimizer() if optimizer is None else optimizer
        offloaded = (
            self.is_encoder_optimizer_offloaded
            if resolved is self.encoder_optimizer
            else self.is_optimizer_offloaded
        )
        if not self.enable_offload or not offloaded:
            return
        if self.fsdp_manager is None:
            raise RuntimeError("Actor offload requires an FSDPModelManager")
        if resolved is self.encoder_optimizer:
            self.is_encoder_optimizer_offloaded = False
        else:
            self.is_optimizer_offloaded = False
        try:
            self.fsdp_manager.onload_optimizer(
                resolved,
                self.torch_device,
            )
        except Exception:
            self._offload_optimizer(resolved)
            raise

    def _offload_optimizer(
        self,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        if not self.enable_offload:
            return
        resolved = self._optimizer() if optimizer is None else optimizer
        already_offloaded = (
            self.is_encoder_optimizer_offloaded
            if resolved is self.encoder_optimizer
            else self.is_optimizer_offloaded
        )
        if already_offloaded:
            return
        if self.fsdp_manager is None:
            raise RuntimeError("Actor offload requires an FSDPModelManager")
        self.fsdp_manager.offload_optimizer(resolved)
        if resolved is self.encoder_optimizer:
            self.is_encoder_optimizer_offloaded = True
        else:
            self.is_optimizer_offloaded = True

    def _load_for_training(self) -> None:
        try:
            self._load_parameters(onload_grad=True)
            self._load_optimizer()
            if self.encoder_optimizer is not None:
                self._load_optimizer(self.encoder_optimizer)
            self.memory_phase = "training_loaded"
        except Exception:
            # Phase acquisition is transactional: an optimizer onload can OOM
            # after parameters are already resident, so release every resource
            # acquired so far before propagating the original failure.
            self._offload_optimizer()
            if self.encoder_optimizer is not None:
                self._offload_optimizer(self.encoder_optimizer)
            self._offload_parameters(offload_grad=True)
            raise

    def _release_after_training(self) -> None:
        optimizer = self._optimizer()
        optimizer.zero_grad(set_to_none=True)
        self._offload_optimizer()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad(set_to_none=True)
            self._offload_optimizer(self.encoder_optimizer)
        self._offload_parameters(offload_grad=True)

    def _memory_metrics(self, phase: str) -> dict[str, float]:
        metrics = {
            f"actor/memory/{phase}_weights_offloaded": float(self.is_weight_offloaded),
            f"actor/memory/{phase}_optimizer_offloaded": float(self.is_optimizer_offloaded),
            f"actor/memory/{phase}_encoder_optimizer_offloaded": float(
                self.encoder_optimizer is None or self.is_encoder_optimizer_offloaded
            ),
        }
        if self.torch_device.type == "cuda" and torch.cuda.is_available():
            metrics[f"actor/memory/{phase}_allocated_bytes"] = float(
                torch.cuda.memory_allocated(self.torch_device)
            )
            metrics[f"actor/memory/{phase}_reserved_bytes"] = float(
                torch.cuda.memory_reserved(self.torch_device)
            )
        return metrics

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

    def _syncer(self) -> BucketWeightSyncer:
        if self.syncer is None:
            syncer_cfg = _as_plain_dict(self.train_cfg.get("syncer", {}))
            store_name = str(syncer_cfg.get("store_name", _DEFAULT_WEIGHT_STORE))
            self.syncer = BucketWeightSyncer(
                store_name=store_name,
                bucket_bytes=int(syncer_cfg.get("bucket_bytes", 128 * 1024 * 1024)),
            )
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
    return {str(name): torch.as_tensor(tensor).to(device) for name, tensor in dict(value).items()}


def _to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value


def _as_vector(value: Any) -> torch.Tensor:
    tensor = _as_tensor(value)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor
    return tensor.reshape(int(tensor.shape[0]), -1).sum(dim=1)


def _trailing_numel(shape: torch.Size | tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _flatten_time_batch(value: torch.Tensor) -> torch.Tensor:
    """Flatten only the leading ``[time, batch]`` dimensions."""

    if value.ndim < 2:
        raise ValueError(
            "trajectory tensor must have leading [time, batch] dimensions; "
            f"got {tuple(value.shape)}"
        )
    return value.reshape(
        int(value.shape[0]) * int(value.shape[1]),
        *value.shape[2:],
    )


def _local_source_indices(
    padded_indices: torch.Tensor,
    *,
    local_time_steps: int,
    local_rollouts: int,
) -> torch.Tensor:
    """Map distributed time-padding rows to safe local rows with a zero mask."""

    local_count = int(local_time_steps) * int(local_rollouts)
    if local_count <= 0:
        raise ValueError("local trajectory batch must not be empty")
    return torch.where(
        padded_indices < local_count,
        padded_indices,
        padded_indices.remainder(int(local_rollouts)),
    )


def _optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _match_logprob_shape(
    value: torch.Tensor,
    logprob: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if value.shape == logprob.shape:
        return value
    if value.ndim == 1 and logprob.ndim > 1 and int(value.shape[0]) == int(logprob.shape[0]):
        return _expand_batch_vector_as(value, logprob)
    raise ValueError(
        f"{name} shape must match log_prob shape; got "
        f"{tuple(value.shape)} and {tuple(logprob.shape)}"
    )


def _expand_batch_vector_as(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1:
        raise ValueError(f"expected a [batch] tensor, got {tuple(value.shape)}")
    if int(value.shape[0]) != int(target.shape[0]):
        raise ValueError(
            "batch vector size must match target batch size; "
            f"got {tuple(value.shape)} and {tuple(target.shape)}"
        )
    return value.reshape((int(value.shape[0]),) + (1,) * (target.ndim - 1))


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
    norms = [param.grad.detach().norm(2) for param in params if param.grad is not None]
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


def _distributed_min_int(value: int, device: torch.device) -> int:
    return _distributed_reduce_int(value, device, torch.distributed.ReduceOp.MIN)


def _distributed_sum_int(value: int, device: torch.device) -> int:
    return _distributed_reduce_int(value, device, torch.distributed.ReduceOp.SUM)


def _distributed_reduce_int(
    value: int,
    device: torch.device,
    op: Any,
) -> int:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
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


def _approx_behavior_kl(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    *,
    clip_log_ratio: float | None,
) -> torch.Tensor:
    log_ratio = new_logprob - old_logprob.detach()
    if clip_log_ratio is not None:
        limit = float(clip_log_ratio)
        log_ratio = log_ratio.clamp(min=-limit, max=limit)
    ratio = torch.exp(log_ratio)
    return ratio - 1.0 - log_ratio


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
    return {str(name): torch.as_tensor(value) for name, value in dict(state).items()}


def _export_policy_sharded_state_dict(policy: nn.Module) -> dict[str, Any]:
    if not _is_fsdp_module(policy):
        raise TypeError("sharded policy export requires an FSDP1 module")
    from torch.distributed.fsdp import (
        FullyShardedDataParallel,
        ShardedStateDictConfig,
        StateDictType,
    )

    cfg = ShardedStateDictConfig(offload_to_cpu=False)
    with FullyShardedDataParallel.state_dict_type(
        policy,
        StateDictType.SHARDED_STATE_DICT,
        cfg,
    ):
        return dict(policy.state_dict())


def _materialize_sharded_state_value(
    value: Any,
    *,
    rank: int,
    device: torch.device,
) -> torch.Tensor | None:
    try:
        from torch.distributed._shard.sharded_tensor import ShardedTensor
    except ImportError:  # pragma: no cover - torch version dependent
        ShardedTensor = ()  # type: ignore[assignment,misc]
    if isinstance(value, ShardedTensor):
        output = (
            torch.empty(tuple(value.size()), dtype=value.dtype, device=device)
            if int(rank) == 0
            else None
        )
        value.gather(dst=0, out=output, dtype=value.dtype)
        return None if output is None else output.detach().cpu()
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - torch version dependent
        DTensor = ()  # type: ignore[assignment,misc]
    if isinstance(value, DTensor):
        full = value.full_tensor()
        return full.detach().cpu() if int(rank) == 0 else None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone() if int(rank) == 0 else None
    tensor = torch.as_tensor(value)
    return tensor.detach().cpu().clone() if int(rank) == 0 else None


def _export_local_policy_state(policy: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot only this Actor rank's FSDP shards for transactional rollback."""

    state: dict[str, torch.Tensor] = {}
    for name, parameter in policy.named_parameters():
        state[f"parameter::{name}"] = parameter.detach().cpu().clone()
    for name, buffer in policy.named_buffers():
        state[f"buffer::{name}"] = buffer.detach().cpu().clone()
    return state


@torch.no_grad()
def _load_local_policy_state(
    policy: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    parameters = dict(policy.named_parameters())
    buffers = dict(policy.named_buffers())
    expected = {
        *(f"parameter::{name}" for name in parameters),
        *(f"buffer::{name}" for name in buffers),
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise RuntimeError(
            "local FSDP transaction state mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    for name, parameter in parameters.items():
        parameter.copy_(
            state[f"parameter::{name}"].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )
    for name, buffer in buffers.items():
        buffer.copy_(
            state[f"buffer::{name}"].to(
                device=buffer.device,
                dtype=buffer.dtype,
            )
        )


def _load_policy_state_dict(
    policy: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    rank: int,
) -> None:
    if _is_fsdp_module(policy):
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel,
            StateDictType,
        )

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        payload = state if int(rank) == 0 else {}
        with FullyShardedDataParallel.state_dict_type(
            policy,
            StateDictType.FULL_STATE_DICT,
            cfg,
        ):
            policy.load_state_dict(payload)
        return
    policy.load_state_dict(_to_device_state(state, next(policy.parameters()).device))


def _is_fsdp_module(policy: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except Exception:
        return False
    return isinstance(policy, FullyShardedDataParallel)


def _unwrap_policy_module(policy: nn.Module) -> nn.Module:
    """Return the user module while preserving FSDP as the forward boundary."""

    if _is_fsdp_module(policy):
        return policy.module
    return policy


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


_GRPO_HELPERS = _load_grpo_helpers()
_entropy_coef = _GRPO_HELPERS._entropy_coef
_group_advantage = _GRPO_HELPERS._group_advantage
_group_variance_mask = _GRPO_HELPERS.group_variance_mask


def _rollout_filter_mask(
    returns: torch.Tensor,
    algorithm_cfg: dict,
    group_size: int,
) -> tuple[torch.Tensor, int]:
    """Per-rollout keep mask for the GRPO loss, matching RLinf's trunk.

    ``filter_rewards`` (RLinf ``fsdp_actor_worker.filter_rewards``): drop whole
    groups whose ``reward_coef``-scaled mean summed reward falls outside
    ``[rewards_lower_bound, rewards_upper_bound]`` (all-fail / all-success
    degenerate groups). Falls back to the zero-variance filter, then to
    keeping everything.
    """
    if bool(algorithm_cfg.get("filter_rewards", False)):
        coef = float(algorithm_cfg.get("reward_coef", 1.0))
        lower = float(algorithm_cfg.get("rewards_lower_bound", float("-inf")))
        upper = float(algorithm_cfg.get("rewards_upper_bound", float("inf")))
        group_mean = (returns * coef).reshape(-1, int(group_size)).mean(dim=1)
        keep = (group_mean >= lower) & (group_mean <= upper)
        mask = keep.to(returns.dtype).repeat_interleave(int(group_size))
    elif bool(algorithm_cfg.get("filter_zero_variance_groups", False)):
        mask = _group_variance_mask(returns, int(group_size), eps=1e-6)
    else:
        mask = torch.ones_like(returns)
    return mask, int((mask <= 0.0).sum().item())


_ppo_clip_term = _GRPO_HELPERS._ppo_clip_term
_ppo_ratio = _GRPO_HELPERS._ppo_ratio

__all__ = ["EmbodiedFSDPActor"]
