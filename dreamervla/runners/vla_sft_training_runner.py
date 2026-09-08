# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DreamerVLA lifecycle around RLinf's offline VLA SFT implementation.

The data construction and inner optimization step are migrated from
``official_sft_data_loader.py``, ``fsdp_vla_sft_worker.py``, and
``fsdp_sft_worker.py`` in RLinf.  ``BaseRunner`` remains the outer lifecycle so
the route follows DreamerVLA's Hydra, logging, and checkpoint contracts.
"""

from __future__ import annotations

import contextlib
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from dreamervla.models.embodiment.pi05.sft_data import (
    configured_download_endpoint,
    get_official_openpi_sft_num_batches,
    openpi_torch_loader,
)
from dreamervla.runners.base_runner import BaseRunner
from dreamervla.utils.training.distributed import NopretokenizeSFTDistributedHelper

_LEGACY_UNUSED_LM_HEAD_PREFIX = "model.paligemma_with_expert.gemma_expert.lm_head."


def _policy_hydra_config(value: Any) -> DictConfig:
    if value is None:
        raise ValueError("actor.policy_cfg is required for VLA SFT")
    if isinstance(value, DictConfig) and OmegaConf.select(value, "_target_", default=None):
        return value
    raw = OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value
    if not isinstance(raw, Mapping):
        raise TypeError("actor.policy_cfg must be a mapping")
    target = raw.get("target")
    kwargs = raw.get("kwargs", {})
    if not target or not isinstance(kwargs, Mapping):
        raise ValueError("actor.policy_cfg requires target and kwargs")
    return OmegaConf.create({"_target_": str(target), **dict(kwargs)})


def _copy_state_dict_containers(value: Any) -> Any:
    """Copy checkpoint containers without duplicating their tensor storage."""

    if isinstance(value, dict):
        return {key: _copy_state_dict_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_state_dict_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_state_dict_containers(item) for item in value)
    return value


def _is_legacy_unused_lm_head(name: object) -> bool:
    return isinstance(name, str) and name.startswith(_LEGACY_UNUSED_LM_HEAD_PREFIX)


def _strip_legacy_unused_lm_head_optimizer_state(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Drop optimizer-only names written before the unused head was frozen."""

    cleaned = _copy_state_dict_containers(state_dict)
    cleaned["state"] = {
        name: value
        for name, value in cleaned.get("state", {}).items()
        if not _is_legacy_unused_lm_head(name)
    }
    for group in cleaned.get("param_groups", []):
        group["params"] = [
            name for name in group.get("params", []) if not _is_legacy_unused_lm_head(name)
        ]
    return cleaned


class VLASFTTrainingRunner(BaseRunner):
    """Train a Hydra-selected OpenPI VLA with RLinf-style FSDP SFT."""

    runner_name = "vla_sft"
    runner_status = "current"
    runner_family = "training"
    include_keys = (
        "global_step",
        "epoch",
        "_data_epoch",
        "_data_iter_offset",
        "_data_generator_state",
    )
    exclude_keys = (
        "data_loader",
        "data_iterator",
        "distributed",
    )

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir)
        self.distributed: NopretokenizeSFTDistributedHelper | None = None
        self.device = torch.device("cpu")
        self.policy: torch.nn.Module | None = None
        self.policy_optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.data_loader: Any | None = None
        self.data_iterator: Any | None = None
        self._data_epoch = 0
        self._data_iter_offset = 0
        self._data_generator_state: torch.Tensor | None = None
        self._first_batch_synchronized = False

    def setup(self) -> None:
        actor_cfg = self.cfg.actor
        distributed_cfg = actor_cfg.distributed
        strategy_name = str(distributed_cfg.strategy).lower()
        if strategy_name not in {"ddp", "fsdp"}:
            raise ValueError("VLA SFT distributed strategy must be ddp or fsdp")
        self.distributed = NopretokenizeSFTDistributedHelper.initialize(
            strategy=strategy_name,
            fsdp_mixed_precision=str(distributed_cfg.get("mixed_precision", "bf16")),
            enable_activation_checkpointing=bool(
                distributed_cfg.get("gradient_checkpointing", False)
            ),
            nccl_timeout_seconds=int(distributed_cfg.get("nccl_timeout_seconds", 1800)),
            backend=str(distributed_cfg.get("backend", "nccl")),
        )
        self.device = self.distributed.resolve_device(str(self.cfg.training.device))
        seed = int(self.cfg.seed) + int(self.distributed.rank)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        loader_factory = hydra.utils.instantiate(self.cfg.data.loader)
        build_loader = getattr(loader_factory, "build", None)
        if not callable(build_loader):
            raise TypeError("data.loader must instantiate RLinf's OpenPI loader factory")
        loader_bundle = build_loader(
            rank=int(self.distributed.rank),
            world_size=int(self.distributed.world_size),
        )
        self.data_loader = getattr(loader_bundle, "data_loader", None)
        source = getattr(loader_bundle, "source", None)
        if self.data_loader is None or not isinstance(source, str):
            raise TypeError("RLinf OpenPI loader factory returned an invalid bundle")
        torch_loader = openpi_torch_loader(self.data_loader)
        generator = getattr(torch_loader, "generator", None)
        self._data_generator_state = None if generator is None else generator.get_state().clone()

        policy = hydra.utils.instantiate(_policy_hydra_config(actor_cfg.policy_cfg))
        if not isinstance(policy, torch.nn.Module):
            raise TypeError("actor.policy_cfg must instantiate torch.nn.Module")
        policy.to(device=self.device)
        self.policy = self.distributed.wrap_trainable_module(
            policy,
            find_unused_parameters=bool(distributed_cfg.get("find_unused_parameters", False)),
            broadcast_buffers=bool(distributed_cfg.get("broadcast_buffers", False)),
            gradient_as_bucket_view=bool(distributed_cfg.get("gradient_as_bucket_view", True)),
            init_sync=bool(distributed_cfg.get("init_sync", True)),
        )

        trainable = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("VLA SFT policy has no trainable parameters")
        optim_cfg = actor_cfg.optim
        self.policy_optimizer = torch.optim.AdamW(
            trainable,
            lr=float(optim_cfg.lr),
            betas=(float(optim_cfg.adam_beta1), float(optim_cfg.adam_beta2)),
            eps=float(optim_cfg.adam_eps),
            weight_decay=float(optim_cfg.weight_decay),
        )
        self.lr_scheduler = _build_lr_scheduler(self.policy_optimizer, optim_cfg)

        super().setup()
        self.append_model_summary(
            {
                "family": "pi05",
                "parameters": sum(parameter.numel() for parameter in policy.parameters()),
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in policy.parameters()
                    if parameter.requires_grad
                ),
                "frozen_unused_parameters": int(getattr(policy, "frozen_unused_parameters", 0)),
                "dataset": source,
                "download_endpoint": configured_download_endpoint(),
                "distributed_strategy": strategy_name,
                "distributed_backend": self.distributed.backend,
                "sft_alignment_source": getattr(policy, "alignment_source", None),
            }
        )
        self.resume(self.cfg)
        self._restore_data_iterator()
        # Rank 0 writes the manifest while other ranks can finish setup much
        # sooner.  Do not let them enter the first CUDA forward independently.
        self.distributed.object_barrier()

    @property
    def gradient_accumulation(self) -> int:
        assert self.distributed is not None
        numerator = int(self.cfg.actor.global_batch_size)
        denominator = int(self.cfg.actor.micro_batch_size) * int(self.distributed.world_size)
        if numerator % denominator != 0:
            raise ValueError(
                "actor.global_batch_size must be divisible by micro_batch_size * world_size"
            )
        return numerator // denominator

    def _restore_data_iterator(self) -> None:
        assert self.data_loader is not None
        torch_loader = openpi_torch_loader(self.data_loader)
        generator = getattr(torch_loader, "generator", None)
        if generator is not None and self._data_generator_state is not None:
            generator.set_state(self._data_generator_state)
        sampler = getattr(torch_loader, "sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(self._data_epoch))
        self.data_iterator = iter(self.data_loader)
        for _ in range(int(self._data_iter_offset)):
            next(self.data_iterator)

    def _next_batch(self) -> Any:
        assert self.data_iterator is not None and self.data_loader is not None
        num_batches = get_official_openpi_sft_num_batches(self.data_loader)
        if self._data_iter_offset >= num_batches:
            self._start_next_data_epoch()
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self._start_next_data_epoch()
            assert self.data_iterator is not None
            batch = next(self.data_iterator)
        self._data_iter_offset += 1
        return batch

    def _synchronize_first_batch(self) -> None:
        """Align the one-time OpenPI worker cold start across DDP ranks."""

        assert self.distributed is not None
        if self._first_batch_synchronized:
            return
        self.distributed.object_barrier()
        self._first_batch_synchronized = True

    def _start_next_data_epoch(self) -> None:
        """Advance OpenPI's infinite wrapper with explicit sampler epoch state."""

        assert self.data_loader is not None
        self._data_epoch += 1
        self.epoch = self._data_epoch
        self._data_iter_offset = 0
        torch_loader = openpi_torch_loader(self.data_loader)
        generator = getattr(torch_loader, "generator", None)
        if generator is not None:
            self._data_generator_state = generator.get_state().clone()
        set_epoch = getattr(getattr(torch_loader, "sampler", None), "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(self._data_epoch))
        self.data_iterator = iter(self.data_loader)

    def _milestone_checkpoint_paths(self, step: int) -> tuple[Path, ...]:
        milestones = {int(value) for value in (self.cfg.training.get("milestone_steps", []) or [])}
        if int(step) not in milestones:
            return ()
        return (self.get_checkpoint_dir() / f"step={int(step)}.ckpt",)

    def run(self) -> list[dict[str, float]]:
        assert self.policy is not None
        assert self.policy_optimizer is not None
        assert self.lr_scheduler is not None
        assert self.distributed is not None
        max_steps = int(self.cfg.training.max_steps)
        log_every = max(1, int(self.cfg.training.log_every))
        checkpoint_every = int(self.cfg.training.checkpoint_every)
        grad_clip = float(self.cfg.actor.optim.clip_grad)
        accumulation = self.gradient_accumulation
        history: list[dict[str, float]] = []
        self.console_banner(
            "VLA SFT",
            subtitle=f"π0.5 steps={max_steps} global_batch={self.cfg.actor.global_batch_size}",
        )
        start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.policy_optimizer.zero_grad(set_to_none=True)
        last_checkpoint_step: int | None = None
        while self.global_step < max_steps:
            self.policy.train()
            loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            step_start = time.perf_counter()
            for micro_step in range(accumulation):
                batch = self._next_batch()
                # The first OpenPI batch starts spawned video workers and can
                # have substantial one-time rank skew.  Align ranks after
                # every process owns a batch and before DDP forward.
                self._synchronize_first_batch()
                last_micro = micro_step + 1 == accumulation
                no_sync = getattr(self.policy, "no_sync", None)
                sync_context = (
                    contextlib.nullcontext() if last_micro or not callable(no_sync) else no_sync()
                )
                with sync_context:
                    output = self.policy(
                        {
                            "mode": "sft",
                            "data": batch,
                            "use_action_chunk_loss": bool(
                                self.cfg.actor.get("use_action_chunk_loss", False)
                            ),
                        }
                    )
                    loss = output[0] if isinstance(output, tuple) else output
                    if isinstance(loss, Mapping):
                        loss = loss["loss"]
                    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                        raise TypeError("π0.5 SFT forward must return one scalar loss")
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"π0.5 SFT loss is not finite: {loss}")
                    (loss / float(accumulation)).backward()
                loss_sum += loss.detach().float()

            grad_norm = self.distributed.clip_grad_norm_tensor(self.policy, grad_clip)
            self.policy_optimizer.step()
            self.policy_optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            self.global_step += 1

            local_metrics: dict[str, float | torch.Tensor] = {
                "train/sft_loss": loss_sum / float(accumulation),
                "train/grad_norm": grad_norm,
                "train/learning_rate": float(self.policy_optimizer.param_groups[0]["lr"]),
                "time/step_s": time.perf_counter() - step_start,
            }
            if self.device.type == "cuda":
                gib = float(1024**3)
                local_metrics.update(
                    {
                        "train/gpu_peak_allocated_gib": (
                            torch.cuda.max_memory_allocated(self.device) / gib
                        ),
                        "train/gpu_peak_reserved_gib": (
                            torch.cuda.max_memory_reserved(self.device) / gib
                        ),
                    }
                )
            metrics = self.distributed.reduce_mean_dict(local_metrics)
            metrics["global_step"] = float(self.global_step)
            history.append(metrics)
            if self.global_step % log_every == 0 or self.global_step == max_steps:
                self.log_metrics(metrics, step=self.global_step)
                self.console_metric_table(
                    step=max(0, self.global_step - 1),
                    total_steps=max_steps,
                    elapsed_s=time.perf_counter() - start,
                    metrics=metrics,
                )
            milestone_paths = self._milestone_checkpoint_paths(self.global_step)
            periodic_checkpoint = checkpoint_every > 0 and self.global_step % checkpoint_every == 0
            if periodic_checkpoint or milestone_paths:
                self.save_checkpoint(tag="latest", extra_paths=milestone_paths)
                last_checkpoint_step = self.global_step
            self.console_progress(self.global_step, max_steps, "pi05-sft", unit="step")

        save_at_end = bool(self.cfg.training.get("save_at_end", True))
        if save_at_end and last_checkpoint_step != self.global_step:
            milestone_paths = self._milestone_checkpoint_paths(self.global_step)
            self.save_checkpoint(tag="latest", extra_paths=milestone_paths)
        self.console_banner("VLA SFT", subtitle=f"completed step={self.global_step}", done=True)
        return history

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        """Save only the learned π0.5 delta; the base checkpoint stays immutable."""

        if key == "policy_optimizer":
            assert self.policy is not None and self.distributed is not None
            return get_optimizer_state_dict(
                self.policy,
                value,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        if key != "policy":
            return super()._state_dict_for_checkpoint(key, value)
        assert self.distributed is not None
        module = self.distributed.unwrap_module(value)
        trainable_names = {
            name for name, parameter in module.named_parameters() if parameter.requires_grad
        }
        full_state = get_model_state_dict(
            value,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        # ``get_model_state_dict`` normalizes DDP/FSDP wrapper names to the
        # unwrapped module's canonical FQNs. FSDP intentionally returns an
        # empty mapping on nonzero ranks when full_state_dict and cpu_offload
        # are both enabled, so only rank 0 validates and materializes the delta.
        if not self.distributed.is_main_process:
            return {}
        missing_trainable = trainable_names.difference(full_state)
        if missing_trainable:
            raise RuntimeError(
                "pi05 delta checkpoint is missing trainable parameters: "
                f"{sorted(missing_trainable)[:5]}"
            )
        policy_state = {name: full_state[name] for name in trainable_names}
        if trainable_names and not policy_state:
            raise RuntimeError("pi05 delta checkpoint is empty")
        return policy_state

    def _load_state_dict_from_checkpoint(
        self,
        key: str,
        value: Any,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if key == "policy_optimizer":
            assert self.policy is not None and self.distributed is not None
            optimizer_state = state_dict.get("state", {})
            parameter_keys = list(optimizer_state)
            if not parameter_keys:
                parameter_keys = [
                    parameter
                    for group in state_dict.get("param_groups", [])
                    for parameter in group.get("params", [])
                ]
            if parameter_keys and all(isinstance(name, str) for name in parameter_keys):
                canonical_state = _strip_legacy_unused_lm_head_optimizer_state(state_dict)
                set_optimizer_state_dict(
                    self.policy,
                    value,
                    canonical_state,
                    options=StateDictOptions(full_state_dict=True),
                )
            else:
                # Checkpoints written before canonical distributed state dicts
                # used integer optimizer parameter IDs under DDP.
                self.distributed.load_optimizer_state_dict(self.policy, value, state_dict)
            return
        if key != "policy":
            super()._load_state_dict_from_checkpoint(key, value, state_dict, **kwargs)
            return
        assert self.distributed is not None
        module = self.distributed.unwrap_module(value)
        trainable_names = {
            name for name, parameter in module.named_parameters() if parameter.requires_grad
        }
        filtered_state = {
            name: tensor
            for name, tensor in state_dict.items()
            if not _is_legacy_unused_lm_head(name)
        }
        checkpoint_names = set(filtered_state)
        missing_trainable = trainable_names.difference(checkpoint_names)
        unexpected = checkpoint_names.difference(trainable_names)
        if missing_trainable or unexpected:
            raise RuntimeError(
                "π0.5 delta checkpoint mismatch: "
                f"missing_trainable={sorted(missing_trainable)[:5]} "
                f"unexpected={sorted(unexpected)[:5]}"
            )
        incompatible = set_model_state_dict(
            value,
            filtered_state,
            options=StateDictOptions(full_state_dict=True, strict=False),
        )
        unexpected_loaded = set(incompatible.unexpected_keys)
        if unexpected_loaded:
            raise RuntimeError(
                "π0.5 delta checkpoint produced unexpected policy keys: "
                f"{sorted(unexpected_loaded)[:5]}"
            )

    def teardown(self) -> None:
        try:
            super().teardown()
        finally:
            close_loader = getattr(self.data_loader, "close", None)
            if callable(close_loader):
                close_loader()
            self.data_iterator = None
            if self.distributed is not None:
                self.distributed.cleanup()


__all__ = ["VLASFTTrainingRunner"]


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    optim_cfg: DictConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Port RLinf's effective constant/cosine warmup scheduler contract."""

    scheduler_name = str(optim_cfg.get("lr_scheduler", "constant")).lower()
    warmup_steps = int(optim_cfg.lr_warmup_steps)
    total_steps = int(optim_cfg.total_training_steps)

    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if scheduler_name == "constant":
            return 1.0
        if scheduler_name == "cosine":
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        raise ValueError(f"unsupported VLA SFT lr_scheduler: {scheduler_name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
