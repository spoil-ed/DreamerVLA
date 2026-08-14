"""Native torch DDP runner for offline VLA supervised fine-tuning."""

from __future__ import annotations

import contextlib
import random
import time
from collections.abc import Mapping
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from dreamervla.dataset.pi05_sft import (
    build_pi05_sft_dataloader,
    configured_download_endpoint,
    openpi_torch_loader,
    resolve_lerobot_source,
)
from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runtime.distributed import NopretokenizeSFTDistributedHelper


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


class VLASFTTrainingRunner(BaseRunner):
    """Train a Hydra-selected VLA with torchrun and native PyTorch DDP."""

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

    def setup(self) -> None:
        actor_cfg = self.cfg.actor
        distributed_cfg = actor_cfg.distributed
        strategy_name = str(distributed_cfg.strategy).lower()
        if strategy_name != "ddp":
            raise ValueError("VLA SFT supports only native torch DDP")
        self.distributed = NopretokenizeSFTDistributedHelper.initialize(
            strategy="ddp",
            fsdp_mixed_precision="bf16",
            enable_activation_checkpointing=False,
            nccl_timeout_seconds=int(distributed_cfg.get("nccl_timeout_seconds", 1800)),
        )
        self.device = self.distributed.resolve_device(str(self.cfg.training.device))
        seed = int(self.cfg.seed) + int(self.distributed.rank)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        policy = hydra.utils.instantiate(_policy_hydra_config(actor_cfg.policy_cfg))
        if not isinstance(policy, torch.nn.Module):
            raise TypeError("actor.policy_cfg must instantiate torch.nn.Module")
        policy.to(device=self.device)
        self.policy = self.distributed.wrap_trainable_module(
            policy,
            find_unused_parameters=bool(distributed_cfg.get("find_unused_parameters", False)),
            broadcast_buffers=bool(distributed_cfg.get("broadcast_buffers", False)),
            gradient_as_bucket_view=bool(distributed_cfg.get("gradient_as_bucket_view", True)),
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

        source = resolve_lerobot_source(self.cfg.data.train_data_paths)
        policy_kwargs = OmegaConf.to_container(actor_cfg.policy_cfg.kwargs, resolve=True)
        assert isinstance(policy_kwargs, dict)
        self.data_loader = build_pi05_sft_dataloader(
            model_path=str(policy_kwargs["model_path"]),
            data_paths=source,
            config_name=str(policy_kwargs.get("config_name", "pi05_libero")),
            micro_batch_size=int(actor_cfg.micro_batch_size),
            world_size=int(self.distributed.world_size),
            num_workers=int(self.cfg.data.num_workers),
            seed=int(self.cfg.seed),
            shuffle=bool(self.cfg.data.shuffle),
        )
        torch_loader = openpi_torch_loader(self.data_loader)
        generator = getattr(torch_loader, "generator", None)
        self._data_generator_state = None if generator is None else generator.get_state().clone()

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
                "dataset": source,
                "download_endpoint": configured_download_endpoint(),
                "distributed_backend": "torch.nn.parallel.DistributedDataParallel",
                "sft_alignment_source": getattr(policy, "alignment_source", None),
            }
        )
        self.resume(self.cfg)
        self._restore_data_iterator()

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
        batch = next(self.data_iterator)
        self._data_iter_offset += 1
        torch_loader = openpi_torch_loader(self.data_loader)
        if self._data_iter_offset >= len(torch_loader):
            self._data_epoch += 1
            self.epoch = self._data_epoch
            self._data_iter_offset = 0
            generator = getattr(torch_loader, "generator", None)
            if generator is not None:
                self._data_generator_state = generator.get_state().clone()
            set_epoch = getattr(getattr(torch_loader, "sampler", None), "set_epoch", None)
            if callable(set_epoch):
                set_epoch(int(self._data_epoch))
        return batch

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
        while self.global_step < max_steps:
            self.policy.train()
            loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            step_start = time.perf_counter()
            for micro_step in range(accumulation):
                batch = self._next_batch()
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

            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
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
            if checkpoint_every > 0 and self.global_step % checkpoint_every == 0:
                self.save_checkpoint(tag="latest")
            self.console_progress(self.global_step, max_steps, "pi05-sft", unit="step")

        save_at_end = bool(self.cfg.training.get("save_at_end", True))
        if save_at_end and (checkpoint_every <= 0 or self.global_step % checkpoint_every != 0):
            self.save_checkpoint(tag="latest")
        self.console_banner("VLA SFT", subtitle=f"completed step={self.global_step}", done=True)
        return history

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        """Save only the learned π0.5 delta; the base checkpoint stays immutable."""

        if key != "policy":
            return super()._state_dict_for_checkpoint(key, value)
        assert self.distributed is not None
        module = self.distributed.unwrap_module(value)
        # Avoid materializing/traversing the full 3.6B-parameter model state just
        # to discard the frozen VLM tensors.  The SFT delta contains trainable
        # parameters only; buffers belong to the immutable base checkpoint.
        return {
            name: parameter.detach()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }

    def _load_state_dict_from_checkpoint(
        self,
        key: str,
        value: Any,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if key != "policy":
            super()._load_state_dict_from_checkpoint(key, value, state_dict, **kwargs)
            return
        assert self.distributed is not None
        missing, unexpected = self.distributed.unwrap_module(value).load_state_dict(
            state_dict,
            strict=False,
        )
        trainable_names = {
            name
            for name, parameter in self.distributed.unwrap_module(value).named_parameters()
            if parameter.requires_grad
        }
        missing_trainable = trainable_names.intersection(missing)
        if missing_trainable or unexpected:
            raise RuntimeError(
                "π0.5 delta checkpoint mismatch: "
                f"missing_trainable={sorted(missing_trainable)[:5]} "
                f"unexpected={list(unexpected)[:5]}"
            )

    def teardown(self) -> None:
        try:
            super().teardown()
        finally:
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
