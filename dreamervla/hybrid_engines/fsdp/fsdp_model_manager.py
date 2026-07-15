"""Manual FSDP model wrapping utilities.

This is intentionally a configuration-driven helper: it exposes FSDP,
precision, CPU offload, and activation checkpointing switches, but does not
infer or tune them from available VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dreamervla.hybrid_engines.fsdp.strategy import (
    FSDPStrategyBase,
    dtype_from_precision,
)


@dataclass(frozen=True)
class FSDPModelManager:
    """Prepare models for manual FSDP-style training."""

    strategy: str = "none"
    precision: str = "fp32"
    cpu_offload: bool = False
    activation_checkpointing: bool = False
    enable_gradient_accumulation: bool = False
    backend: str | None = None
    use_orig_params: bool = False
    sync_module_states: bool = False
    sharding_strategy: str = "full_shard"
    forward_prefetch: bool = False
    backward_prefetch: str | None = "backward_pre"
    limit_all_gathers: bool = True
    require_layer_wrap: bool = False

    def __post_init__(self) -> None:
        self.make_strategy()

    @property
    def param_dtype(self) -> torch.dtype:
        return dtype_from_precision(self.precision)

    def ensure_process_group(self) -> bool:
        """Initialize torch.distributed for explicit multi-worker FSDP runs."""

        normalized = str(self.strategy).strip().lower()
        if normalized in {"", "none", "ddp"}:
            return False
        return self.make_strategy().ensure_process_group()

    def make_strategy(self) -> FSDPStrategyBase:
        """Build the pluggable FSDP strategy described by this config."""

        return FSDPStrategyBase.create(
            self.strategy,
            precision=self.precision,
            cpu_offload=self.cpu_offload,
            activation_checkpointing=self.activation_checkpointing,
            enable_gradient_accumulation=self.enable_gradient_accumulation,
            backend=self.backend,
            use_orig_params=self.use_orig_params,
            sync_module_states=self.sync_module_states,
            sharding_strategy=self.sharding_strategy,
            forward_prefetch=self.forward_prefetch,
            backward_prefetch=self.backward_prefetch,
            limit_all_gathers=self.limit_all_gathers,
            require_layer_wrap=self.require_layer_wrap,
        )

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return self.make_strategy().wrap_model(model)

    def offload_param_and_grad(
        self,
        model: torch.nn.Module,
        *,
        offload_grad: bool,
    ) -> None:
        self.make_strategy().offload_param_and_grad(model, offload_grad)

    def onload_param_and_grad(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        onload_grad: bool,
    ) -> None:
        self.make_strategy().onload_param_and_grad(model, device, onload_grad)

    def offload_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.make_strategy().offload_optimizer(optimizer)

    def onload_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.make_strategy().onload_optimizer(optimizer, device)


__all__ = ["FSDPModelManager"]
