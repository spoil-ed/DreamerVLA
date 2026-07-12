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
    backend: str | None = None
    use_orig_params: bool = False
    sync_module_states: bool = False

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
            backend=self.backend,
            use_orig_params=self.use_orig_params,
            sync_module_states=self.sync_module_states,
        )

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return self.make_strategy().wrap_model(model)


__all__ = ["FSDPModelManager"]
