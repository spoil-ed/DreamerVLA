"""Manual FSDP model wrapping utilities.

This is intentionally a configuration-driven helper: it exposes FSDP,
precision, CPU offload, and activation checkpointing switches, but does not
infer or tune them from available VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


def _dtype_from_precision(value: str) -> torch.dtype:
    normalized = str(value).strip().lower()
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"precision must be fp32, bf16, or fp16; got {value!r}")


@dataclass(frozen=True)
class FSDPModelManager:
    """Prepare models for manual FSDP-style training."""

    strategy: str = "none"
    precision: str = "fp32"
    cpu_offload: bool = False
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        _dtype_from_precision(self.precision)

    @property
    def param_dtype(self) -> torch.dtype:
        return _dtype_from_precision(self.precision)

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.activation_checkpointing and hasattr(
            model, "gradient_checkpointing_enable"
        ):
            model.gradient_checkpointing_enable()

        normalized = str(self.strategy).strip().lower()
        if normalized in {"", "none", "ddp"}:
            return model
        if normalized not in {"fsdp", "fsdp1"}:
            raise ValueError(f"unsupported FSDP strategy: {self.strategy!r}")
        if not (dist.is_available() and dist.is_initialized()):
            return model

        from torch.distributed.fsdp import (
            CPUOffload,
            FullyShardedDataParallel,
            MixedPrecision,
        )

        mixed_precision = None
        if self.param_dtype is not torch.float32:
            mixed_precision = MixedPrecision(
                param_dtype=self.param_dtype,
                reduce_dtype=self.param_dtype,
                buffer_dtype=self.param_dtype,
            )
        return FullyShardedDataParallel(
            model,
            cpu_offload=CPUOffload(offload_params=bool(self.cpu_offload)),
            mixed_precision=mixed_precision,
        )


__all__ = ["FSDPModelManager"]
