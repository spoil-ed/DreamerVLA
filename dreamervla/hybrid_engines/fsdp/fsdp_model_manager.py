"""Manual FSDP model wrapping utilities.

This is intentionally a configuration-driven helper: it exposes FSDP,
precision, CPU offload, and activation checkpointing switches, but does not
infer or tune them from available VRAM.
"""

from __future__ import annotations

import os
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
    backend: str | None = None

    def __post_init__(self) -> None:
        _dtype_from_precision(self.precision)

    @property
    def param_dtype(self) -> torch.dtype:
        return _dtype_from_precision(self.precision)

    def ensure_process_group(self) -> bool:
        """Initialize torch.distributed for explicit multi-worker FSDP runs."""

        normalized = str(self.strategy).strip().lower()
        if normalized in {"", "none", "ddp"}:
            return False
        if normalized not in {"fsdp", "fsdp1"}:
            raise ValueError(f"unsupported FSDP strategy: {self.strategy!r}")
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available for FSDP")
        if dist.is_initialized():
            return True

        world_size = _int_env("WORLD_SIZE", default=1)
        if world_size <= 1:
            return False

        missing = [
            name
            for name in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(
                "FSDP multi-worker setup requires rendezvous env vars: "
                f"{', '.join(missing)}"
            )

        backend = self.backend or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(
            backend=str(backend),
            rank=_int_env("RANK", default=0),
            world_size=world_size,
        )
        return True

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
        self.ensure_process_group()
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


def _int_env(key: str, *, default: int) -> int:
    value = os.environ.get(key)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{key} must be an integer, got {value!r}") from exc


__all__ = ["FSDPModelManager"]
