"""Pluggable FSDP strategy base + factory.

Mirrors RLinf's ``rlinf/hybrid_engines/fsdp/strategy`` shape: a base class with
a ``create`` factory that routes a config string to an FSDP1 / FSDP2 / no-shard
strategy. Manual, config-driven (no VRAM inference). Single-node verifiable: a
``WORLD_SIZE<=1`` run is a passthrough (still applies activation checkpointing).
"""

from __future__ import annotations

import gc
import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist


def dtype_from_precision(value: str) -> torch.dtype:
    normalized = str(value).strip().lower()
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"precision must be fp32, bf16, or fp16; got {value!r}")


def _int_env(key: str, *, default: int) -> int:
    value = os.environ.get(key)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{key} must be an integer, got {value!r}") from exc


class FSDPStrategyBase(ABC):
    """Common config + lifecycle for an FSDP wrapping strategy."""

    def __init__(
        self,
        *,
        precision: str = "fp32",
        cpu_offload: bool = False,
        activation_checkpointing: bool = False,
        enable_gradient_accumulation: bool = False,
        backend: str | None = None,
        use_orig_params: bool = False,
        sync_module_states: bool = False,
        sharding_strategy: str = "full_shard",
        forward_prefetch: bool = False,
        backward_prefetch: str | None = "backward_pre",
        limit_all_gathers: bool = True,
        require_layer_wrap: bool = False,
    ) -> None:
        self.precision = precision
        self.cpu_offload = bool(cpu_offload)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.enable_gradient_accumulation = bool(enable_gradient_accumulation)
        self.backend = backend
        self.use_orig_params = bool(use_orig_params)
        self.sync_module_states = bool(sync_module_states)
        self.sharding_strategy = str(sharding_strategy).strip().lower()
        self.forward_prefetch = bool(forward_prefetch)
        self.backward_prefetch = (
            None if backward_prefetch is None else str(backward_prefetch).strip().lower()
        )
        self.limit_all_gathers = bool(limit_all_gathers)
        self.require_layer_wrap = bool(require_layer_wrap)
        dtype_from_precision(precision)  # validate eagerly

    @property
    def param_dtype(self) -> torch.dtype:
        return dtype_from_precision(self.precision)

    @classmethod
    def create(cls, strategy: str, **kwargs) -> FSDPStrategyBase:
        from dreamervla.hybrid_engines.fsdp.strategy.fsdp import (
            FSDPStrategy,
            NoShardStrategy,
        )
        from dreamervla.hybrid_engines.fsdp.strategy.fsdp2 import FSDP2Strategy

        normalized = str(strategy).strip().lower()
        if normalized in {"", "none", "ddp"}:
            return NoShardStrategy(**kwargs)
        if normalized in {"fsdp", "fsdp1"}:
            return FSDPStrategy(**kwargs)
        if normalized == "fsdp2":
            return FSDP2Strategy(**kwargs)
        raise ValueError(f"unsupported FSDP strategy: {strategy!r}")

    @abstractmethod
    def fsdp_version(self) -> str:
        """Return a short tag for the wrapping flavor (none/fsdp1/fsdp2)."""

    @abstractmethod
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply checkpointing + (multi-rank) sharding; return the model."""

    def _apply_checkpointing(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.activation_checkpointing:
            enable = getattr(model, "gradient_checkpointing_enable", None)
            if not callable(enable):
                raise TypeError(
                    "activation_checkpointing requires the policy to expose "
                    "gradient_checkpointing_enable()"
                )
            enable()
        return model

    @torch.no_grad()
    def offload_param_and_grad(
        self,
        model: torch.nn.Module,
        offload_grad: bool,
    ) -> None:
        model.to("cpu")
        if offload_grad:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad = parameter.grad.to("cpu", non_blocking=True)
        self.clear_memory()

    @torch.no_grad()
    def onload_param_and_grad(
        self,
        model: torch.nn.Module,
        device: torch.device,
        onload_grad: bool,
    ) -> None:
        model.to(device)
        if onload_grad:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad = parameter.grad.to(device, non_blocking=True)
        self.clear_memory()

    @torch.no_grad()
    def offload_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self._move_optimizer_state(optimizer, torch.device("cpu"))
        self.clear_memory()

    @torch.no_grad()
    def onload_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self._move_optimizer_state(optimizer, device)
        self.clear_memory()

    @staticmethod
    def _move_optimizer_state(
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        for state in optimizer.state.values():
            for key, value in tuple(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

    @staticmethod
    def clear_memory() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    def ensure_process_group(self) -> bool:
        """Initialize torch.distributed for explicit multi-worker FSDP runs.

        Returns True when a group is (now or already) initialized; False on a
        single-rank run, where wrapping is a passthrough.
        """

        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available for FSDP")
        if dist.is_initialized():
            return True
        if _int_env("WORLD_SIZE", default=1) <= 1:
            return False
        missing = [
            name
            for name in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(
                f"FSDP multi-worker setup requires rendezvous env vars: {', '.join(missing)}"
            )
        backend = self.backend or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(
            backend=str(backend),
            rank=_int_env("RANK", default=0),
            world_size=_int_env("WORLD_SIZE", default=1),
        )
        return True
