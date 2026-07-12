"""FSDP1 + no-shard strategies."""

from __future__ import annotations

import torch
import torch.distributed as dist

from dreamervla.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase


class NoShardStrategy(FSDPStrategyBase):
    """``none``/``ddp``: no parameter sharding (DDP handled outside this helper)."""

    def fsdp_version(self) -> str:
        return "none"

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return self._apply_checkpointing(model)


class FSDPStrategy(FSDPStrategyBase):
    """Classic ``torch.distributed.fsdp.FullyShardedDataParallel`` wrapping."""

    def fsdp_version(self) -> str:
        return "fsdp1"

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        self._apply_checkpointing(model)
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
            use_orig_params=bool(self.use_orig_params),
            sync_module_states=bool(self.sync_module_states),
        )
