"""FSDP2 (``fully_shard``) strategy."""

from __future__ import annotations

import torch
import torch.distributed as dist

from dreamervla.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase


class FSDP2Strategy(FSDPStrategyBase):
    """Per-parameter FSDP2 via ``torch.distributed._composable.fsdp.fully_shard``.

    Single-rank runs are a passthrough (checkpointing only). Real sharding needs
    a multi-GPU process group, so the ``fully_shard`` path is exercised only when
    ``WORLD_SIZE>1`` with rendezvous env set.
    """

    def fsdp_version(self) -> str:
        return "fsdp2"

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        self._apply_checkpointing(model)
        self.ensure_process_group()
        if not (dist.is_available() and dist.is_initialized()):
            return model

        try:
            from torch.distributed._composable.fsdp import (
                CPUOffloadPolicy,
                MixedPrecisionPolicy,
                fully_shard,
            )
        except ImportError as exc:  # pragma: no cover - depends on torch build
            raise RuntimeError(
                "FSDP2 (fully_shard) requires a newer torch build"
            ) from exc

        kwargs: dict[str, object] = {}
        if self.param_dtype is not torch.float32:
            kwargs["mp_policy"] = MixedPrecisionPolicy(param_dtype=self.param_dtype)
        if self.cpu_offload:
            kwargs["offload_policy"] = CPUOffloadPolicy()
        fully_shard(model, **kwargs)
        return model
