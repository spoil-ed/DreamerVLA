"""Torchrun/NCCL distributed helpers for the standalone online DreamerVLA loop.

Extracted verbatim from online_dreamervla.py (P3 god-file split). Pure relocation — the
RynnVLA online main() and external importers (online_cotrain_runner, frozen_wm) keep working
via the re-export in online_dreamervla. Also the clean seam for RUN-01.
"""

from __future__ import annotations

import datetime as _dt
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _init_distributed() -> tuple[int, int, int, bool]:
    """Init NCCL process group from torchrun env vars; no-op for single-process.

    Returns: (rank, world_size, local_rank, is_dist).
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0, False
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        timeout_s = int(os.environ.get("DVLA_DDP_TIMEOUT_SEC", "600"))
        dist.init_process_group(
            backend="nccl", timeout=_dt.timedelta(seconds=timeout_s)
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank, True


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """Return underlying module from a DDP wrapper, or pass through."""
    return module.module if isinstance(module, DDP) else module


def _dist_barrier(*, local_rank: int) -> None:
    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _dist_all_reduce_flag(
    value: bool,
    *,
    device: torch.device,
    op: dist.ReduceOp,
    label: str,
    rank: int,
    env_step: int,
) -> bool:
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    try:
        dist.all_reduce(tensor, op=op)
    except Exception as exc:
        raise RuntimeError(
            f"DDP all_reduce failed at {label} on rank={rank} env_step={env_step}; "
            f"local_value={int(value)}. Check other rank logs for the first local "
            "exception or a hung environment step."
        ) from exc
    return bool(int(tensor.item()))


def _dist_all_reduce_int(
    value: int,
    *,
    device: torch.device,
    op: dist.ReduceOp,
    label: str,
    rank: int,
    env_step: int,
) -> int:
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    try:
        dist.all_reduce(tensor, op=op)
    except Exception as exc:
        raise RuntimeError(
            f"DDP all_reduce failed at {label} on rank={rank} env_step={env_step}; "
            f"local_value={int(value)}. Check other rank logs for the first local "
            "exception or a hung environment step."
        ) from exc
    return int(tensor.item())
