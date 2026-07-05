"""No-Ray torchrun multi-GPU smoke.

Run with:

    CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run \
      --standalone --nproc_per_node=2 \
      -m dreamervla.diagnostics.smoke_torchrun_multigpu

The smoke initializes a torch process group, binds each rank to LOCAL_RANK,
does one all-reduce, and exits without importing or starting Ray.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist


def main() -> None:
    """Run a minimal no-Ray distributed GPU communication check."""

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 2:
        raise RuntimeError("smoke_torchrun_multigpu requires WORLD_SIZE>=2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the multi-GPU smoke")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"visible CUDA devices ({torch.cuda.device_count()}) < WORLD_SIZE ({world_size})"
        )
    if "ray" in sys.modules:
        raise RuntimeError("Ray must not be imported in no-Ray torchrun smoke")

    torch.cuda.set_device(local_rank)
    backend = "nccl"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = float(world_size * (world_size + 1) // 2)
        if float(value.item()) != expected:
            raise RuntimeError(
                f"all_reduce mismatch on rank {rank}: {float(value.item())} != {expected}"
            )
        print(
            json.dumps(
                {
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "backend": backend,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "device_name": torch.cuda.get_device_name(local_rank),
                    "all_reduce_sum": float(value.item()),
                    "ray_imported": "ray" in sys.modules,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
