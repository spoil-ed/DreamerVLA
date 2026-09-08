"""Exercise local LeRobot v3 → OpenPI batches under torchrun without a model."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch.distributed as dist

from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.utils.integrations.openpi_imports import configure_openpi_jax_runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--assets-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _distributed_identity() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("gloo")
    if dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    return 0, 1


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load the requested number of local batches and gather rank timings."""

    rank, world_size = _distributed_identity()
    configure_openpi_jax_runtime()
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    register_dreamervla_resolvers()
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=pi05_libero_sft"])
    cfg.data.source = str(args.data_path)
    cfg.task.pi05.base_ckpt_path = str(args.model_path)
    cfg.task.pi05.assets_path = str(args.assets_path)
    cfg.task.pi05.action_horizon = int(args.action_horizon)
    cfg.data.loader.batch_size = int(args.micro_batch_size)
    cfg.data.loader.num_workers = int(args.num_workers)
    cfg.seed = int(args.seed)
    loader = instantiate(cfg.data.loader).build(world_size=world_size, rank=rank).data_loader
    iterator = iter(loader)
    durations: list[float] = []
    action_shape: list[int] | None = None
    for _ in range(int(args.num_batches)):
        started = time.perf_counter()
        _observation, actions = next(iterator)
        durations.append(time.perf_counter() - started)
        action_shape = [int(value) for value in actions.shape]
        if action_shape[0] != int(args.micro_batch_size):
            raise RuntimeError(
                f"rank {rank} received local batch {action_shape[0]}, "
                f"expected {args.micro_batch_size}"
            )
        if action_shape[1] != int(args.action_horizon):
            raise RuntimeError(
                f"rank {rank} received action horizon {action_shape[1]}, "
                f"expected {args.action_horizon}"
            )

    result = {
        "rank": rank,
        "world_size": world_size,
        "num_batches": int(args.num_batches),
        "action_shape": action_shape,
        "first_batch_s": durations[0],
        "max_batch_s": max(durations),
        "mean_batch_s": sum(durations) / len(durations),
    }
    if not dist.is_initialized():
        return [result]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    return [item for item in gathered if item is not None]


def main() -> None:
    """CLI entry point."""

    args = _parser().parse_args()
    if int(args.num_batches) <= 0:
        raise ValueError("--num-batches must be positive")
    results = run(args)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(json.dumps(results, indent=2, sort_keys=True))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
