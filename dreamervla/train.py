# ruff: noqa: E402
from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from dreamervla.config import validate_cfg
from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.runners.base_runner import BaseRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _auto_apply_distributed(cfg: DictConfig) -> None:
    """Force DDP when launched under torchrun with more than one process."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    try:
        world_size = int(os.environ["WORLD_SIZE"])
    except ValueError:
        return
    if world_size <= 1:
        return
    training = cfg.get("training") if hasattr(cfg, "get") else None
    if training is None:
        return
    if "distributed_strategy" in training:
        training.distributed_strategy = "ddp"
    if "data_parallel" in training:
        training.data_parallel = False


def run(cfg: DictConfig) -> None:
    register_dreamervla_resolvers()
    _auto_apply_distributed(cfg)
    OmegaConf.resolve(cfg)
    cfg = validate_cfg(cfg)
    runner_cls = hydra.utils.get_class(cfg._target_)
    runner: BaseRunner = runner_cls(cfg)
    runner.setup()
    try:
        runner.execute()
    finally:
        runner.teardown()


def _parse_hydra_like_args(argv: list[str]) -> tuple[str, list[str]]:
    config_name = "train"
    overrides: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            print(
                "Usage: python -m dreamervla.train --config-name CONFIG [overrides]\n\n"
                "Examples:\n"
                "  python -m dreamervla.train experiment=openvla_onetraj_libero_cotrain task=openvla_onetraj_coldstart_libero\n"
                "  python -m dreamervla.train experiment=collect_rollouts\n"
                "  python -m dreamervla.train experiment=eval_libero_vla"
            )
            raise SystemExit(0)
        if arg in ("--config-name", "-cn"):
            if i + 1 >= len(argv):
                raise SystemExit(f"{arg} requires a value")
            config_name = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
            i += 1
            continue
        overrides.append(arg)
        i += 1
    return config_name, overrides


def main(argv: list[str] | None = None) -> None:
    register_dreamervla_resolvers()
    config_name, overrides = _parse_hydra_like_args(
        list(sys.argv[1:] if argv is None else argv)
    )
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configs"), version_base=None
    ):
        cfg = compose(config_name=config_name, overrides=overrides)
    run(cfg)


if __name__ == "__main__":
    main()
