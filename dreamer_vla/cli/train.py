# ruff: noqa: E402
from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dreamer_vla.runners import BaseRunner


def _auto_apply_distributed(cfg: DictConfig) -> None:
    """If launched under torchrun (RANK + WORLD_SIZE>1 in env), force DDP.

    Lets shell wrappers stay distribution-agnostic: a single ``python -m
    dreamer_vla.cli.train`` invocation works for both single-GPU and torchrun launches,
    no need to forward ``training.distributed_strategy=ddp`` etc. by hand.
    """
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
    # Resolve config
    _auto_apply_distributed(cfg)
    OmegaConf.resolve(cfg)
    # Runner class
    cls = hydra.utils.get_class(cfg._target_)
    runner: BaseRunner = cls(cfg)
    # Run through the public lifecycle API.
    runner.setup()
    try:
        runner.execute()
    finally:
        runner.teardown()


def _parse_hydra_like_args(argv: list[str]) -> tuple[str, list[str]]:
    config_name = "world_model_dinowm_chunk"
    overrides: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            print(
                "Usage: python -m dreamer_vla.cli.train --config-name CONFIG [overrides]\n\n"
                "Examples:\n"
                "  python -m dreamer_vla.cli.train --config-name vla_rynnvla_action_head training.num_epochs=5\n"
                "  python -m dreamer_vla.cli.train --config-name world_model_dinowm_chunk training.max_steps=10\n"
                "  python -m dreamer_vla.cli.train --config-name dreamervla_rynn_dino_wm_actor_critic task=libero_object"
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
