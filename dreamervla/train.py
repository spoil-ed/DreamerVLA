# ruff: noqa: E402
from __future__ import annotations

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from dreamervla.config import validate_cfg
from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.runners.base_runner import BaseRunner

register_dreamervla_resolvers()

logger = logging.getLogger(__name__)


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
    try:
        runner.setup()
    except BaseException:
        try:
            runner.teardown_after_setup_failure()
        except BaseException:
            logger.exception("Runner cleanup failed after setup error")
        raise
    try:
        runner.execute()
    finally:
        runner.teardown()


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
