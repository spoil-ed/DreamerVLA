from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg


def _cfg(**manual_overrides):
    manual = {
        "ngpu": 0,
        "global_steps": 1,
        "learner_update_step": 1,
        "sync_every": 1,
        "rollout_epoch": 1,
        "max_steps_per_rollout_epoch": 4,
        "num_action_chunks": 2,
        "envs_per_worker": 1,
    }
    manual.update(manual_overrides)
    return OmegaConf.create(
        {
            "_target_": "dreamervla.runners.ManualCotrainRayRunner",
            "training": {"out_dir": "/tmp/dvla-config-test"},
            "logger": {"logger_backends": []},
            "cluster": {"num_nodes": 1},
            "manual_cotrain": manual,
            "actor": {
                "train_cfg": {
                    "fsdp": {"strategy": "none", "precision": "fp32"}
                }
            },
            "learner": {"train_cfg": {}},
        }
    )


def test_manual_cotrain_allows_zero_gpu() -> None:
    validate_cfg(_cfg(ngpu=0))


def test_manual_cotrain_rejects_non_divisible_chunk_steps() -> None:
    cfg = _cfg(max_steps_per_rollout_epoch=5, num_action_chunks=2)
    with pytest.raises(ValueError, match="must be divisible"):
        validate_cfg(cfg)


def test_manual_cotrain_rejects_bad_actor_fsdp_strategy() -> None:
    cfg = _cfg()
    cfg.actor.train_cfg.fsdp.strategy = "bad"
    with pytest.raises(ValueError, match="actor.train_cfg.fsdp.strategy"):
        validate_cfg(cfg)
