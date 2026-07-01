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


def test_manual_cotrain_rejects_negative_gpu_count() -> None:
    cfg = _cfg(ngpu=-1)
    with pytest.raises(ValueError, match="manual_cotrain.ngpu"):
        validate_cfg(cfg)


def test_manual_cotrain_rejects_negative_task_id() -> None:
    cfg = _cfg(task_id=-1)
    with pytest.raises(ValueError, match="manual_cotrain.task_id"):
        validate_cfg(cfg)


def test_manual_cotrain_rejects_negative_env_rollout_timeout() -> None:
    cfg = _cfg(env_rollout_timeout_s=-1.0)
    with pytest.raises(ValueError, match="manual_cotrain.env_rollout_timeout_s"):
        validate_cfg(cfg)


def test_manual_cotrain_rejects_fractional_negative_env_rollout_timeout() -> None:
    cfg = _cfg(env_rollout_timeout_s=-0.5)
    with pytest.raises(ValueError, match="manual_cotrain.env_rollout_timeout_s"):
        validate_cfg(cfg)


def test_manual_cotrain_rejects_negative_checkpoint_interval() -> None:
    cfg = _cfg(checkpoint_every=-1)
    with pytest.raises(ValueError, match="manual_cotrain.checkpoint_every"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ngpu", 1.5),
        ("task_id", 1.5),
        ("checkpoint_every", 1.5),
    ),
)
def test_manual_cotrain_rejects_fractional_discrete_non_negative_controls(
    field: str,
    value: float,
) -> None:
    cfg = _cfg(**{field: value})
    with pytest.raises(ValueError, match=f"manual_cotrain.{field}"):
        validate_cfg(cfg)


def test_manual_cotrain_rejects_non_divisible_chunk_steps() -> None:
    cfg = _cfg(max_steps_per_rollout_epoch=5, num_action_chunks=2)
    with pytest.raises(ValueError, match="must be divisible"):
        validate_cfg(cfg)


def test_manual_cotrain_rejects_non_positive_wm_rollout_multiplier() -> None:
    cfg = _cfg(wm_rollout_multiplier=0)
    with pytest.raises(ValueError, match="manual_cotrain.wm_rollout_multiplier"):
        validate_cfg(cfg)


@pytest.mark.parametrize("field", ("real_rollout_epoch", "wm_rollout_epoch"))
def test_manual_cotrain_rejects_non_positive_split_rollout_epochs(
    field: str,
) -> None:
    cfg = _cfg(**{field: 0})
    with pytest.raises(ValueError, match=f"manual_cotrain.{field}"):
        validate_cfg(cfg)


def test_manual_cotrain_rejects_bad_actor_fsdp_strategy() -> None:
    cfg = _cfg()
    cfg.actor.train_cfg.fsdp.strategy = "bad"
    with pytest.raises(ValueError, match="actor.train_cfg.fsdp.strategy"):
        validate_cfg(cfg)


def test_manual_cotrain_rejects_env_slots_not_divisible_by_actor_group_size() -> None:
    cfg = _cfg(ngpu=6, envs_per_worker=1)
    cfg.actor.train_cfg.algorithm_cfg = {"group_size": 8}

    with pytest.raises(ValueError, match="logical trajectory count"):
        validate_cfg(cfg)


def test_manual_cotrain_geometry_counts_rollout_epoch_for_multigpu_profile() -> None:
    cfg = _cfg(ngpu=6, envs_per_worker=2, rollout_epoch=16)
    cfg.actor.train_cfg.algorithm_cfg = {"group_size": 8}

    validate_cfg(cfg)


def test_manual_cotrain_geometry_counts_split_real_and_wm_rollout_epochs() -> None:
    cfg = _cfg(
        ngpu=4,
        envs_per_worker=2,
        rollout_epoch=16,
        real_rollout_epoch=1,
        wm_rollout_epoch=16,
    )
    cfg.actor.train_cfg.algorithm_cfg = {"group_size": 8}

    with pytest.raises(ValueError, match="real_rollout_epoch=1"):
        validate_cfg(cfg)

    cfg.manual_cotrain.real_rollout_epoch = 4
    validate_cfg(cfg)


def test_manual_cotrain_accepts_env_slots_divisible_by_actor_group_size() -> None:
    cfg = _cfg(ngpu=6, envs_per_worker=8)
    cfg.actor.train_cfg.algorithm_cfg = {"group_size": 8}

    validate_cfg(cfg)


def test_manual_cotrain_rejects_rollout_shorter_than_replay_sequence() -> None:
    cfg = _cfg(max_steps_per_rollout_epoch=8, num_action_chunks=8)
    cfg.replay = {"cfg": {"sequence_length": 12}}

    with pytest.raises(ValueError, match="replay.cfg.sequence_length"):
        validate_cfg(cfg)


def test_manual_cotrain_accepts_rollout_covering_replay_sequence() -> None:
    cfg = _cfg(max_steps_per_rollout_epoch=16, num_action_chunks=8)
    cfg.replay = {"cfg": {"sequence_length": 12}}

    validate_cfg(cfg)


def test_manual_cotrain_rejects_rollout_shorter_than_classifier_window() -> None:
    cfg = _cfg(max_steps_per_rollout_epoch=16, num_action_chunks=8)
    cfg.replay = {"cfg": {"sequence_length": 12}}
    cfg.learner.model_cfg = {
        "classifier": {"kwargs": {"window": 8, "chunk_size": 8}}
    }

    with pytest.raises(ValueError, match="classifier window"):
        validate_cfg(cfg)


def test_manual_cotrain_accepts_rollout_covering_classifier_window() -> None:
    cfg = _cfg(max_steps_per_rollout_epoch=64, num_action_chunks=8)
    cfg.replay = {"cfg": {"sequence_length": 12}}
    cfg.learner.model_cfg = {
        "classifier": {"kwargs": {"window": 8, "chunk_size": 8}}
    }

    validate_cfg(cfg)
