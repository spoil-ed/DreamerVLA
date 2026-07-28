from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from dreamervla.config import validate_cfg
from dreamervla.runners.cotrain_runner import _split_actor_keyed_shard_counts


def _compose():
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name="train",
            overrides=["experiment=openvla_libero_task1_9_success_sft_extended"],
        )


def _compose_continual():
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name="train",
            overrides=["experiment=openvla_libero_task1_9_success_sft_continual"],
        )


def _compose_safe_tasks():
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name="train",
            overrides=[
                "experiment=openvla_libero_safe_tasks_success_sft_extended"
            ],
        )


def test_task1_9_success_sft_has_exact_global_trajectory_contract() -> None:
    cfg = _compose()

    with pytest.warns(UserWarning):
        validate_cfg(cfg)

    quota_task_ids = list(cfg.manual_cotrain.wm_success_quota_task_ids)
    assert quota_task_ids == list(range(1, 10))
    assert cfg.manual_cotrain.wm_success_quota_per_task == 8
    assert len(quota_task_ids) * cfg.manual_cotrain.wm_success_quota_per_task == 72
    assert cfg.env.wm.cfg.emit_actor_success_only is True
    assert cfg.env.wm.cfg.actor_success_quota_per_task == 1
    assert cfg.actor.train_cfg.success_sft.replicated_trajectory_batch is False
    assert cfg.actor.train_cfg.success_sft.epochs == 4
    assert cfg.actor.train_cfg.success_sft.optimizer_steps_per_epoch == 8
    assert list(cfg.manual_cotrain.eval_protocol.task_ids) == list(range(10))
    assert cfg.manual_cotrain.eval_protocol.num_episodes_per_task == 10


def test_task_quota_rejects_batched_retention_that_multiplies_dataset() -> None:
    cfg = _compose()
    cfg.env.wm.cfg.actor_success_quota_per_task = 8

    with pytest.raises(
        ValueError,
        match="actor_success_quota_per_task must be 1",
    ):
        validate_cfg(cfg)


def test_continual_success_sft_uses_eval_guard_and_final_checkpoint() -> None:
    cfg = _compose_continual()

    with pytest.warns(UserWarning):
        validate_cfg(cfg)

    assert cfg.manual_cotrain.global_steps == 2
    assert cfg.manual_cotrain.eval_interval_global_steps == 1
    assert cfg.manual_cotrain.rollback_on_eval_regression is True
    assert cfg.manual_cotrain.checkpoint_every == 2
    assert cfg.actor.train_cfg.success_sft.epochs == 8
    assert cfg.actor.train_cfg.success_sft.optimizer_steps_per_epoch == 8
    assert cfg.checkpoint.topk.monitor_key == "eval/accepted_success_rate"


def test_safe_task_success_sft_matches_classifier_calibration_contract() -> None:
    cfg = _compose_safe_tasks()

    with pytest.warns(UserWarning):
        validate_cfg(cfg)

    assert list(cfg.manual_cotrain.wm_success_quota_task_ids) == [1, 2, 4, 8]
    assert list(cfg.ray_data.task_ids) == [1, 2, 4, 8]
    assert cfg.manual_cotrain.wm_success_quota_per_task == 8
    assert cfg.env.wm.cfg.classifier_success_terminal_only is True
    assert cfg.env.wm.cfg.kwargs.success_consecutive_chunks == 2
    assert cfg.env.wm.cfg.kwargs.terminate_on_success is False
    assert cfg.actor.train_cfg.success_sft.epochs == 4
    assert cfg.actor.train_cfg.success_sft.optimizer_steps_per_epoch == 8


@pytest.mark.parametrize("actor_ranks", [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16])
def test_exact_72_trajectories_balance_across_actor_card_counts(actor_ranks: int) -> None:
    counts = _split_actor_keyed_shard_counts(
        real_shards=0,
        wm_shards=72,
        wm_shard_batch_size=16,
        actor_ranks=actor_ranks,
        group_size=16,
    )
    per_rank = [
        sum(count for key, count in rank_counts if key == "wm_env")
        for rank_counts in counts
    ]

    assert sum(per_rank) == 72
    assert min(per_rank) > 0
    assert max(per_rank) - min(per_rank) <= 1
