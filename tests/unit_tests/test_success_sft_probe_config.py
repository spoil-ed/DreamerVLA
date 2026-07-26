from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from dreamervla.config import validate_cfg


def _compose():
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name="train",
            overrides=["experiment=openvla_libero_success_sft_probe"],
        )


def test_success_sft_probe_is_one_step_frozen_and_self_verifying() -> None:
    cfg = _compose()

    with pytest.warns(
        UserWarning,
        match="real_rollout_target_trajectories overrides the mainline baseline 32",
    ):
        validate_cfg(cfg)

    assert cfg._target_ == "dreamervla.runners.DreamerRunner"
    assert cfg.manual_cotrain.training_mode == "imagined_success_sft"
    assert cfg.manual_cotrain.learner_updates_enabled is False
    assert cfg.manual_cotrain.staged_policy_update is False
    assert cfg.manual_cotrain.require_training_signal is True
    assert cfg.manual_cotrain.global_steps == 1
    assert cfg.manual_cotrain.checkpoint_every == 0
    assert cfg.manual_cotrain.eval_interval_global_steps == 1
    assert cfg.manual_cotrain.eval_initial_global_step is True
    assert list(cfg.manual_cotrain.eval_protocol.task_ids) == list(range(10))
    assert cfg.manual_cotrain.eval_protocol.num_episodes_per_task == 10
    assert cfg.manual_cotrain.eval_protocol.num_envs == 25
    assert cfg.manual_cotrain.eval_protocol.compute_diagnostics is False
    assert cfg.manual_cotrain.real_rollout_target_trajectories == 1
    assert cfg.manual_cotrain.wm_rollout_target_trajectories == 192
    assert cfg.manual_cotrain.max_steps_per_rollout_epoch == 512
    assert cfg.manual_cotrain.max_policy_kl == pytest.approx(0.10)
    assert cfg.learner.train_cfg.classifier_threshold is None
    assert cfg.learner.train_cfg.require_trajectory_classifier_threshold is True
    assert cfg.actor.train_cfg.lr == pytest.approx(1.0e-6)
    assert cfg.actor.train_cfg.global_batch_size == 12288
    assert cfg.actor.train_cfg.success_sft.epochs == 1
