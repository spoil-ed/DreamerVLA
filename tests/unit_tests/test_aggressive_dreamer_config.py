from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from dreamervla.config import validate_cfg


def _compose(experiment: str):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="train", overrides=[f"experiment={experiment}"])


def test_aggressive_dreamer_experiment_is_explicit_and_isolated() -> None:
    aggressive = _compose("openvla_libero_aggressive")
    original = _compose("openvla_libero")

    with pytest.warns(
        UserWarning,
        match="real_rollout_target_trajectories overrides the mainline baseline 32",
    ):
        validate_cfg(aggressive)
    validate_cfg(original)

    assert aggressive._target_ == "dreamervla.runners.DreamerRunner"
    assert aggressive.run.name == "openvla_libero_aggressive"
    assert aggressive.manual_cotrain.global_steps == 20
    assert aggressive.manual_cotrain.checkpoint_every == 5
    assert aggressive.manual_cotrain.eval_interval_global_steps == 5
    assert aggressive.manual_cotrain.eval_initial_global_step is True
    assert aggressive.manual_cotrain.eval_protocol.num_episodes_per_task == 10
    assert aggressive.manual_cotrain.real_rollout_target_trajectories == 64
    assert aggressive.manual_cotrain.wm_rollout_target_trajectories == 1024
    assert aggressive.manual_cotrain.max_policy_kl == 0.03
    assert aggressive.algorithm.group_size == 16
    assert aggressive.algorithm.entropy_bonus == 1.0e-3
    assert aggressive.actor.train_cfg.algorithm_cfg.group_size == 16
    assert aggressive.actor.train_cfg.algorithm_cfg.entropy_coef == 1.0e-3
    assert aggressive.actor.train_cfg.algorithm_cfg.ppo_update_epochs == 2
    assert aggressive.actor.train_cfg.lr == 1.0e-6
    assert aggressive.actor.train_cfg.optimizers.policy.lr == 1.0e-6
    assert aggressive.actor.train_cfg.global_batch_size == 16384
    assert aggressive.actor.train_cfg.micro_batch_size == 8
    assert aggressive.replay.cfg.capacity == 80000

    assert original.run.name == "openvla_libero"
    assert original.manual_cotrain.global_steps == 20_000
    assert original.manual_cotrain.checkpoint_every == 10
    assert original.manual_cotrain.eval_interval_global_steps == 10
    assert original.manual_cotrain.real_rollout_target_trajectories == 32
    assert original.manual_cotrain.max_policy_kl == 0.1
    assert original.algorithm.group_size == 8
    assert original.algorithm.entropy_bonus == 0.0
    assert original.actor.train_cfg.algorithm_cfg.ppo_update_epochs == 1
    assert original.actor.train_cfg.lr == 5.0e-7
    assert original.actor.train_cfg.optimizers.policy.lr == 5.0e-7
