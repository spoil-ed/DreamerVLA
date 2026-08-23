from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.runners.dreamer_runner import DreamerRunner


def _compose_experiment(experiment: str, *overrides: str):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name="train",
            overrides=[f"experiment={experiment}", *overrides],
        )


def test_debug_profile_owns_cotrain_budget() -> None:
    cfg = _compose_experiment("openvla_libero", "profile=debug")

    assert cfg.profile.name == "debug"
    assert cfg.training.debug is True
    assert cfg.manual_cotrain.global_steps == 10
    assert cfg.manual_cotrain.checkpoint_every == 1
    assert cfg.manual_cotrain.eval_interval_global_steps == 1
    assert cfg.manual_cotrain.real_rollout_target_trajectories == 8
    assert cfg.manual_cotrain.wm_rollout_target_trajectories == 256
    assert cfg.manual_cotrain.real_env_workers == 1
    assert cfg.manual_cotrain.real_envs_per_worker == 1


def test_dreamer_constructor_does_not_mutate_debug_profile() -> None:
    cfg = _compose_experiment("openvla_libero", "profile=debug")
    before = OmegaConf.to_container(cfg, resolve=True)

    DreamerRunner(cfg)

    assert OmegaConf.to_container(cfg, resolve=True) == before


def test_production_profile_preserves_experiment_budget() -> None:
    cfg = _compose_experiment("openvla_libero")

    assert cfg.profile.name == "production"
    assert cfg.training.debug is False
    assert cfg.manual_cotrain.global_steps == 20_000
    assert cfg.manual_cotrain.checkpoint_every == 10
