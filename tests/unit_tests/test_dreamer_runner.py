from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.runners import CotrainRunner, DreamerRunner


def _compose_experiment(name: str):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=[f"experiment={name}"])
    OmegaConf.resolve(cfg)
    return cfg


def test_dreamer_runner_preserves_original_cotrain_runner() -> None:
    assert CotrainRunner.__module__ == "dreamervla.runners.cotrain_runner"
    assert DreamerRunner.__module__ == "dreamervla.runners.dreamer_runner"
    assert DreamerRunner is not CotrainRunner
    assert issubclass(DreamerRunner, CotrainRunner)
    assert DreamerRunner._restore_manual_resume_state is CotrainRunner._restore_manual_resume_state
    assert (
        DreamerRunner._maybe_save_manual_checkpoint is CotrainRunner._maybe_save_manual_checkpoint
    )


def test_dreamer_runner_places_frozen_learner_on_cpu() -> None:
    cfg = _compose_experiment("openvla_libero")
    runner = DreamerRunner(cfg)
    plan = runner._placement_plan()

    assert [spec.gpu_ids for spec in plan.actor_specs] == [[gpu] for gpu in range(8)]
    assert plan.learner_spec is not None
    assert plan.learner_spec.gpu_ids == []


def test_dreamer_runner_uses_25_one_slot_osmesa_real_workers() -> None:
    cfg = _compose_experiment("openvla_libero")
    runner = DreamerRunner(cfg)
    plan = runner._placement_plan()

    assert cfg.render_backend == "osmesa"
    assert runner._real_env_workers() == 25
    assert runner._real_envs_per_worker() == 1
    assert runner._real_env_cfg()["num_envs_per_worker"] == 1
    assert runner._real_rollout_epochs_by_worker(25) == [2] * 7 + [1] * 18
    assert runner._real_rollout_total_chunks() == 1216
    assert len(plan.real_env_ranks) == 25
    assert len(plan.wm_env_ranks) == 7


def test_cotrain_and_dreamer_are_separate_public_routes() -> None:
    cotrain_cfg = _compose_experiment("openvla_onetraj_libero_cotrain")
    dreamer_cfg = _compose_experiment("openvla_libero")

    assert cotrain_cfg._target_ == "dreamervla.runners.CotrainRunner"
    assert dreamer_cfg._target_ == "dreamervla.runners.DreamerRunner"
    assert CotrainRunner.runner_family == "cotrain"
    assert DreamerRunner.runner_family == "dreamer"
