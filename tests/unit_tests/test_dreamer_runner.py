from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.runners import CotrainRunner, DreamerRunner


def test_dreamer_runner_preserves_original_cotrain_runner() -> None:
    assert CotrainRunner.__module__ == "dreamervla.runners.cotrain_runner"
    assert DreamerRunner.__module__ == "dreamervla.runners.dreamer_runner"
    assert DreamerRunner is not CotrainRunner


def test_dreamer_runner_places_frozen_learner_on_cpu() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=openvla_libero"])
    OmegaConf.resolve(cfg)

    runner = DreamerRunner(cfg)
    plan = runner._placement_plan()

    assert [spec.gpu_ids for spec in plan.actor_specs] == [
        [gpu] for gpu in range(8)
    ]
    assert plan.learner_spec is not None
    assert plan.learner_spec.gpu_ids == []
