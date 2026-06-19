from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_manual_precision_and_parallelism_groups_compose_for_ray_backend() -> None:
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=online_cotrain_ray_synthetic",
                "+precision=bf16",
                "+parallelism=fsdp",
            ],
        )

    assert cfg.learner.train_cfg.precision == "bf16"
    assert cfg.learner.train_cfg.fsdp.strategy == "fsdp"
    assert cfg.learner.train_cfg.fsdp.activation_checkpointing is True
