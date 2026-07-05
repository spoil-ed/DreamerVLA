from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_manual_precision_and_parallelism_groups_compose_for_ray_backend() -> None:
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=manual_cotrain_ray_tiny",
                "+precision=bf16",
                "+parallelism=fsdp",
            ],
        )

    assert cfg.learner.train_cfg.precision == "bf16"
    assert cfg.learner.num_workers == 1
    assert cfg.learner.placement.strategy == "packed"
    assert cfg.learner.placement.end_gpu == 0
    assert cfg.learner.train_cfg.device == "auto"
    assert cfg.learner.train_cfg.fsdp.strategy == "fsdp"
    assert cfg.learner.train_cfg.fsdp.activation_checkpointing is True


def test_scheduler_group_and_ray_scripts_are_manual_ops_entrypoints() -> None:
    root = Path(__file__).resolve().parents[2]
    config_dir = str(root / "configs")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=manual_cotrain_ray_tiny",
                "+scheduler=local",
            ],
        )

    assert cfg.scheduler.cluster.num_nodes == 1
    assert cfg.scheduler.component_placement.learner.strategy == "node"
    assert "conda activate dreamervla" in (root / "scripts/start_ray.sh").read_text(
        encoding="utf-8"
    )
    assert "ray status" in (root / "scripts/check_ray.sh").read_text(
        encoding="utf-8"
    )
