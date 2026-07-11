from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def _tiny_fixture_with(*group_files: str):
    root = Path(__file__).resolve().parents[2]
    configs = [
        OmegaConf.load(root / "tests" / "fixtures" / "manual_cotrain_ray_tiny.yaml")
    ]
    configs.extend(OmegaConf.load(root / "configs" / path) for path in group_files)
    return OmegaConf.merge(*configs)


def test_manual_precision_and_parallelism_groups_compose_for_ray_backend() -> None:
    cfg = _tiny_fixture_with("precision/bf16.yaml", "parallelism/fsdp.yaml")

    assert cfg.learner.train_cfg.precision == "bf16"
    assert cfg.learner.num_workers == 1
    assert cfg.learner.placement.strategy == "packed"
    assert cfg.learner.placement.end_gpu == 0
    assert cfg.learner.train_cfg.device == "auto"
    assert cfg.learner.train_cfg.fsdp.strategy == "fsdp"
    assert cfg.learner.train_cfg.fsdp.activation_checkpointing is True


def test_scheduler_group_and_ray_scripts_are_manual_ops_entrypoints() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = _tiny_fixture_with("scheduler/local.yaml")

    assert cfg.scheduler.cluster.num_nodes == 1
    assert cfg.scheduler.component_placement.learner.strategy == "node"
    assert "conda activate dreamervla" in (root / "scripts/start_ray.sh").read_text(
        encoding="utf-8"
    )
    assert "ray status" in (root / "scripts/check_ray.sh").read_text(
        encoding="utf-8"
    )
