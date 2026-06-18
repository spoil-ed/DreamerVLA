from __future__ import annotations

from pathlib import Path

import pytest


def test_scheduler_package_exports_public_primitives() -> None:
    from dreamervla.scheduler import (
        Channel,
        Cluster,
        NodePlacementStrategy,
        PackedPlacementStrategy,
        Placement,
        Worker,
        WorkerGroup,
    )

    assert Cluster.__name__ == "Cluster"
    assert Worker.__name__ == "Worker"
    assert WorkerGroup.__name__ == "WorkerGroup"
    assert Channel.__name__ == "Channel"
    assert Placement.__name__ == "Placement"
    assert PackedPlacementStrategy.__name__ == "PackedPlacementStrategy"
    assert NodePlacementStrategy.__name__ == "NodePlacementStrategy"


def test_ray_dependency_is_declared_for_fresh_installs() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")

    assert "[project.optional-dependencies]" in pyproject
    assert "ray = [" in pyproject
    assert '"ray[default]>=2.47.0"' in pyproject
    assert "ray[default]" not in requirements


def test_packed_placement_rejects_non_positive_gpus_per_worker() -> None:
    from dreamervla.scheduler.placement import PackedPlacementStrategy

    with pytest.raises(ValueError, match="num_gpus_per_worker"):
        PackedPlacementStrategy(0, 1, num_gpus_per_worker=0)
