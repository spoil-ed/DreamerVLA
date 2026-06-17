from __future__ import annotations

from types import SimpleNamespace

import pytest

from dreamervla.scheduler.placement import (
    NodePlacementStrategy,
    PackedPlacementStrategy,
)


def _cluster(num_gpus: int) -> SimpleNamespace:
    """Minimal stand-in exposing only what get_placement reads."""
    return SimpleNamespace(num_gpus=num_gpus)


def test_packed_one_gpu_per_worker_maps_each_rank_to_one_gpu() -> None:
    placements = PackedPlacementStrategy(0, 3).get_placement(_cluster(num_gpus=4))

    assert [p.rank for p in placements] == [0, 1, 2, 3]
    assert [p.device for p in placements] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert [p.visible_accelerators for p in placements] == [["0"], ["1"], ["2"], ["3"]]
    assert all(p.local_world_size == 4 for p in placements)


def test_packed_two_gpus_per_worker_packs_contiguous_gpus() -> None:
    placements = PackedPlacementStrategy(0, 3, num_gpus_per_worker=2).get_placement(
        _cluster(num_gpus=4)
    )

    assert len(placements) == 2
    assert placements[0].visible_accelerators == ["0", "1"]
    assert placements[0].device == "cuda:0"
    assert placements[1].visible_accelerators == ["2", "3"]
    assert placements[1].device == "cuda:2"


def test_packed_raises_when_cluster_has_too_few_gpus() -> None:
    with pytest.raises(ValueError, match="GPU"):
        PackedPlacementStrategy(0, 3).get_placement(_cluster(num_gpus=2))


def test_node_placement_yields_cpu_only_ranks() -> None:
    placements = NodePlacementStrategy(3).get_placement(_cluster(num_gpus=0))

    assert [p.rank for p in placements] == [0, 1, 2]
    assert all(p.device == "cpu" for p in placements)
    assert all(p.visible_accelerators == [] for p in placements)
    assert all(p.local_world_size == 3 for p in placements)
