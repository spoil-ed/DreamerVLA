from __future__ import annotations

from types import SimpleNamespace

import pytest

from dreamervla.scheduler.placement import (
    ComponentPlacement,
    FlexiblePlacementStrategy,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    ResourceMapPlacementStrategy,
    parse_accelerator_range,
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


def test_parse_accelerator_range_expands_ranges_and_commas() -> None:
    assert parse_accelerator_range("0-2,4,6-7") == [0, 1, 2, 4, 6, 7]


def test_parse_accelerator_range_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_accelerator_range("0-2,2")


def test_flexible_placement_maps_explicit_gpu_groups() -> None:
    placements = FlexiblePlacementStrategy([[2], [0, 1]]).get_placement(
        _cluster(num_gpus=4)
    )

    assert [p.rank for p in placements] == [0, 1]
    assert [p.visible_accelerators for p in placements] == [["0", "1"], ["2"]]
    assert [p.device for p in placements] == ["cuda:0", "cuda:2"]
    assert all(p.local_world_size == 2 for p in placements)


def test_flexible_placement_accepts_range_strings() -> None:
    placements = FlexiblePlacementStrategy(["1-2", "3"]).get_placement(
        _cluster(num_gpus=4)
    )

    assert [p.visible_accelerators for p in placements] == [["1", "2"], ["3"]]


def test_flexible_placement_rejects_out_of_range_gpu() -> None:
    with pytest.raises(ValueError, match="GPU"):
        FlexiblePlacementStrategy([[0], [4]]).get_placement(_cluster(num_gpus=4))


def test_resource_map_placement_maps_many_workers_to_one_gpu() -> None:
    placements = ResourceMapPlacementStrategy("2:0-3").get_placement(_cluster(num_gpus=4))

    assert [p.rank for p in placements] == [0, 1, 2, 3]
    assert [p.visible_accelerators for p in placements] == [["2"], ["2"], ["2"], ["2"]]
    assert [p.device for p in placements] == ["cuda:2"] * 4
    assert all(p.local_world_size == 4 for p in placements)


def test_resource_map_placement_supports_resource_and_process_grouping() -> None:
    shared = ResourceMapPlacementStrategy("0-1:0-3").get_placement(_cluster(num_gpus=4))
    packed = ResourceMapPlacementStrategy("0-3:0-1").get_placement(_cluster(num_gpus=4))

    assert [p.visible_accelerators for p in shared] == [["0"], ["0"], ["1"], ["1"]]
    assert [p.visible_accelerators for p in packed] == [["0", "1"], ["2", "3"]]


def test_component_placement_parses_rlinf_style_component_map() -> None:
    cfg = {
        "cluster": {
            "component_placement": {
                "env,rollout": "2:0-3",
                "actor": {"placement": "0-1"},
            }
        }
    }

    placement = ComponentPlacement(cfg)

    env = placement.get_strategy("env").get_placement(_cluster(num_gpus=3))
    rollout = placement.get_strategy("rollout").get_placement(_cluster(num_gpus=3))
    actor = placement.get_strategy("actor").get_placement(_cluster(num_gpus=3))
    assert [p.visible_accelerators for p in env] == [["2"], ["2"], ["2"], ["2"]]
    assert [p.visible_accelerators for p in rollout] == [["2"], ["2"], ["2"], ["2"]]
    assert [p.visible_accelerators for p in actor] == [["0"], ["1"]]


def test_scheduler_package_exports_flexible_placement() -> None:
    from dreamervla import scheduler

    assert scheduler.FlexiblePlacementStrategy is FlexiblePlacementStrategy
