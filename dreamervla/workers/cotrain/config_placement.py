"""Hydra adapter for the pure manual-cotrain placement model."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf

from dreamervla.scheduler.placement import ComponentPlacement
from dreamervla.workers.cotrain.placement import (
    ManualCotrainPlacementPlan,
    build_manual_cotrain_placement,
)


def build_manual_cotrain_placement_from_config(
    cfg: Any,
) -> ManualCotrainPlacementPlan:
    """Resolve the manual cotrain topology declared by a Hydra config."""
    ngpu = int(OmegaConf.select(cfg, "manual_cotrain.ngpu", default=1))
    real_env_workers = int(OmegaConf.select(cfg, "manual_cotrain.real_env_workers", default=1))
    component_gpu_groups = _component_gpu_groups_from_config(cfg, ngpu=ngpu)
    return build_manual_cotrain_placement(
        ngpu,
        real_env_workers=real_env_workers,
        include_learner=True,
        component_gpu_groups=component_gpu_groups,
    )


def _component_gpu_groups_from_config(
    cfg: Any,
    *,
    ngpu: int,
) -> dict[str, list[list[int]]] | None:
    component_cfg = OmegaConf.select(
        cfg,
        "cluster.component_placement",
        default=None,
    )
    if component_cfg is None:
        return None

    placement = ComponentPlacement(cfg)
    cluster = SimpleNamespace(num_gpus=ngpu)
    groups: dict[str, list[list[int]]] = {}
    for component in ("env", "real_env", "wm_env", "rollout", "actor", "learner"):
        if not placement.has_component(component):
            continue
        resolved = placement.get_strategy(component).get_placement(cluster)
        groups[component] = [[int(gpu) for gpu in item.visible_accelerators] for item in resolved]
    return groups or None


__all__ = ["build_manual_cotrain_placement_from_config"]
