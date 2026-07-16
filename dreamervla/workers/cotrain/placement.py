from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RolePlacement:
    """Placement for one manual-cotrain worker role rank."""

    kind: str
    role: str
    rank: int
    gpu_ids: list[int]

    @property
    def resource_map(self) -> str:
        """Return the compact resource map string used by worker placement configs."""
        if not self.gpu_ids:
            return "node"
        return ",".join(str(gpu) for gpu in self.gpu_ids)


@dataclass(frozen=True)
class ManualCotrainPlacementPlan:
    """Manual-notes topology for env, rollout, actor, and learner roles."""

    ngpu: int
    env_specs: list[RolePlacement]
    rollout_specs: list[RolePlacement]
    actor_specs: list[RolePlacement]
    learner_spec: RolePlacement | None
    actor_fsdp_strategy: str

    @property
    def real_env_ranks(self) -> list[int]:
        """Ranks assigned to the real environment role."""
        return [spec.rank for spec in self.env_specs if spec.role == "real_env"]

    @property
    def wm_env_ranks(self) -> list[int]:
        """Ranks assigned to world-model environment roles."""
        return [spec.rank for spec in self.env_specs if spec.role == "wm_env"]


def build_manual_cotrain_placement(
    ngpu: int,
    *,
    real_env_workers: int = 1,
    include_learner: bool = True,
    component_gpu_groups: Mapping[str, Sequence[Any]] | None = None,
) -> ManualCotrainPlacementPlan:
    """Build the manual-notes placement plan for a local GPU count."""
    count = int(ngpu)
    if count < 0:
        raise ValueError(f"ngpu must be >= 0, got {ngpu!r}")
    real_workers = int(real_env_workers)
    if real_workers < 0:
        raise ValueError(f"real_env_workers must be nonnegative, got {real_env_workers!r}")

    if count == 0:
        if component_gpu_groups:
            raise ValueError("component_gpu_groups require ngpu > 0")
        return ManualCotrainPlacementPlan(
            ngpu=0,
            env_specs=[
                RolePlacement(
                    kind="env",
                    role="real_env" if real_workers else "wm_env",
                    rank=0,
                    gpu_ids=[],
                )
            ],
            rollout_specs=[RolePlacement(kind="rollout", role="rollout", rank=0, gpu_ids=[])],
            actor_specs=[RolePlacement(kind="actor", role="actor", rank=0, gpu_ids=[])],
            learner_spec=(
                RolePlacement(kind="learner", role="learner", rank=0, gpu_ids=[])
                if include_learner
                else None
            ),
            actor_fsdp_strategy="none",
        )

    component_groups = _normalize_component_gpu_groups(component_gpu_groups)
    if component_groups:
        return _build_component_placement_plan(
            count,
            real_workers=real_workers,
            include_learner=bool(include_learner),
            component_groups=component_groups,
        )

    env_specs = [
        RolePlacement(
            kind="env",
            role="real_env",
            rank=rank,
            gpu_ids=[],
        )
        for rank in range(real_workers)
    ]
    wm_gpus = list(range(1, count)) if real_workers else list(range(count))
    env_specs.extend(
        RolePlacement(
            kind="env",
            role="wm_env",
            rank=real_workers + rank,
            gpu_ids=[gpu],
        )
        for rank, gpu in enumerate(wm_gpus)
    )
    rollout_specs = [
        RolePlacement(kind="rollout", role="rollout", rank=gpu, gpu_ids=[gpu])
        for gpu in range(count)
    ]
    # Keep ActorGroup identical between online cotrain and frozen-WM/CLS RL.
    # LearnerGroup is intentionally co-located with actor rank 0; excluding GPU
    # 0 would turn the eight-GPU mainline into a seven-rank FSDP job and break
    # the RLinf global-batch contract.
    actor_gpus = list(range(count))
    actor_specs = [
        RolePlacement(kind="actor", role="actor", rank=rank, gpu_ids=[gpu])
        for rank, gpu in enumerate(actor_gpus)
    ]

    return ManualCotrainPlacementPlan(
        ngpu=count,
        env_specs=env_specs,
        rollout_specs=rollout_specs,
        actor_specs=actor_specs,
        learner_spec=(
            RolePlacement(kind="learner", role="learner", rank=0, gpu_ids=[0])
            if include_learner
            else None
        ),
        actor_fsdp_strategy="fsdp",
    )


def _build_component_placement_plan(
    count: int,
    *,
    real_workers: int,
    include_learner: bool,
    component_groups: Mapping[str, list[list[int]]],
) -> ManualCotrainPlacementPlan:
    env_specs = _component_env_specs(component_groups, real_workers=real_workers)
    rollout_specs = _component_role_specs(
        kind="rollout",
        role="rollout",
        groups=component_groups.get("rollout") or _default_rollout_groups(count),
    )
    if len(rollout_specs) < len(env_specs):
        raise ValueError(
            "manual cotrain rollout component placement must cover every env rank: "
            f"got {len(rollout_specs)} rollout worker(s) for {len(env_specs)} env worker(s)"
        )
    actor_groups = component_groups.get("actor") or _default_actor_groups(
        count,
        include_learner=include_learner,
    )
    actor_specs = _component_role_specs(
        kind="actor",
        role="actor",
        groups=actor_groups,
    )
    learner_spec = None
    if include_learner:
        learner_groups = component_groups.get("learner") or actor_groups[:1]
        if len(learner_groups) != 1:
            raise ValueError(
                "manual cotrain learner component placement must produce exactly one worker"
            )
        learner_spec = RolePlacement(
            kind="learner",
            role="learner",
            rank=0,
            gpu_ids=list(learner_groups[0]),
        )

    return ManualCotrainPlacementPlan(
        ngpu=count,
        env_specs=env_specs,
        rollout_specs=rollout_specs,
        actor_specs=actor_specs,
        learner_spec=learner_spec,
        actor_fsdp_strategy="fsdp",
    )


def _component_env_specs(
    component_groups: Mapping[str, list[list[int]]],
    *,
    real_workers: int,
) -> list[RolePlacement]:
    explicit_real = component_groups.get("real_env")
    explicit_wm = component_groups.get("wm_env")
    if explicit_real is not None or explicit_wm is not None:
        specs = _component_role_specs(
            kind="env",
            role="real_env",
            groups=explicit_real or [],
        )
        offset = len(specs)
        specs.extend(
            RolePlacement(kind="env", role="wm_env", rank=offset + idx, gpu_ids=list(group))
            for idx, group in enumerate(explicit_wm or [])
        )
        if not specs:
            raise ValueError("manual cotrain component placement must include env workers")
        return specs

    env_groups = component_groups.get("env")
    if not env_groups:
        raise ValueError(
            "manual cotrain component placement requires an env, real_env, or wm_env entry"
        )
    real_count = min(real_workers, len(env_groups))
    return [
        RolePlacement(
            kind="env",
            role="real_env" if idx < real_count else "wm_env",
            rank=idx,
            gpu_ids=list(group),
        )
        for idx, group in enumerate(env_groups)
    ]


def _component_role_specs(
    *,
    kind: str,
    role: str,
    groups: Sequence[Sequence[int]],
) -> list[RolePlacement]:
    if not groups:
        raise ValueError(f"manual cotrain component placement for {role} is empty")
    return [
        RolePlacement(kind=kind, role=role, rank=rank, gpu_ids=list(group))
        for rank, group in enumerate(groups)
    ]


def _default_rollout_groups(count: int) -> list[list[int]]:
    return [[gpu] for gpu in range(count)]


def _default_actor_groups(
    count: int,
    *,
    include_learner: bool = True,
) -> list[list[int]]:
    del include_learner
    actor_gpus = list(range(count))
    return [[gpu] for gpu in actor_gpus]


def _normalize_component_gpu_groups(
    component_gpu_groups: Mapping[str, Sequence[Any]] | None,
) -> dict[str, list[list[int]]]:
    if not component_gpu_groups:
        return {}
    normalized: dict[str, list[list[int]]] = {}
    for raw_role, raw_groups in component_gpu_groups.items():
        role = str(raw_role)
        groups = _normalize_role_groups(raw_groups)
        if groups:
            normalized[role] = groups
    return normalized


def _normalize_role_groups(raw_groups: Sequence[Any]) -> list[list[int]]:
    if isinstance(raw_groups, str) or not isinstance(raw_groups, Sequence):
        return [_normalize_gpu_group(raw_groups)]
    values = list(raw_groups)
    if not values:
        return []
    if all(_is_gpu_scalar(value) for value in values):
        return [_normalize_gpu_group(value) for value in values]
    return [_normalize_gpu_group(value) for value in values]


def _normalize_gpu_group(raw_group: Any) -> list[int]:
    if _is_gpu_scalar(raw_group):
        group = [int(raw_group)]
    elif isinstance(raw_group, str):
        group = [int(part.strip()) for part in raw_group.split(",") if part.strip()]
    else:
        group = [int(value) for value in raw_group]
    if not group:
        raise ValueError("manual cotrain component placement GPU group must not be empty")
    if any(gpu < 0 for gpu in group):
        raise ValueError(f"manual cotrain component GPU ids must be >= 0, got {group}")
    if len(group) != len(set(group)):
        raise ValueError(f"manual cotrain component GPU group has duplicates: {group}")
    return group


def _is_gpu_scalar(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
