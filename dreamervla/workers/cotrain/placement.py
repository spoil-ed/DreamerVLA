from __future__ import annotations

from dataclasses import dataclass


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
    learner_spec: RolePlacement
    actor_fsdp_strategy: str

    @property
    def real_env_ranks(self) -> list[int]:
        """Ranks assigned to the real environment role."""
        return [spec.rank for spec in self.env_specs if spec.role == "real_env"]

    @property
    def wm_env_ranks(self) -> list[int]:
        """Ranks assigned to world-model environment roles."""
        return [spec.rank for spec in self.env_specs if spec.role == "wm_env"]


def build_manual_cotrain_placement(ngpu: int) -> ManualCotrainPlacementPlan:
    """Build the manual-notes placement plan for a local GPU count."""
    count = int(ngpu)
    if count < 0:
        raise ValueError(f"ngpu must be >= 0, got {ngpu!r}")

    if count == 0:
        return ManualCotrainPlacementPlan(
            ngpu=0,
            env_specs=[RolePlacement(kind="env", role="real_env", rank=0, gpu_ids=[])],
            rollout_specs=[RolePlacement(kind="rollout", role="rollout", rank=0, gpu_ids=[])],
            actor_specs=[RolePlacement(kind="actor", role="actor", rank=0, gpu_ids=[])],
            learner_spec=RolePlacement(kind="learner", role="learner", rank=0, gpu_ids=[]),
            actor_fsdp_strategy="none",
        )

    env_specs = [
        RolePlacement(
            kind="env",
            role="real_env" if gpu == 0 else "wm_env",
            rank=gpu,
            gpu_ids=[gpu],
        )
        for gpu in range(count)
    ]
    rollout_specs = [
        RolePlacement(kind="rollout", role="rollout", rank=gpu, gpu_ids=[gpu])
        for gpu in range(count)
    ]
    actor_gpus = list(range(1, count)) or [0]
    actor_specs = [
        RolePlacement(kind="actor", role="actor", rank=rank, gpu_ids=[gpu])
        for rank, gpu in enumerate(actor_gpus)
    ]

    return ManualCotrainPlacementPlan(
        ngpu=count,
        env_specs=env_specs,
        rollout_specs=rollout_specs,
        actor_specs=actor_specs,
        learner_spec=RolePlacement(kind="learner", role="learner", rank=0, gpu_ids=[0]),
        actor_fsdp_strategy="fsdp",
    )
