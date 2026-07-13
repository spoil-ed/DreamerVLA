from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from dreamervla.runtime.rollout_collection_ray import _RayRolloutCollection
from dreamervla.runtime.rollout_collection_vectorized import _VectorizedRolloutCollection


class RolloutCollectionRunner(_RayRolloutCollection):
    """Collect the real LIBERO trajectories used to seed mainline training."""

    runner_name = "rollout_collection"
    runner_status = "current"
    runner_family = "rollout"

    def __init__(self, cfg: dict[str, Any] | DictConfig) -> None:
        config = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
        backend = str(
            OmegaConf.select(config, "collect.backend", default="ray")
        ).strip().lower()
        if backend not in {"ray", "vectorized"}:
            raise ValueError(
                "collect.backend must be one of ['ray', 'vectorized'], "
                f"got {backend!r}"
            )
        self.collection_backend = backend
        super().__init__(config)

    def setup(self) -> None:
        if self.collection_backend == "vectorized":
            _VectorizedRolloutCollection.setup(self)
            return
        super().setup()

    def run(self) -> object:
        if self.collection_backend == "vectorized":
            return _VectorizedRolloutCollection.run(self)
        return super().run()

    def _build_collect_cfg(self) -> dict[str, Any]:
        return _VectorizedRolloutCollection._build_collect_cfg(self)
