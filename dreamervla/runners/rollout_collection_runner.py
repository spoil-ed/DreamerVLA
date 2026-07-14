from __future__ import annotations

from dreamervla.runtime.rollout_collection_ray import _RayRolloutCollection


class RolloutCollectionRunner(_RayRolloutCollection):
    """Collect the real LIBERO trajectories used to seed mainline training."""

    runner_name = "rollout_collection"
    runner_status = "current"
    runner_family = "rollout"
