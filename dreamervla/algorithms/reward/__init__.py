"""Swappable WMPO reward definitions (protocol + registry)."""

# Import implementations for their registration side effect.
from dreamervla.algorithms.reward import probability_outcome as _probability_outcome  # noqa: F401
from dreamervla.algorithms.reward import sparse_outcome as _sparse_outcome  # noqa: F401
from dreamervla.algorithms.reward.protocol import RewardModel
from dreamervla.algorithms.reward.registry import (
    get_reward_model,
    register_reward_model,
    reward_model_names,
)

__all__ = [
    "RewardModel",
    "get_reward_model",
    "register_reward_model",
    "reward_model_names",
]
