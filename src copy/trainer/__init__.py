"""Training scaffolds for Dreamer-VLA."""

from .main_ppo import DreamerRewardManager, build_trainer

__all__ = ["DreamerRewardManager", "build_trainer"]
