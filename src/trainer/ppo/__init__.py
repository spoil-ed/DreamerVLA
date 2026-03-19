"""PPO training scaffolds for Dreamer-VLA."""

from .ray_trainer import RayTrainer, ResourcePoolManager, Role

__all__ = ["RayTrainer", "ResourcePoolManager", "Role"]
