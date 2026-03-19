"""WMPO-style entrypoint for the minimal Dreamer-VLA trainer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch

from dreamer_vla.config import DreamerVLAConfig, load_config
from dreamer_vla.trainer.ppo.ray_trainer import RayTrainer


@dataclass
class DreamerRewardManager:
    """Compatibility wrapper kept close to WMPO's entrypoint contract."""

    num_examine: int = 0
    config: Any | None = None

    def verify(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float], dict[str, float], dict[str, float]]:
        reward = data["reward"]
        reward_metrics = {"reward_mean": reward.mean().item(), "reward_std": reward.std().item()}
        return reward, reward_metrics, {}, reward_metrics

    def __call__(self, data: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        reward = data["reward"]
        return {"all": reward}, {"reward_mean": reward.mean().item()}


def build_trainer(
    config: DreamerVLAConfig | None = None,
    tokenizer: Any | None = None,
    role_worker_mapping: dict[Any, Any] | None = None,
    resource_pool_manager: Any | None = None,
    ray_worker_group_cls: Any | None = None,
    reward_fn: Any | None = None,
    val_reward_fn: Any | None = None,
) -> RayTrainer:
    resolved_config = config or load_config()
    if reward_fn is None:
        reward_fn = DreamerRewardManager(num_examine=0, config=resolved_config)
    if val_reward_fn is None:
        val_reward_fn = DreamerRewardManager(num_examine=1, config=resolved_config)
    return RayTrainer(
        config=resolved_config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping or {},
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the minimal Dreamer-VLA stack.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config override.")
    args = parser.parse_args(argv)

    trainer = build_trainer(config=load_config(args.config))
    history = trainer.fit()
    if history:
        print("Final epoch metrics:")
        for key, value in sorted(history[-1].items()):
            if key == "epoch":
                continue
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
