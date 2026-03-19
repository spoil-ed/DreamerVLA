"""WMPO-style trainer shell for a minimal Dreamer-VLA implementation.

The public surface mirrors WMPO's trainer organization, but the concrete
training logic is single-process Dreamer-style world-model learning with
imagination-based actor/critic updates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dreamer_vla.config import DreamerVLAConfig
from dreamer_vla.pipeline import DreamerVLAPipeline


class Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """Retained for compatibility with WMPO-style wiring.

    The minimal implementation does not start Ray workers, but preserving the
    object shape makes it easier to replace with real distributed placement later.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, dict[str, Any]] = field(default_factory=dict)

    def create_resource_pool(self) -> None:
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            self.resource_pool_dict[resource_pool_name] = {
                "name": resource_pool_name,
                "process_on_nodes": list(process_on_nodes),
                "driver": "local",
            }

    def get_resource_pool(self, role: Role) -> dict[str, Any]:
        return self.resource_pool_dict[self.mapping[role]]


class SyntheticReplayDataset(Dataset):
    def __init__(
        self,
        num_sequences: int,
        sequence_length: int,
        state_dim: int,
        image_dim: int,
        proprio_dim: int,
        text_dim: int,
        action_dim: int,
        seed: int,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.samples = []

        dynamics = torch.randn(state_dim, state_dim, generator=generator) * 0.15
        control = torch.randn(action_dim, state_dim, generator=generator) * 0.2
        image_proj = torch.randn(state_dim, image_dim, generator=generator)
        proprio_proj = torch.randn(state_dim, proprio_dim, generator=generator)
        text_proj = torch.randn(state_dim, text_dim, generator=generator)

        for _ in range(num_sequences):
            state = torch.randn(state_dim, generator=generator)
            goal = torch.randn(state_dim, generator=generator) * 0.5
            language = goal @ text_proj

            image_steps = []
            proprio_steps = []
            text_steps = []
            action_steps = []
            reward_steps = []
            done_steps = []

            for step in range(sequence_length):
                action = torch.tanh(torch.randn(action_dim, generator=generator))
                reward = -((state - goal) ** 2).mean() - 0.05 * action.pow(2).mean()
                done = float(step == sequence_length - 1)

                image_steps.append(state @ image_proj + 0.05 * torch.randn(image_dim, generator=generator))
                proprio_steps.append(state @ proprio_proj + 0.01 * torch.randn(proprio_dim, generator=generator))
                text_steps.append(language + 0.01 * torch.randn(text_dim, generator=generator))
                action_steps.append(action)
                reward_steps.append(reward)
                done_steps.append(done)

                transition = state @ dynamics + action @ control + 0.2 * goal
                state = torch.tanh(transition + 0.05 * torch.randn(state_dim, generator=generator))

            self.samples.append(
                {
                    "image": torch.stack(image_steps),
                    "proprio": torch.stack(proprio_steps),
                    "text": torch.stack(text_steps),
                    "action": torch.stack(action_steps),
                    "reward": torch.stack(reward_steps),
                    "done": torch.tensor(done_steps, dtype=torch.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class NpzReplayDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        archive = np.load(path)
        required_keys = {"image", "proprio", "text", "action", "reward", "done"}
        missing = required_keys.difference(archive.files)
        if missing:
            raise KeyError(f"Dataset {path} is missing keys: {sorted(missing)}")
        self.data = {key: torch.from_numpy(archive[key]).float() for key in required_keys}

    def __len__(self) -> int:
        return int(self.data["reward"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.data.items()}


class RayTrainer:
    def __init__(
        self,
        config: DreamerVLAConfig,
        tokenizer: Any = None,
        role_worker_mapping: dict[Role, Any] | None = None,
        resource_pool_manager: ResourcePoolManager | None = None,
        ray_worker_group_cls: Any = None,
        reward_fn: Any = None,
        val_reward_fn: Any = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.role_worker_mapping = role_worker_mapping or {}
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.pipeline = DreamerVLAPipeline(config)

        self.resource_pool_to_cls: dict[Any, dict[str, Any]] = {}
        self.wg_dicts: list[Any] = []
        self.train_dataloader: DataLoader | None = None
        self.val_dataloader: DataLoader | None = None
        self.rollout_dataloader: DataLoader | None = None
        self._create_dataloader()

    def _build_dataset(self, split: str) -> Dataset:
        data_cfg = self.config.data
        model_cfg = self.config.model

        if data_cfg.dataset_path:
            return NpzReplayDataset(data_cfg.dataset_path)

        counts = {
            "train": data_cfg.train_num_sequences,
            "val": data_cfg.val_num_sequences,
            "rollout": data_cfg.rollout_num_sequences,
        }
        split_seed = {
            "train": data_cfg.seed,
            "val": data_cfg.seed + 1,
            "rollout": data_cfg.seed + 2,
        }
        return SyntheticReplayDataset(
            num_sequences=counts[split],
            sequence_length=data_cfg.sequence_length,
            state_dim=data_cfg.synthetic_state_dim,
            image_dim=model_cfg.image_dim,
            proprio_dim=model_cfg.proprio_dim,
            text_dim=model_cfg.text_dim,
            action_dim=model_cfg.action_dim,
            seed=split_seed[split],
        )

    def _create_dataloader(self) -> None:
        self.train_dataloader = DataLoader(
            self._build_dataset("train"),
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
        )
        self.val_dataloader = DataLoader(
            self._build_dataset("val"),
            batch_size=self.config.data.val_batch_size,
            shuffle=False,
        )
        self.rollout_dataloader = DataLoader(
            self._build_dataset("rollout"),
            batch_size=self.config.data.rollout_batch_size,
            shuffle=False,
        )

    def _save_rollouts(self, global_steps: int = 0, rollout_epoch: int = 1, use_wm: bool = True) -> dict[str, float]:
        del global_steps, rollout_epoch, use_wm
        metrics = defaultdict(list)
        for batch in self.rollout_dataloader:
            batch_metrics = self.pipeline.evaluate_batch(batch)
            metrics["rollout/reward_mean"].append(batch_metrics["eval/imagination_reward_mean"])
            metrics["rollout/return_mean"].append(batch_metrics["eval/imagination_return_mean"])
        return {key: float(np.mean(values)) for key, values in metrics.items()}

    def _validate(self, global_steps: int = 0) -> dict[str, float]:
        del global_steps
        metrics = defaultdict(list)
        for batch in self.val_dataloader:
            batch_metrics = self.pipeline.evaluate_batch(batch)
            for key, value in batch_metrics.items():
                metrics[key].append(value)
        return {key: float(np.mean(values)) for key, values in metrics.items()}

    def init_workers(self) -> None:
        if self.resource_pool_manager is None:
            self.resource_pool_manager = ResourcePoolManager(
                resource_pool_spec={"local": [0]},
                mapping={Role.ActorRollout: "local", Role.Critic: "local"},
            )
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool_name: {"status": "local-ready"} for pool_name in self.resource_pool_manager.resource_pool_dict
        }

    def fit(self) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        self.init_workers()

        for epoch in range(self.config.trainer.total_epochs):
            train_metrics = defaultdict(list)
            for batch in self.train_dataloader:
                step_metrics = self.pipeline.training_step(batch)
                for key, value in step_metrics.items():
                    train_metrics[key].append(value)

            epoch_metrics = {key: float(np.mean(values)) for key, values in train_metrics.items()}
            epoch_metrics["epoch"] = float(epoch)

            if (epoch + 1) % self.config.trainer.validate_every == 0:
                val_metrics = self._validate(global_steps=epoch + 1)
                epoch_metrics.update({f"val/{key}": value for key, value in val_metrics.items()})

            history.append(epoch_metrics)
            if self.config.trainer.log_every and (epoch + 1) % self.config.trainer.log_every == 0:
                printable = ", ".join(f"{key}={value:.4f}" for key, value in sorted(epoch_metrics.items()) if key != "epoch")
                print(f"[epoch {epoch}] {printable}")

        return history

    def filter_format(self, reward_tensor: Any, batch: Any, n_samples: int) -> Any:
        del reward_tensor, n_samples
        return batch

    def filter(self, reward_tensor: Any, batch: Any, n_samples: int) -> Any:
        del reward_tensor, n_samples
        return batch

    def add_to_buffer(self, batch: Any, batch_size: int, n_samples: int) -> Any:
        del batch_size, n_samples
        return batch
