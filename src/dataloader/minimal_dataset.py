from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.dataloader.base_dataset import BaseDataset


def _pad_action_batch(actions: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_steps = max((int(action.shape[0]) for action in actions), default=0)
    action_dim = max((int(action.shape[-1]) for action in actions if action.ndim == 2), default=0)
    padded = torch.zeros(len(actions), max_steps, action_dim, dtype=torch.float32)
    mask = torch.zeros(len(actions), max_steps, dtype=torch.bool)
    for idx, action in enumerate(actions):
        if action.numel() == 0:
            continue
        steps = int(action.shape[0])
        padded[idx, :steps] = action
        mask[idx, :steps] = True
    return padded, mask


def _pad_state_batch(states: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_dim = max((int(state.numel()) for state in states), default=0)
    padded = torch.zeros(len(states), max_dim, dtype=torch.float32)
    mask = torch.zeros(len(states), max_dim, dtype=torch.bool)
    for idx, state in enumerate(states):
        if state.numel() == 0:
            continue
        dim = int(state.numel())
        padded[idx, :dim] = state.reshape(-1)
        mask[idx, :dim] = True
    return padded, mask


def _stack_long(values: list[int]) -> torch.Tensor:
    if not values:
        return torch.zeros(0, dtype=torch.long)
    return torch.tensor(values, dtype=torch.long)


def _make_image(seed: int, size: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(image, mode="RGB")


@dataclass(frozen=True)
class MinimalDataSpec:
    num_samples: int
    image_size: int
    action_dim: int
    state_dim: int
    action_horizon: int
    with_wrist: bool
    with_world_model: bool


class MinimalRynnVLADataset(BaseDataset):
    def __init__(
        self,
        num_samples: int = 8,
        image_size: int = 256,
        action_dim: int = 7,
        state_dim: int = 8,
        action_horizon: int = 5,
        with_wrist: bool = True,
        with_world_model: bool = True,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.action_horizon = int(action_horizon)
        self.with_wrist = bool(with_wrist)
        self.with_world_model = bool(with_world_model)
        self.seed = int(seed)
        self._data_spec = MinimalDataSpec(
            num_samples=self.num_samples,
            image_size=self.image_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            with_wrist=self.with_wrist,
            with_world_model=self.with_world_model,
        )

    @property
    def data_spec(self) -> MinimalDataSpec:
        return self._data_spec

    def get_normalizer(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "action": {
                "mean": torch.zeros(self.action_dim, dtype=torch.float32),
                "std": torch.ones(self.action_dim, dtype=torch.float32),
            },
            "state": {
                "mean": torch.zeros(self.state_dim, dtype=torch.float32),
                "std": torch.ones(self.state_dim, dtype=torch.float32),
            },
        }

    def __len__(self) -> int:
        return self.num_samples

    def _build_action_sample(self, idx: int) -> dict[str, Any]:
        prompt = f"move the robot to synthetic task {idx}"
        images = [_make_image(self.seed + idx * 11, self.image_size)]
        if self.with_wrist:
            images.append(_make_image(self.seed + idx * 11 + 1, self.image_size))
        action = torch.linspace(-0.5, 0.5, steps=self.action_horizon * self.action_dim, dtype=torch.float32).reshape(
            self.action_horizon, self.action_dim
        )
        action = action + idx * 0.01
        state = torch.linspace(-1.0, 1.0, steps=self.state_dim, dtype=torch.float32) + idx * 0.01
        conversations = [
            {
                "from": "human",
                "value": f"What action should the robot take to {prompt}?" + "<|state|>" + "<|image|>" * len(images),
            },
            {
                "from": "gpt",
                "value": "<|action|>" * self.action_horizon,
            },
        ]
        return {
            "obs": {
                "conversations": conversations,
                "images": images,
                "prompt_text": prompt,
                "task_type": "action",
                "task_id": idx,
                "state": state,
            },
            "next_obs": {
                "images": [],
                "prompt_text": prompt,
                "task_type": "action",
            },
            "action": action,
            "reward": torch.zeros(1, dtype=torch.float32),
            "meta": {
                "sample_id": idx,
                "task_name": prompt,
                "task_id": idx,
                "task_type": "action",
            },
        }

    def _build_world_sample(self, idx: int) -> dict[str, Any]:
        prompt = f"predict synthetic next image {idx}"
        history_images = [_make_image(self.seed + idx * 23, self.image_size)]
        future_images = [_make_image(self.seed + idx * 23 + 1, self.image_size)]
        if self.with_wrist:
            history_images.append(_make_image(self.seed + idx * 23 + 2, self.image_size))
            future_images.append(_make_image(self.seed + idx * 23 + 3, self.image_size))
            human_value = "Generate the next image based on the provided sequence of historical images and corresponding actions." + "<|image|><|image|><|action|>"
            assistant_value = "<|image|><|image|>"
        else:
            human_value = "Generate the next image based on the provided sequence of historical images and corresponding actions." + "<|image|><|action|>"
            assistant_value = "<|image|>"
        action = torch.linspace(-0.25, 0.25, steps=self.action_dim, dtype=torch.float32).reshape(1, self.action_dim)
        conversations = [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": assistant_value},
        ]
        return {
            "obs": {
                "conversations": conversations,
                "images": history_images,
                "prompt_text": prompt,
                "task_type": "world",
                "task_id": idx,
                "state": torch.zeros(0, dtype=torch.float32),
            },
            "next_obs": {
                "images": future_images,
                "prompt_text": prompt,
                "task_type": "world",
            },
            "action": action,
            "reward": torch.zeros(1, dtype=torch.float32),
            "meta": {
                "sample_id": idx,
                "task_name": prompt,
                "task_id": idx,
                "task_type": "world",
            },
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.with_world_model and idx % 2 == 1:
            item = self._build_world_sample(idx)
        else:
            item = self._build_action_sample(idx)

        obs = item["obs"]
        next_obs = item["next_obs"]
        action = item["action"]
        state = obs["state"]

        obs["encoder_inputs"] = {
            "prompt_text": [obs["prompt_text"]],
            "conversations": [obs["conversations"]],
            "images": [obs["images"]],
            "state": state.unsqueeze(0),
            "state_mask": torch.ones(1, state.numel(), dtype=torch.bool) if state.numel() else torch.zeros(1, 0, dtype=torch.bool),
            "action": action.unsqueeze(0),
            "action_mask": torch.ones(1, action.shape[0], dtype=torch.bool),
            "task_type": [obs["task_type"]],
            "task_id": torch.tensor([int(obs["task_id"])], dtype=torch.long),
            "meta": [item["meta"]],
        }
        next_obs["encoder_inputs"] = {
            "prompt_text": [next_obs["prompt_text"]],
            "conversations": [[]],
            "images": [next_obs["images"]],
            "state": None,
            "state_mask": None,
            "action": None,
            "action_mask": None,
            "task_type": [next_obs["task_type"]],
            "task_id": torch.tensor([int(item["meta"]["task_id"])], dtype=torch.long),
            "meta": [item["meta"]],
        }
        return item

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        obs_list = [item["obs"] for item in batch]
        next_obs_list = [item["next_obs"] for item in batch]
        actions = [item["action"] for item in batch]
        rewards = torch.stack([item["reward"] for item in batch], dim=0)
        states = [obs["state"] for obs in obs_list]

        padded_actions, action_mask = _pad_action_batch(actions)
        padded_states, state_mask = _pad_state_batch(states)

        collated_obs = {
            "conversations": [obs["conversations"] for obs in obs_list],
            "images": [obs["images"] for obs in obs_list],
            "prompt_text": [obs["prompt_text"] for obs in obs_list],
            "task_type": [obs["task_type"] for obs in obs_list],
            "task_id": _stack_long([int(obs["task_id"]) for obs in obs_list]),
            "state": padded_states,
            "state_mask": state_mask,
            "raw_state": states,
        }
        collated_next_obs = {
            "images": [obs["images"] for obs in next_obs_list],
            "prompt_text": [obs["prompt_text"] for obs in next_obs_list],
            "task_type": [obs["task_type"] for obs in next_obs_list],
        }
        collated_obs["encoder_inputs"] = {
            "prompt_text": collated_obs["prompt_text"],
            "conversations": collated_obs["conversations"],
            "images": collated_obs["images"],
            "state": padded_states,
            "state_mask": state_mask,
            "action": padded_actions,
            "action_mask": action_mask,
            "task_type": collated_obs["task_type"],
            "task_id": collated_obs["task_id"],
            "meta": [item["meta"] for item in batch],
        }
        collated_next_obs["encoder_inputs"] = {
            "prompt_text": collated_next_obs["prompt_text"],
            "conversations": [[] for _ in next_obs_list],
            "images": collated_next_obs["images"],
            "state": None,
            "state_mask": None,
            "action": None,
            "action_mask": None,
            "task_type": collated_next_obs["task_type"],
            "task_id": _stack_long([int(item["meta"]["task_id"]) for item in batch]),
            "meta": [item["meta"] for item in batch],
        }
        return {
            "obs": collated_obs,
            "next_obs": collated_next_obs,
            "action": padded_actions,
            "action_mask": action_mask,
            "raw_action": actions,
            "reward": rewards,
            "meta": [item["meta"] for item in batch],
        }


__all__ = ["MinimalDataSpec", "MinimalRynnVLADataset"]
