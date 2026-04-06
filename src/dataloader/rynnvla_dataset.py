from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from PIL import Image

from src.dataloader.base_dataset import BaseDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / path).resolve()


def _discover_task_files(raw_data_dir: Path, task_suite_name: str) -> list[Path]:
    """Accept either a suite directory, a LIBERO datasets root, or a flat directory."""
    direct_files = sorted(raw_data_dir.glob("*_demo.hdf5"))
    if direct_files:
        return direct_files

    suite_dir = raw_data_dir / task_suite_name
    if suite_dir.is_dir():
        suite_files = sorted(suite_dir.glob("*_demo.hdf5"))
        if suite_files:
            return suite_files

    nested_files = sorted(raw_data_dir.glob("libero_*/*_demo.hdf5"))
    if nested_files:
        if task_suite_name == "all":
            return nested_files
        filtered = [path for path in nested_files if path.parent.name == task_suite_name]
        if filtered:
            return filtered

    return []


def _extract_demo_index(demo_key: str) -> int:
    match = re.fullmatch(r"demo_(\d+)", demo_key)
    if match is None:
        return -1
    return int(match.group(1))


def _list_demo_keys(data_group: h5py.Group) -> list[str]:
    demo_keys = [key for key in data_group.keys() if _extract_demo_index(key) >= 0]
    return sorted(demo_keys, key=_extract_demo_index)


def _image_from_array(rgb_frame: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb_frame[::-1, ::-1].astype(np.uint8))


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


@dataclass(frozen=True)
class RynnVLADataSpec:
    config_path: str
    raw_data_dir: str
    split: str
    task_suite_name: str
    num_tasks: int
    num_samples: int
    resolution: int
    with_state: bool
    with_wrist: bool
    with_action: bool
    with_world_model: bool
    action_dim: int
    state_dim: int
    action_horizon: int
    action_history: int
    world_history: int


class RynnVLALIBERODataset(BaseDataset):
    """RynnVLA-style LIBERO dataset that mirrors the sample building logic in RynnVLA-002."""

    def __init__(
        self,
        config_path: str | Path,
        resolution: int = 256,
        with_state: bool = True,
        with_wrist: bool = True,
        with_action: bool = True,
        with_world_model: bool = True,
        split_override: str | None = None,
    ) -> None:
        super().__init__()
        self.config_path = _resolve_path(config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.load(handle, Loader=yaml.FullLoader)

        self.resolution = int(resolution)
        self.with_state = bool(with_state)
        self.with_wrist = bool(with_wrist)
        self.with_action = bool(with_action)
        self.with_world_model = bool(with_world_model)

        configured_split = str(self.config["META"].get("split", "all"))
        self.split = str(split_override) if split_override is not None else configured_split
        self.task_suite_name = str(self.config["META"].get("libero_task_suite", "unknown"))
        self.raw_data_dir = _resolve_path(self.config["META"]["raw_data_dir"], base_dir=self.config_path.parent)
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"RynnVLA raw_data_dir does not exist: {self.raw_data_dir}")

        self.action_horizon = int(self.config["action_model"]["len_action"])
        self.action_history = int(self.config["action_model"]["his"])
        self.world_history = int(self.config["world_model"]["his"])

        self.task_files = _discover_task_files(self.raw_data_dir, self.task_suite_name)
        if not self.task_files:
            raise RuntimeError(
                f"No HDF5 demo files found for suite '{self.task_suite_name}' under {self.raw_data_dir}"
            )

        self.task_names = [path.stem[:-5] if path.stem.endswith("_demo") else path.stem for path in self.task_files]
        self.task_name_to_id = {task_name: idx for idx, task_name in enumerate(self.task_names)}
        self.num_tasks = len(self.task_files)
        self.split_training_set = self.split != "all"

        self.samples: list[dict[str, Any]] = []
        self.action_dim = 0
        self.state_dim = 0
        self._build_index()

        self._data_spec = RynnVLADataSpec(
            config_path=str(self.config_path),
            raw_data_dir=str(self.raw_data_dir),
            split=self.split,
            task_suite_name=self.task_suite_name,
            num_tasks=self.num_tasks,
            num_samples=len(self.samples),
            resolution=self.resolution,
            with_state=self.with_state,
            with_wrist=self.with_wrist,
            with_action=self.with_action,
            with_world_model=self.with_world_model,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_history=self.action_history,
            world_history=self.world_history,
        )

    @property
    def data_spec(self) -> RynnVLADataSpec:
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
        return len(self.samples)

    def _build_index(self) -> None:
        split_index_ood = math.ceil(self.num_tasks * 0.9)
        for task_order_idx, task_path in enumerate(self.task_files):
            task_name = self.task_names[task_order_idx]
            task_id = self.task_name_to_id[task_name]
            with h5py.File(task_path, "r") as handle:
                data_group = handle["data"]
                demo_keys = _list_demo_keys(data_group)
                if not demo_keys:
                    continue
                first_demo = data_group[demo_keys[0]]
                if self.action_dim == 0:
                    self.action_dim = int(first_demo["actions"].shape[-1])
                    ee_dim = int(np.prod(first_demo["obs"]["ee_states"].shape[1:]))
                    gripper_dim = int(np.prod(first_demo["obs"]["gripper_states"].shape[1:]))
                    self.state_dim = ee_dim + gripper_dim

                split_index_ind = math.ceil(len(demo_keys) * 0.9)
                for demo_offset, demo_key in enumerate(demo_keys):
                    current_split = self._resolve_split(task_order_idx, demo_offset, split_index_ood, split_index_ind)
                    if self.split_training_set and current_split != self.split:
                        continue

                    episode = data_group[demo_key]
                    action_count = int(episode["actions"].shape[0])
                    for action_idx in range(action_count):
                        if self.with_action:
                            sample = self._make_action_sample(
                                task_path=task_path,
                                task_name=task_name,
                                task_id=task_id,
                                demo_key=demo_key,
                                action_idx=action_idx,
                                action_count=action_count,
                            )
                            if sample is not None:
                                self.samples.append(sample)
                        if self.with_world_model:
                            sample = self._make_world_sample(
                                task_path=task_path,
                                task_name=task_name,
                                task_id=task_id,
                                demo_key=demo_key,
                                action_idx=action_idx,
                                action_count=action_count,
                            )
                            if sample is not None:
                                self.samples.append(sample)

    def _resolve_split(
        self,
        task_order_idx: int,
        demo_offset: int,
        split_index_ood: int,
        split_index_ind: int,
    ) -> str:
        if task_order_idx < split_index_ood:
            return "train" if demo_offset < split_index_ind else "val"
        return "val_ood"

    def _make_action_sample(
        self,
        task_path: Path,
        task_name: str,
        task_id: int,
        demo_key: str,
        action_idx: int,
        action_count: int,
    ) -> dict[str, Any] | None:
        if action_idx > action_count - self.action_horizon:
            return None
        image_history_start_idx = max(0, action_idx - self.action_history + 1)
        return {
            "file_path": str(task_path),
            "task_name": task_name,
            "task_id": task_id,
            "demo_key": demo_key,
            "task_type": "action",
            "image_indices": list(range(image_history_start_idx, action_idx + 1)),
            "action_indices": list(range(action_idx, action_idx + self.action_horizon)),
            "state_index": action_idx,
        }

    def _make_world_sample(
        self,
        task_path: Path,
        task_name: str,
        task_id: int,
        demo_key: str,
        action_idx: int,
        action_count: int,
    ) -> dict[str, Any] | None:
        if action_idx > action_count - self.world_history - 1:
            return None
        historical_indices = list(range(max(action_idx - self.world_history + 1, 0), action_idx + 1))
        future_indices = list(range(action_idx + 1, action_idx + 2))
        return {
            "file_path": str(task_path),
            "task_name": task_name,
            "task_id": task_id,
            "demo_key": demo_key,
            "task_type": "world",
            "image_indices": historical_indices + future_indices,
            "action_indices": historical_indices,
            "future_image_indices": future_indices,
        }

    def _build_images(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        image_indices: list[int],
    ) -> list[Image.Image]:
        images: list[Image.Image] = []
        for image_idx in image_indices:
            images.append(_image_from_array(front_rgb[image_idx]))
            if self.with_wrist:
                images.append(_image_from_array(wrist_rgb[image_idx]))
        return images

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        file_path = Path(sample["file_path"])
        with h5py.File(file_path, "r") as handle:
            episode = handle["data"][sample["demo_key"]]
            obs_group = episode["obs"]
            front_rgb = obs_group["agentview_rgb"][()]
            wrist_rgb = obs_group["eye_in_hand_rgb"][()]
            actions = episode["actions"][()]
            ee_states = obs_group["ee_states"][()]
            gripper_states = obs_group["gripper_states"][()]

        images = self._build_images(front_rgb, wrist_rgb, sample["image_indices"])
        prompt_text = sample["task_name"].replace("_", " ")
        action_tensor = torch.tensor(
            np.asarray([copy.deepcopy(actions[idx]) for idx in sample["action_indices"]], dtype=np.float32),
            dtype=torch.float32,
        )

        if sample["task_type"] == "action":
            if self.with_state:
                state_idx = int(sample["state_index"])
                combined_state = np.concatenate([ee_states[state_idx], gripper_states[state_idx]]).astype(np.float32)
                conversations = [
                    {
                        "from": "human",
                        "value": f"What action should the robot take to {prompt_text}?"
                        + "<|state|>"
                        + "<|image|>" * len(images),
                    },
                    {"from": "gpt", "value": "<|action|>" * len(sample["action_indices"])},
                ]
            else:
                combined_state = np.zeros(0, dtype=np.float32)
                conversations = [
                    {
                        "from": "human",
                        "value": f"What action should the robot take to {prompt_text}?" + "<|image|>" * len(images),
                    },
                    {"from": "gpt", "value": "<|action|>" * len(sample["action_indices"])},
                ]
            target_images: list[Image.Image] = []
        else:
            combined_state = np.zeros(0, dtype=np.float32)
            future_indices = sample["future_image_indices"]
            target_images = self._build_images(front_rgb, wrist_rgb, future_indices)
            if self.with_wrist:
                human_value = (
                    "Generate the next image based on the provided sequence of historical images and corresponding "
                    "actions."
                    + "<|image|><|image|><|action|>" * len(sample["action_indices"])
                )
                assistant_value = "<|image|><|image|>"
            else:
                human_value = (
                    "Generate the next image based on the provided sequence of historical images and corresponding "
                    "actions."
                    + "<|image|><|action|>" * len(sample["action_indices"])
                )
                assistant_value = "<|image|>"
            conversations = [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": assistant_value},
            ]

        state_tensor = torch.tensor(combined_state, dtype=torch.float32)
        obs = {
            "conversations": conversations,
            "images": images,
            "prompt_text": prompt_text,
            "task_type": sample["task_type"],
            "task_id": int(sample["task_id"]),
            "state": state_tensor,
        }
        next_obs = {
            "images": target_images,
            "prompt_text": prompt_text,
            "task_type": sample["task_type"],
        }
        obs["encoder_inputs"] = {
            "prompt_text": [prompt_text],
            "conversations": [conversations],
            "images": [images],
            "state": state_tensor.unsqueeze(0),
            "state_mask": torch.ones(1, state_tensor.numel(), dtype=torch.bool) if state_tensor.numel() else torch.zeros(1, 0, dtype=torch.bool),
            "task_type": [sample["task_type"]],
            "task_id": torch.tensor([int(sample["task_id"])], dtype=torch.long),
        }
        next_obs["encoder_inputs"] = {
            "prompt_text": [prompt_text],
            "conversations": [[]],
            "images": [target_images],
            "state": None,
            "state_mask": None,
            "task_type": [sample["task_type"]],
            "task_id": torch.tensor([int(sample["task_id"])], dtype=torch.long),
        }
        return {
            "obs": obs,
            "next_obs": next_obs,
            "action": action_tensor,
            "reward": torch.zeros(1, dtype=torch.float32),
            "meta": {
                "file_path": str(file_path),
                "task_name": sample["task_name"],
                "task_id": int(sample["task_id"]),
                "demo_key": sample["demo_key"],
                "task_type": sample["task_type"],
                "image_indices": list(sample["image_indices"]),
                "action_indices": list(sample["action_indices"]),
            },
        }

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


__all__ = ["RynnVLADataSpec", "RynnVLALIBERODataset"]
