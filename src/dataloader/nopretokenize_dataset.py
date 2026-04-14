from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from PIL import Image

from src.dataloader.base_dataset import BaseDataset


@dataclass(frozen=True)
class NopretokenizeDataSpec:
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
    prompt_text: str


class NopretokenizeDataset(BaseDataset):
    def __init__(
        self,
        config_path: str | Path,
        resolution: int = 256,
        with_state: bool = True,
        with_wrist: bool = True,
        with_action: bool = True,
        with_world_model: bool = True,
        split_override: str | None = None,
        raw_data_dir_override: str | Path | None = None,
        task_suite_override: str | None = None,
    ) -> None:
        super().__init__()
        self.config_path = self.resolve_project_path(config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.load(handle, Loader=yaml.FullLoader)

        self.resolution = int(resolution)
        self.with_state = bool(with_state)
        self.with_wrist = bool(with_wrist)
        self.with_action = bool(with_action)
        self.with_world_model = bool(with_world_model)

        configured_split = str(self.config["META"].get("split", "all"))
        self.split = str(split_override) if split_override is not None else configured_split
        configured_task_suite = str(self.config["META"].get("libero_task_suite", "unknown"))
        self.task_suite_name = str(task_suite_override) if task_suite_override is not None else configured_task_suite
        configured_raw_data_dir = self.config["META"]["raw_data_dir"]
        raw_data_dir_value = raw_data_dir_override if raw_data_dir_override is not None else configured_raw_data_dir
        self.raw_data_dir = self.resolve_project_path(raw_data_dir_value, base_dir=self.config_path.parent)
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Nopretokenize raw_data_dir does not exist: {self.raw_data_dir}")

        self.action_horizon = int(self.config["action_model"]["len_action"])
        self.action_history = int(self.config["action_model"]["his"])
        self.world_history = int(self.config["world_model"]["his"])
        self.prompt_text = str(self.config.get("prompt_text", "Finish the task: {task_text}."))

        self.task_files = self.discover_task_files(self.raw_data_dir, self.task_suite_name)
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

        self._data_spec = NopretokenizeDataSpec(
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
            prompt_text=self.prompt_text,
        )

    @property
    def data_spec(self) -> NopretokenizeDataSpec:
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
                demo_keys = self.list_demo_keys(data_group)
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
                        sample = self._make_sample(
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

    def _make_sample(
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
        if action_idx >= action_count - 1:
            return None

        image_history_start_idx = max(0, action_idx - self.action_history + 1)
        conditioning_action_start_idx = max(0, action_idx - self.world_history + 1)
        return {
            "file_path": str(task_path),
            "task_name": task_name,
            "task_id": task_id,
            "demo_key": demo_key,
            "image_indices": list(range(image_history_start_idx, action_idx + 1)),
            "target_action_indices": list(range(action_idx, action_idx + self.action_horizon)),
            "conditioning_action_indices": list(range(conditioning_action_start_idx, action_idx + 1)),
            "target_image_indices": [action_idx + 1],
            "state_index": action_idx,
        }

    def _build_images(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        image_indices: list[int],
    ) -> list[Any]:
        images: list[Any] = []
        for image_idx in image_indices:
            images.append(self.image_from_array(front_rgb[image_idx]))
            if self.with_wrist:
                images.append(self.image_from_array(wrist_rgb[image_idx]))
        return images

    def _format_prompt_text(self, task_text: str) -> str:
        template = self.prompt_text
        try:
            return str(template.format(task_text=task_text, task_name=task_text))
        except (IndexError, KeyError, ValueError):
            return template

    def _build_record(
        self,
        sample: dict[str, Any],
        images: list[Image.Image],
        actions: np.ndarray,
        rewards: np.ndarray,
        ee_states: np.ndarray,
        gripper_states: np.ndarray,
        target_images: list[Image.Image],
    ) -> tuple[dict[str, Any], float]:
        task_text = sample["task_name"].replace("_", " ")
        prompt_text = self._format_prompt_text(task_text)
        target_action_indices = list(sample["target_action_indices"])
        conditioning_action_indices = list(sample["conditioning_action_indices"])
        action_targets = [copy.deepcopy(actions[idx]) for idx in target_action_indices]
        conditioning_actions = [copy.deepcopy(actions[idx]) for idx in conditioning_action_indices]
        reward_value = float(rewards[int(sample["target_image_indices"][-1])])

        state_paths: list[Any] = []
        if self.with_state:
            state_idx = int(sample["state_index"])
            combined_state = np.concatenate([ee_states[state_idx], gripper_states[state_idx]]).astype(np.float32)
            state_paths = [combined_state]
            human_value = prompt_text + "<|state|>" * len(state_paths) + "<|image|>" * len(images)
        else:
            human_value = prompt_text + "<|image|>" * len(images)

        record = {
            "task_name": sample["task_name"],
            "task_text": task_text,
            "prompt_text": prompt_text,
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": "<|action|>" * len(action_targets)},
            ],
            "image": copy.deepcopy(images),
            "action": copy.deepcopy(action_targets),
            "state": copy.deepcopy(state_paths),
            "input_image": copy.deepcopy(images),
            "input_action": copy.deepcopy(conditioning_actions),
            "input_state": copy.deepcopy(state_paths),
            "target_image": copy.deepcopy(target_images),
            "target_action": copy.deepcopy(action_targets),
            "target_state": [],
        }
        return record, reward_value

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        file_path = Path(sample["file_path"])
        with h5py.File(file_path, "r") as handle:
            episode = handle["data"][sample["demo_key"]]
            obs_group = episode["obs"]
            front_rgb = obs_group["agentview_rgb"][()]
            wrist_rgb = obs_group["eye_in_hand_rgb"][()]
            actions = episode["actions"][()]
            rewards = episode["rewards"][()]
            ee_states = obs_group["ee_states"][()]
            gripper_states = obs_group["gripper_states"][()]

        images = self._build_images(front_rgb, wrist_rgb, sample["image_indices"])
        target_images = self._build_images(front_rgb, wrist_rgb, sample["target_image_indices"])
        record, reward_value = self._build_record(
            sample=sample,
            images=images,
            actions=actions,
            rewards=rewards,
            ee_states=ee_states,
            gripper_states=gripper_states,
            target_images=target_images,
        )

        prompt_text = str(record["prompt_text"])
        task_text = str(record["task_text"])
        images = list(record["input_image"])
        target_images = list(record["target_image"])
        conversations = list(record["conversations"])

        state_array = np.asarray(record["input_state"][0], dtype=np.float32) if record["input_state"] else np.zeros(0, dtype=np.float32)
        state_tensor = torch.tensor(state_array, dtype=torch.float32)
        conditioning_action_array = np.asarray(record["input_action"], dtype=np.float32) if record["input_action"] else np.zeros((0, self.action_dim), dtype=np.float32)
        target_action_array = np.asarray(record["target_action"], dtype=np.float32) if record["target_action"] else np.zeros((0, self.action_dim), dtype=np.float32)
        full_action_array = np.asarray(record["action"], dtype=np.float32) if record["action"] else np.zeros((0, self.action_dim), dtype=np.float32)
        conditioning_action = torch.tensor(conditioning_action_array, dtype=torch.float32)
        target_action = torch.tensor(target_action_array, dtype=torch.float32)
        action_tensor = torch.tensor(full_action_array, dtype=torch.float32)

        obs = {
            "conversations": conversations,
            "images": images,
            "prompt_text": prompt_text,
            "task_text": task_text,
            "task_id": int(sample["task_id"]),
            "state": state_tensor,
        }
        next_obs = {
            "images": target_images,
            "prompt_text": prompt_text,
            "task_text": task_text,
        }

        obs["encoder_inputs"] = {
            "prompt_text": [prompt_text],
            "conversations": [[]],
            "images": [images],
            "state": state_tensor.unsqueeze(0),
            "state_mask": torch.ones(1, state_tensor.numel(), dtype=torch.bool) if state_tensor.numel() else torch.zeros(1, 0, dtype=torch.bool),
            "task_id": torch.tensor([int(sample["task_id"])], dtype=torch.long),
        }
        next_obs["encoder_inputs"] = {
            "prompt_text": [prompt_text],
            "conversations": [[]],
            "images": [target_images],
            "state": None,
            "state_mask": None,
            "task_id": torch.tensor([int(sample["task_id"])], dtype=torch.long),
        }

        return {
            "obs": obs,
            "next_obs": next_obs,
            "action": action_tensor,
            "conditioning_action": conditioning_action,
            "target_action": target_action,
            "reward": torch.tensor([reward_value], dtype=torch.float32),
            "record": record,
            "meta": {
                "file_path": str(file_path),
                "task_name": sample["task_name"],
                "task_text": task_text,
                "prompt_text": prompt_text,
                "task_id": int(sample["task_id"]),
                "demo_key": sample["demo_key"],
                "image_indices": list(sample["image_indices"]),
                "input_image_indices": list(sample["image_indices"]),
                "target_image_indices": list(sample["target_image_indices"]),
                "conditioning_action_indices": list(sample["conditioning_action_indices"]),
                "target_action_indices": list(sample["target_action_indices"]),
            },
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        obs_list = [item["obs"] for item in batch]
        next_obs_list = [item["next_obs"] for item in batch]
        actions = [item["action"] for item in batch]
        conditioning_actions = [item["conditioning_action"] for item in batch]
        target_actions = [item["target_action"] for item in batch]
        rewards = torch.stack([item["reward"] for item in batch], dim=0)
        states = [obs["state"] for obs in obs_list]
        records = [item["record"] for item in batch]

        padded_actions, action_mask = NopretokenizeDataset.pad_action_batch(actions)
        padded_conditioning_actions, conditioning_action_mask = NopretokenizeDataset.pad_action_batch(conditioning_actions)
        padded_target_actions, target_action_mask = NopretokenizeDataset.pad_action_batch(target_actions)
        padded_states, state_mask = NopretokenizeDataset.pad_state_batch(states)

        collated_obs = {
            "conversations": [obs["conversations"] for obs in obs_list],
            "images": [obs["images"] for obs in obs_list],
            "prompt_text": [obs["prompt_text"] for obs in obs_list],
            "task_text": [obs.get("task_text", obs["prompt_text"]) for obs in obs_list],
            "task_id": NopretokenizeDataset.stack_long([int(obs["task_id"]) for obs in obs_list]),
            "state": padded_states,
            "state_mask": state_mask,
            "raw_state": states,
        }
        collated_next_obs = {
            "images": [obs["images"] for obs in next_obs_list],
            "prompt_text": [obs["prompt_text"] for obs in next_obs_list],
        }
        collated_obs["encoder_inputs"] = {
            "prompt_text": collated_obs["prompt_text"],
            "conversations": [[] for _ in obs_list],
            "images": collated_obs["images"],
            "state": padded_states,
            "state_mask": state_mask,
            "action": padded_conditioning_actions,
            "action_mask": conditioning_action_mask,
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
            "task_id": NopretokenizeDataset.stack_long([int(item["meta"]["task_id"]) for item in batch]),
            "meta": [item["meta"] for item in batch],
        }

        return {
            "obs": collated_obs,
            "next_obs": collated_next_obs,
            "action": padded_actions,
            "action_mask": action_mask,
            "conditioning_action": padded_conditioning_actions,
            "conditioning_action_mask": conditioning_action_mask,
            "target_action": padded_target_actions,
            "target_action_mask": target_action_mask,
            "raw_action": actions,
            "reward": rewards,
            "record": records,
            "meta": [item["meta"] for item in batch],
        }


__all__ = ["NopretokenizeDataSpec", "NopretokenizeDataset"]
