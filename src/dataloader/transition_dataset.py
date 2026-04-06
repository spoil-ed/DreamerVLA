from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from src.dataloader.base_dataset import BaseDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    direct_candidate = (PROJECT_ROOT / path).resolve()
    if direct_candidate.exists():
        return direct_candidate
    data_candidate = (PROJECT_ROOT / "data" / path).resolve()
    if data_candidate.exists():
        return data_candidate
    return direct_candidate


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _infer_task_name(train_config: Mapping[str, Any], hdf5_path: Path) -> str:
    experiment_name = str(train_config.get("experiment", {}).get("name", ""))
    match = re.search(r"_ds_(.+?)_D\d+", experiment_name)
    if match is not None:
        return match.group(1).replace("_", " ")
    return hdf5_path.parent.name.replace("_", " ")


class SimpleWordTokenizer:
    def __init__(self, texts: Sequence[str], lowercase: bool = True) -> None:
        self.lowercase = bool(lowercase)
        self.pad_token_id = 0
        self.unk_token_id = 1
        self._vocab: dict[str, int] = {
            "<pad>": self.pad_token_id,
            "<unk>": self.unk_token_id,
        }

        for text in texts:
            for token in self._tokenize(text):
                if token not in self._vocab:
                    self._vocab[token] = len(self._vocab)

        self.max_text_length = max((len(self._tokenize(text)) for text in texts), default=1)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def _normalize(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def _tokenize(self, text: str) -> list[str]:
        normalized = self._normalize(text).strip()
        if not normalized:
            return ["<unk>"]
        return normalized.split()

    def encode(self, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [self._vocab.get(token, self.unk_token_id) for token in self._tokenize(text)]
        tokens = tokens[:max_length]
        attention_mask = [1] * len(tokens)
        if len(tokens) < max_length:
            padding = max_length - len(tokens)
            tokens.extend([self.pad_token_id] * padding)
            attention_mask.extend([0] * padding)
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.bool),
        )


@dataclass(frozen=True)
class TrainingDataSpec:
    train_config_path: str
    hdf5_path: str
    task_name: str
    num_transitions: int
    action_dim: int
    proprio_dim: int
    vocab_size: int
    max_text_length: int
    image_key: str
    wrist_image_key: str | None
    low_dim_keys: tuple[str, ...]


class TransitionDataset(BaseDataset):
    def __init__(
        self,
        train_config_path: str | Path,
        image_size: int,
        max_language_length: int,
        terminal_reward: float = 1.0,
        default_reward: float = 0.0,
        lowercase_text: bool = True,
        image_key: str | None = None,
        wrist_image_key: str | None = None,
        low_dim_keys: Sequence[str] | None = None,
        max_episodes: int | None = None,
        max_transitions: int | None = None,
    ) -> None:
        super().__init__()
        self.train_config_path = _resolve_project_path(train_config_path)
        train_config = _load_json(self.train_config_path)

        train_section = train_config["train"]
        obs_modalities = train_config["observation"]["modalities"]["obs"]
        default_image_key = str(obs_modalities["rgb"][0])
        default_wrist_key = str(obs_modalities["rgb"][1]) if len(obs_modalities["rgb"]) > 1 else None
        default_low_dim_keys = tuple(str(key) for key in obs_modalities["low_dim"])

        self.hdf5_path = _resolve_project_path(train_section["data"])
        self.image_size = int(image_size)
        self.terminal_reward = float(terminal_reward)
        self.default_reward = float(default_reward)
        self.image_key = str(image_key or default_image_key)
        self.wrist_image_key = (
            str(wrist_image_key) if wrist_image_key is not None else default_wrist_key
        )
        self.low_dim_keys = tuple(low_dim_keys or default_low_dim_keys)
        self.task_name = _infer_task_name(train_config, self.hdf5_path)
        self.tokenizer = SimpleWordTokenizer([self.task_name], lowercase=lowercase_text)
        self.max_language_length = max(int(max_language_length), self.tokenizer.max_text_length)
        self._encoded_text = self.tokenizer.encode(self.task_name, self.max_language_length)
        self._hdf5_file: h5py.File | None = None

        self._hdf5_open_kwargs = {
            "mode": "r",
            "swmr": bool(train_section.get("hdf5_use_swmr", True)),
            "libver": "latest",
        }

        self._transition_index: list[tuple[str, int]] = []
        self._episode_lengths: dict[str, int] = {}
        with h5py.File(self.hdf5_path, **self._hdf5_open_kwargs) as handle:
            episode_keys = sorted(handle["data"].keys())
            if max_episodes is not None:
                episode_keys = episode_keys[: int(max_episodes)]

            first_episode = handle["data"][episode_keys[0]]
            self.action_dim = int(first_episode["actions"].shape[-1])
            self.proprio_dim = int(
                sum(int(np.prod(first_episode["obs"][key].shape[1:])) for key in self.low_dim_keys)
            )

            for episode_key in episode_keys:
                episode = handle["data"][episode_key]
                episode_length = int(episode["actions"].shape[0])
                self._episode_lengths[episode_key] = episode_length
                for step_index in range(max(episode_length - 1, 0)):
                    self._transition_index.append((episode_key, step_index))

        if max_transitions is not None:
            self._transition_index = self._transition_index[: int(max_transitions)]

        self._data_spec = TrainingDataSpec(
            train_config_path=str(self.train_config_path),
            hdf5_path=str(self.hdf5_path),
            task_name=self.task_name,
            num_transitions=len(self._transition_index),
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            vocab_size=self.tokenizer.vocab_size,
            max_text_length=self.max_language_length,
            image_key=self.image_key,
            wrist_image_key=self.wrist_image_key,
            low_dim_keys=self.low_dim_keys,
        )

    @property
    def data_spec(self) -> TrainingDataSpec:
        return self._data_spec

    def get_normalizer(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "action": {
                "mean": torch.zeros(self.action_dim, dtype=torch.float32),
                "std": torch.ones(self.action_dim, dtype=torch.float32),
            },
            "proprio": {
                "mean": torch.zeros(self.proprio_dim, dtype=torch.float32),
                "std": torch.ones(self.proprio_dim, dtype=torch.float32),
            },
        }

    def __len__(self) -> int:
        return len(self._transition_index)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_hdf5_file"] = None
        return state

    def __del__(self) -> None:
        file_handle = getattr(self, "_hdf5_file", None)
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception:
                pass

    def _require_hdf5_file(self) -> h5py.File:
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, **self._hdf5_open_kwargs)
        return self._hdf5_file

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor.contiguous()

    def _proprio_to_tensor(self, obs_group: h5py.Group, step_index: int) -> torch.Tensor:
        parts = [
            np.asarray(obs_group[key][step_index], dtype=np.float32).reshape(-1)
            for key in self.low_dim_keys
        ]
        return torch.from_numpy(np.concatenate(parts, axis=0))

    def _build_obs(self, episode: h5py.Group, step_index: int) -> dict[str, Any]:
        obs_group = episode["obs"]
        text, attention_mask = self._encoded_text
        front_image = np.asarray(obs_group[self.image_key][step_index], dtype=np.uint8)
        wrist_key = self.wrist_image_key or self.image_key
        wrist_image = np.asarray(obs_group[wrist_key][step_index], dtype=np.uint8)
        proprio = self._proprio_to_tensor(obs_group, step_index)
        return {
            "image": self._image_to_tensor(front_image),
            "text": text.clone(),
            "proprio": proprio.clone(),
            "text_attention_mask": attention_mask.clone(),
            "prompt_text": self.task_name,
            "raw_obs": {
                "rgb_obs": {
                    "rgb_static": front_image.copy(),
                    "wrist_static": wrist_image.copy(),
                },
                "state": proprio.detach().cpu().numpy().copy(),
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_key, step_index = self._transition_index[index]
        handle = self._require_hdf5_file()
        episode = handle["data"][episode_key]
        next_step_index = step_index + 1
        episode_length = self._episode_lengths[episode_key]

        action = torch.from_numpy(np.asarray(episode["actions"][step_index], dtype=np.float32))
        reward_value = self.default_reward
        if next_step_index == episode_length - 1:
            reward_value = self.terminal_reward

        return {
            "obs": self._build_obs(episode, step_index),
            "next_obs": self._build_obs(episode, next_step_index),
            "action": action,
            "reward": torch.tensor([reward_value], dtype=torch.float32),
            "meta": {
                "task_name": self.task_name,
                "episode_key": episode_key,
                "step_index": step_index,
            },
        }


__all__ = ["TransitionDataset", "TrainingDataSpec"]
