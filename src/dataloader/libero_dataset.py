from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from src.dataloader.base_dataset import BaseDataset
from src.dataloader.transition_dataset import SimpleWordTokenizer

DEFAULT_LIBERO_ROOT = Path("/home/yuxinglei/workspace/2026nips/RynnVLA-002/LIBERO")


def _resolve_libero_paths(
    libero_root: str | Path | None,
    datasets_root: str | Path | None,
) -> tuple[Path, Path]:
    if datasets_root is not None:
        resolved_datasets_root = BaseDataset.resolve_project_path(datasets_root)
        resolved_libero_root = (
            BaseDataset.resolve_project_path(libero_root) if libero_root is not None else resolved_datasets_root.parents[1]
        )
        return resolved_libero_root, resolved_datasets_root

    if libero_root is None:
        libero_root = DEFAULT_LIBERO_ROOT

    resolved_libero_root = BaseDataset.resolve_project_path(libero_root)
    return resolved_libero_root, (resolved_libero_root / "libero" / "datasets").resolve()


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        try:
            return value.decode("utf-8")
        except Exception:
            pass
    return str(value)


def _infer_task_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_demo"):
        stem = stem[:-5]
    return stem.replace("_", " ")


def _load_prompt_text(data_group: h5py.Group, path: Path) -> str:
    problem_info_raw = data_group.attrs.get("problem_info")
    if problem_info_raw is not None:
        try:
            problem_info = json.loads(_decode_attr(problem_info_raw))
            language_instruction = problem_info.get("language_instruction")
            if isinstance(language_instruction, str) and language_instruction.strip():
                return language_instruction.strip()
        except Exception:
            pass
    return _infer_task_name_from_path(path)


def _safe_std(sum_sq: np.ndarray, mean: np.ndarray, count: int) -> np.ndarray:
    variance = np.maximum(sum_sq / max(count, 1) - mean ** 2, 1e-6)
    return np.sqrt(variance).astype(np.float32)


@dataclass(frozen=True)
class LIBERODataSpec:
    libero_root: str
    datasets_root: str
    suite_names: tuple[str, ...]
    num_files: int
    num_transitions: int
    action_dim: int
    proprio_dim: int
    vocab_size: int
    max_text_length: int
    image_key: str
    wrist_image_key: str | None
    proprio_keys: tuple[str, ...]


@dataclass(frozen=True)
class _TransitionEntry:
    file_path: str
    suite_name: str
    demo_key: str
    step_index: int
    episode_length: int
    prompt_text: str
    task_name: str


class LIBEROTransitionDataset(BaseDataset):
    def __init__(
        self,
        libero_root: str | Path | None = None,
        datasets_root: str | Path | None = None,
        suite_names: Sequence[str] | None = None,
        image_size: int = 256,
        max_language_length: int = 32,
        lowercase_text: bool = True,
        image_key: str = "agentview_rgb",
        wrist_image_key: str | None = "eye_in_hand_rgb",
        proprio_keys: Sequence[str] = ("ee_states", "gripper_states"),
        terminal_reward: float | None = None,
        default_reward: float = 0.0,
        max_files: int | None = None,
        max_demos_per_file: int | None = None,
        max_transitions: int | None = None,
    ) -> None:
        super().__init__()
        self.libero_root, self.datasets_root = _resolve_libero_paths(libero_root, datasets_root)
        if not self.datasets_root.exists():
            raise FileNotFoundError(f"LIBERO datasets root does not exist: {self.datasets_root}")

        if suite_names is None:
            discovered_suites = sorted(
                path.name for path in self.datasets_root.iterdir() if path.is_dir() and path.name.startswith("libero_")
            )
            if not discovered_suites:
                raise RuntimeError(f"No LIBERO suites found under {self.datasets_root}")
            self.suite_names = tuple(discovered_suites)
        else:
            self.suite_names = tuple(str(name) for name in suite_names)

        self.image_size = int(image_size)
        self.image_key = str(image_key)
        self.wrist_image_key = str(wrist_image_key) if wrist_image_key is not None else None
        self.proprio_keys = tuple(str(key) for key in proprio_keys)
        self.terminal_reward = None if terminal_reward is None else float(terminal_reward)
        self.default_reward = float(default_reward)
        self._hdf5_open_kwargs = {"mode": "r", "swmr": True, "libver": "latest"}
        self._file_cache: dict[str, h5py.File] = {}

        self._transition_index: list[_TransitionEntry] = []
        self._prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._episode_lengths: dict[tuple[str, str], int] = {}

        action_sum: np.ndarray | None = None
        action_sq_sum: np.ndarray | None = None
        proprio_sum: np.ndarray | None = None
        proprio_sq_sum: np.ndarray | None = None
        transition_count = 0
        prompt_texts: list[str] = []
        discovered_files: list[Path] = []

        for suite_name in self.suite_names:
            suite_dir = self.datasets_root / suite_name
            if not suite_dir.exists():
                raise FileNotFoundError(f"LIBERO suite directory not found: {suite_dir}")
            suite_files = sorted(suite_dir.glob("*.hdf5"))
            discovered_files.extend(suite_files)

        if max_files is not None:
            discovered_files = discovered_files[: int(max_files)]
        if not discovered_files:
            raise RuntimeError(f"No LIBERO HDF5 files found under {self.datasets_root}")

        self.action_dim = 0
        self.proprio_dim = 0

        stop_indexing = False
        for file_path in discovered_files:
            suite_name = file_path.parent.name
            with h5py.File(file_path, **self._hdf5_open_kwargs) as handle:
                data_group = handle["data"]
                prompt_text = _load_prompt_text(data_group, file_path)
                prompt_texts.append(prompt_text)
                demo_keys = sorted(data_group.keys())
                if max_demos_per_file is not None:
                    demo_keys = demo_keys[: int(max_demos_per_file)]

                for demo_key in demo_keys:
                    episode = data_group[demo_key]
                    obs_group = episode["obs"]
                    actions = np.asarray(episode["actions"], dtype=np.float32)
                    episode_length = int(actions.shape[0])
                    if episode_length <= 1:
                        continue

                    proprio_parts = [
                        np.asarray(obs_group[key], dtype=np.float32).reshape(episode_length, -1)
                        for key in self.proprio_keys
                    ]
                    proprio = np.concatenate(proprio_parts, axis=-1)
                    num_transitions_in_demo = episode_length - 1

                    if self.action_dim == 0:
                        self.action_dim = int(actions.shape[-1])
                        self.proprio_dim = int(proprio.shape[-1])
                        action_sum = np.zeros(self.action_dim, dtype=np.float64)
                        action_sq_sum = np.zeros(self.action_dim, dtype=np.float64)
                        proprio_sum = np.zeros(self.proprio_dim, dtype=np.float64)
                        proprio_sq_sum = np.zeros(self.proprio_dim, dtype=np.float64)

                    if max_transitions is not None:
                        remaining = int(max_transitions) - len(self._transition_index)
                        if remaining <= 0:
                            stop_indexing = True
                            break
                        num_transitions_in_demo = min(num_transitions_in_demo, remaining)

                    if num_transitions_in_demo <= 0:
                        continue

                    selected_actions = actions[:num_transitions_in_demo]
                    selected_proprio = proprio[:num_transitions_in_demo]
                    action_sum += selected_actions.sum(axis=0)
                    action_sq_sum += np.square(selected_actions).sum(axis=0)
                    proprio_sum += selected_proprio.sum(axis=0)
                    proprio_sq_sum += np.square(selected_proprio).sum(axis=0)
                    transition_count += num_transitions_in_demo

                    task_name = _infer_task_name_from_path(file_path)
                    file_key = str(file_path)
                    self._episode_lengths[(file_key, demo_key)] = episode_length
                    for step_index in range(num_transitions_in_demo):
                        self._transition_index.append(
                            _TransitionEntry(
                                file_path=file_key,
                                suite_name=suite_name,
                                demo_key=demo_key,
                                step_index=step_index,
                                episode_length=episode_length,
                                prompt_text=prompt_text,
                                task_name=task_name,
                            )
                        )

                if stop_indexing:
                    break
            if stop_indexing:
                break

        if not self._transition_index:
            raise RuntimeError(f"No transitions indexed from LIBERO datasets under {self.datasets_root}")
        assert action_sum is not None and action_sq_sum is not None
        assert proprio_sum is not None and proprio_sq_sum is not None

        self.tokenizer = SimpleWordTokenizer(prompt_texts, lowercase=lowercase_text)
        self.max_language_length = max(int(max_language_length), self.tokenizer.max_text_length)
        unique_prompts = sorted(set(prompt_texts))
        for prompt_text in unique_prompts:
            self._prompt_cache[prompt_text] = self.tokenizer.encode(prompt_text, self.max_language_length)

        self.action_mean = torch.from_numpy((action_sum / transition_count).astype(np.float32))
        self.action_std = torch.from_numpy(
            _safe_std(action_sq_sum, self.action_mean.numpy(), transition_count)
        )
        self.proprio_mean = torch.from_numpy((proprio_sum / transition_count).astype(np.float32))
        self.proprio_std = torch.from_numpy(
            _safe_std(proprio_sq_sum, self.proprio_mean.numpy(), transition_count)
        )

        self._data_spec = LIBERODataSpec(
            libero_root=str(self.libero_root),
            datasets_root=str(self.datasets_root),
            suite_names=self.suite_names,
            num_files=len(discovered_files),
            num_transitions=len(self._transition_index),
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            vocab_size=self.tokenizer.vocab_size,
            max_text_length=self.max_language_length,
            image_key=self.image_key,
            wrist_image_key=self.wrist_image_key,
            proprio_keys=self.proprio_keys,
        )

    @property
    def data_spec(self) -> LIBERODataSpec:
        return self._data_spec

    def get_normalizer(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "action": {"mean": self.action_mean.clone(), "std": self.action_std.clone()},
            "proprio": {"mean": self.proprio_mean.clone(), "std": self.proprio_std.clone()},
        }

    def __len__(self) -> int:
        return len(self._transition_index)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file_cache"] = {}
        return state

    def __del__(self) -> None:
        for handle in self._file_cache.values():
            try:
                handle.close()
            except Exception:
                pass

    def _require_hdf5_file(self, file_path: str) -> h5py.File:
        if file_path not in self._file_cache:
            self._file_cache[file_path] = h5py.File(file_path, **self._hdf5_open_kwargs)
        return self._file_cache[file_path]

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

    def _proprio_to_numpy(self, obs_group: h5py.Group, step_index: int) -> np.ndarray:
        parts = [
            np.asarray(obs_group[key][step_index], dtype=np.float32).reshape(-1)
            for key in self.proprio_keys
        ]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def _build_obs(self, episode: h5py.Group, step_index: int, prompt_text: str) -> dict[str, Any]:
        obs_group = episode["obs"]
        front_image = np.asarray(obs_group[self.image_key][step_index], dtype=np.uint8)
        wrist_key = self.wrist_image_key or self.image_key
        wrist_image = np.asarray(obs_group[wrist_key][step_index], dtype=np.uint8)
        proprio = self._proprio_to_numpy(obs_group, step_index)
        text, attention_mask = self._prompt_cache[prompt_text]

        return {
            "image": self._image_to_tensor(front_image),
            "proprio": torch.from_numpy(proprio.copy()),
            "text": text.clone(),
            "text_attention_mask": attention_mask.clone(),
            "prompt_text": prompt_text,
            "raw_obs": {
                "rgb_obs": {
                    "rgb_static": front_image.copy(),
                    "wrist_static": wrist_image.copy(),
                },
                "state": proprio.copy(),
            },
        }

    def _resolve_reward(self, episode: h5py.Group, step_index: int, episode_length: int) -> float:
        if "rewards" in episode:
            return float(np.asarray(episode["rewards"][step_index], dtype=np.float32))
        if self.terminal_reward is not None and step_index + 1 == episode_length - 1:
            return self.terminal_reward
        return self.default_reward

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self._transition_index[index]
        handle = self._require_hdf5_file(entry.file_path)
        episode = handle["data"][entry.demo_key]
        next_step_index = entry.step_index + 1

        action = torch.from_numpy(np.asarray(episode["actions"][entry.step_index], dtype=np.float32))
        reward = self._resolve_reward(episode, entry.step_index, entry.episode_length)

        return {
            "obs": self._build_obs(episode, entry.step_index, entry.prompt_text),
            "next_obs": self._build_obs(episode, next_step_index, entry.prompt_text),
            "action": action,
            "reward": torch.tensor([reward], dtype=torch.float32),
            "meta": {
                "suite_name": entry.suite_name,
                "task_name": entry.task_name,
                "prompt_text": entry.prompt_text,
                "file_path": entry.file_path,
                "demo_key": entry.demo_key,
                "step_index": entry.step_index,
            },
        }


__all__ = ["LIBERODataSpec", "LIBEROTransitionDataset"]
