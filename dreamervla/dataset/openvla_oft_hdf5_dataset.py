from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

_DEMO_RE = re.compile(r"^demo_(\d+)$")


@dataclass(frozen=True)
class OpenVLAOFTHDF5Spec:
    hdf5_dir: str
    num_files: int
    num_samples: int
    action_horizon: int
    image_keys: tuple[str, ...]
    use_proprio: bool
    one_trajectory_sft: bool = False
    demos_per_task: int | None = None
    demo_selection_seed: int | None = None


@dataclass(frozen=True)
class _HDF5Sample:
    file_path: str
    demo_key: str
    index: int


def _list_demo_keys(data_group: h5py.Group) -> list[str]:
    keys = list(data_group.keys())
    return sorted(
        keys,
        key=lambda key: int(_DEMO_RE.match(key).group(1))
        if _DEMO_RE.match(key)
        else key,
    )


def _task_from_path(path: str | Path) -> str:
    stem = Path(path).name
    if stem.endswith("_demo.hdf5"):
        stem = stem[: -len("_demo.hdf5")]
    else:
        stem = Path(stem).stem
    return stem.replace("_", " ").strip().lower()


def _normalize_bounds_q99(
    values: np.ndarray, stats: dict[str, Any], mask_default: bool = True
) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    if not mask_default and "mask" not in stats:
        mask = np.zeros_like(low, dtype=bool)
    normalized = np.clip(2.0 * (values - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)
    return np.where(mask, normalized, values).astype(np.float32, copy=False)


def _libero_oft_action_transform(actions: np.ndarray) -> np.ndarray:
    actions = actions.astype(np.float32, copy=True)
    gripper = actions[:, -1:]
    # Match OpenVLA-OFT's LIBERO transform: -1=open, +1=close -> 1=open, 0=close.
    actions[:, -1:] = 1.0 - np.clip(gripper, 0.0, 1.0)
    return actions


def _select_demo_keys(
    demo_keys: Sequence[str],
    *,
    file_path: Path,
    demos_per_task: int | None,
    demo_selection_seed: int,
    max_demos_per_file: int | None,
) -> list[str]:
    ordered = list(demo_keys)
    if demos_per_task is not None:
        count = int(demos_per_task)
        if count < 1:
            raise ValueError("demos_per_task must be >= 1 when set.")
        rng = random.Random(f"{int(demo_selection_seed)}:{file_path.name}")
        return sorted(
            rng.sample(ordered, k=min(count, len(ordered))), key=ordered.index
        )
    if max_demos_per_file is not None:
        return ordered[: int(max_demos_per_file)]
    return ordered


class OpenVLAOFTHDF5Dataset(Dataset):
    """Map-style LIBERO HDF5 dataset that emits OpenVLA-OFT training samples."""

    def __init__(
        self,
        hdf5_dir: str | Path,
        processor: Any,
        action_tokenizer: Any,
        dataset_statistics: dict[str, Any],
        action_horizon: int = 8,
        image_keys: Sequence[str] = ("agentview_rgb", "eye_in_hand_rgb"),
        use_wrist_image: bool = True,
        use_proprio: bool = True,
        max_files: int | None = None,
        max_demos_per_file: int | None = None,
        demos_per_task: int | None = None,
        demo_selection_seed: int = 0,
        max_samples: int | None = None,
    ) -> None:
        ensure_openvla_oft_on_path()
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.vla.constants import IGNORE_INDEX

        self.hdf5_dir = Path(hdf5_dir).expanduser().resolve()
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.dataset_statistics = dataset_statistics
        self.action_horizon = int(action_horizon)
        self.image_keys = tuple(str(key) for key in image_keys)
        self.use_wrist_image = bool(use_wrist_image)
        self.use_proprio = bool(use_proprio)
        self.demos_per_task = None if demos_per_task is None else int(demos_per_task)
        self.demo_selection_seed = int(demo_selection_seed)
        self.prompt_builder_cls = PurePromptBuilder
        self.ignore_index = int(IGNORE_INDEX)
        self._hdf5_open_kwargs = {"mode": "r", "swmr": True, "libver": "latest"}
        self._file_cache: dict[str, h5py.File] = {}

        files = sorted(self.hdf5_dir.glob("*.hdf5"))
        if max_files is not None:
            files = files[: int(max_files)]
        if not files:
            raise RuntimeError(f"No HDF5 files found under {self.hdf5_dir}")

        self.samples: list[_HDF5Sample] = []
        stop = False
        for file_path in files:
            with h5py.File(file_path, **self._hdf5_open_kwargs) as handle:
                data = handle["data"]
                demo_keys = _select_demo_keys(
                    _list_demo_keys(data),
                    file_path=file_path,
                    demos_per_task=self.demos_per_task,
                    demo_selection_seed=self.demo_selection_seed,
                    max_demos_per_file=max_demos_per_file,
                )
                for demo_key in demo_keys:
                    demo = data[demo_key]
                    length = int(demo["actions"].shape[0])
                    obs_group = demo["obs"]
                    for key in self.image_keys:
                        if key not in obs_group:
                            raise KeyError(f"{file_path}:{demo_key} missing obs/{key}")
                    for index in range(length):
                        self.samples.append(
                            _HDF5Sample(str(file_path), demo_key, index)
                        )
                        if max_samples is not None and len(self.samples) >= int(
                            max_samples
                        ):
                            stop = True
                            break
                    if stop:
                        break
            if stop:
                break

        self._spec = OpenVLAOFTHDF5Spec(
            hdf5_dir=str(self.hdf5_dir),
            num_files=len(files),
            num_samples=len(self.samples),
            action_horizon=self.action_horizon,
            image_keys=self.image_keys,
            use_proprio=self.use_proprio,
            one_trajectory_sft=self.demos_per_task == 1,
            demos_per_task=self.demos_per_task,
            demo_selection_seed=self.demo_selection_seed
            if self.demos_per_task is not None
            else None,
        )

    @property
    def data_spec(self) -> OpenVLAOFTHDF5Spec:
        return self._spec

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self, path: str) -> h5py.File:
        handle = self._file_cache.get(path)
        if handle is None:
            handle = h5py.File(path, **self._hdf5_open_kwargs)
            self._file_cache[path] = handle
        return handle

    def _action_chunk(self, demo: h5py.Group, index: int) -> np.ndarray:
        raw = np.asarray(demo["actions"], dtype=np.float32)
        indices = np.minimum(
            np.arange(index, index + self.action_horizon, dtype=np.int64),
            raw.shape[0] - 1,
        )
        actions = _libero_oft_action_transform(raw[indices])
        return _normalize_bounds_q99(actions, self.dataset_statistics["action"])

    def _proprio(self, obs_group: h5py.Group, index: int) -> np.ndarray:
        eef = np.asarray(obs_group["ee_states"][index], dtype=np.float32)
        gripper = np.asarray(obs_group["gripper_states"][index], dtype=np.float32)
        proprio = np.concatenate([eef, gripper], axis=0)
        return _normalize_bounds_q99(proprio, self.dataset_statistics["proprio"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[int(index)]
        handle = self._file(sample.file_path)
        demo = handle["data"][sample.demo_key]
        obs_group = demo["obs"]
        task = _task_from_path(sample.file_path)

        images = [
            Image.fromarray(
                np.asarray(obs_group[self.image_keys[0]][sample.index], dtype=np.uint8)
            )
        ]
        if self.use_wrist_image:
            images.append(
                Image.fromarray(
                    np.asarray(
                        obs_group[self.image_keys[1]][sample.index], dtype=np.uint8
                    )
                )
            )
        pixel_values = self.processor.image_processor.apply_transform(images[0])
        item: dict[str, Any] = {"pixel_values": pixel_values}
        if self.use_wrist_image:
            item["pixel_values_wrist"] = self.processor.image_processor.apply_transform(
                images[1]
            )

        actions = self._action_chunk(demo, sample.index)
        current_action_string = self.action_tokenizer(actions[0])
        future_actions_string = "".join(self.action_tokenizer(actions[1:]))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        prompt_builder = self.prompt_builder_cls("openvla")
        prompt_builder.add_turn(
            "human", f"What action should the robot take to {task}?"
        )
        prompt_builder.add_turn("gpt", action_chunk_string)
        input_ids = self.processor.tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        labels = list(input_ids)
        labels[: -(action_chunk_len + 1)] = [self.ignore_index] * (
            len(labels) - (action_chunk_len + 1)
        )

        item.update(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "actions": actions.astype(np.float32, copy=False),
                "dataset_name": "libero_goal_no_noops",
            }
        )
        if self.use_proprio:
            item["proprio"] = self._proprio(obs_group, sample.index)
        return item


class OpenVLAOFTHDF5DatasetFactory:
    def __init__(
        self,
        hdf5_dir: str | Path,
        dataset_statistics_path: str | Path | None = None,
        dataset_statistics_key: str = "libero_goal_no_noops",
        action_horizon: int = 8,
        image_keys: Sequence[str] = ("agentview_rgb", "eye_in_hand_rgb"),
        use_wrist_image: bool = True,
        use_proprio: bool = True,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        max_files: int | None = None,
        max_demos_per_file: int | None = None,
        demos_per_task: int | None = None,
        demo_selection_seed: int = 0,
        max_samples: int | None = None,
        **_unused_compat_kwargs: Any,
    ) -> None:
        self.hdf5_dir = str(Path(hdf5_dir).expanduser().resolve())
        self.dataset_statistics_path = (
            None
            if dataset_statistics_path is None
            else str(Path(dataset_statistics_path).expanduser().resolve())
        )
        self.dataset_statistics_key = str(dataset_statistics_key)
        self.action_horizon = int(action_horizon)
        self.image_keys = tuple(str(key) for key in image_keys)
        self.use_wrist_image = bool(use_wrist_image)
        self.use_proprio = bool(use_proprio)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.max_files = max_files
        self.max_demos_per_file = max_demos_per_file
        self.demos_per_task = demos_per_task
        self.demo_selection_seed = int(demo_selection_seed)
        self.max_samples = max_samples

    def _load_statistics(self, policy: Any) -> dict[str, Any]:
        path = self.dataset_statistics_path
        if path is None:
            path = str(Path(policy.model_path) / "dataset_statistics.json")
        with Path(path).open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        if self.dataset_statistics_key not in stats:
            raise KeyError(
                f"{path} does not contain dataset statistics key {self.dataset_statistics_key!r}"
            )
        return stats[self.dataset_statistics_key]

    def build(self, policy: Any, *, train: bool = True) -> Any:
        ensure_openvla_oft_on_path()
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction
        from prismatic.vla.action_tokenizer import ActionTokenizer

        from dreamervla.dataset.openvla_oft_rlds_dataset import OpenVLAOFTRLDSDatasetBundle

        stats = self._load_statistics(policy)
        action_tokenizer = ActionTokenizer(policy.processor.tokenizer)
        dataset = OpenVLAOFTHDF5Dataset(
            hdf5_dir=self.hdf5_dir,
            processor=policy.processor,
            action_tokenizer=action_tokenizer,
            dataset_statistics=stats,
            action_horizon=self.action_horizon,
            image_keys=self.image_keys,
            use_wrist_image=self.use_wrist_image,
            use_proprio=self.use_proprio,
            max_files=self.max_files,
            max_demos_per_file=self.max_demos_per_file,
            demos_per_task=self.demos_per_task,
            demo_selection_seed=self.demo_selection_seed,
            max_samples=self.max_samples,
        )
        collator = PaddedCollatorForActionPrediction(
            policy.processor.tokenizer.model_max_length,
            policy.processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        sampler = None
        shuffle = self.shuffle if train else False
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=self.drop_last,
            )
            shuffle = False
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            collate_fn=collator,
            num_workers=self.num_workers,
        )
        return OpenVLAOFTRLDSDatasetBundle(
            dataset=dataset,
            dataloader=dataloader,
            dataset_statistics={self.dataset_statistics_key: stats},
        )


__all__ = [
    "OpenVLAOFTHDF5Dataset",
    "OpenVLAOFTHDF5DatasetFactory",
    "OpenVLAOFTHDF5Spec",
]
