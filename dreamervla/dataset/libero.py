"""Native LIBERO feature mapping and config-selected loader assembly."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dreamervla.dataset.base.base_dataloader import DatasetLoaderBundle
from dreamervla.dataset.base.hdf5_dataloader import HDF5ActionChunkDataset
from dreamervla.dataset.base.lerobot_v3_dataloader import LeRobotV3DataLoader
from dreamervla.utils.integrations.openvla_oft_imports import ensure_openvla_oft_on_path


class LiberoDataset(LeRobotV3DataLoader):
    """Map native LIBERO v3 frames and action chunks into policy input fields.

    Source actions/state retain their independent dimensions. This adapter
    performs no pose conversion, image rotation or normalization; those belong
    to the selected model's input transforms and checkpoint assets.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        sequence_length: int,
        image_key: str,
        wrist_image_key: str,
        action_key: str,
        state_key: str,
        **kwargs: Any,
    ) -> None:
        self.image_key = image_key
        self.wrist_image_key = wrist_image_key
        self.action_key = action_key
        self.state_key = state_key
        super().__init__(
            dataset_dir,
            sequence_length=sequence_length,
            feature_keys=(action_key, state_key),
            camera_keys=(image_key, wrist_image_key),
            image_offsets=(0,),
            **kwargs,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        return {
            "image": sample[self.image_key][0],
            "wrist_image": sample[self.wrist_image_key][0],
            "state": sample[self.state_key][0],
            "actions": sample[self.action_key],
            "prompt": sample["prompt"],
            "action_mask": sample["time_mask"],
            "episode_index": sample["episode_index"],
            "frame_index": sample["frame_index"],
        }


@dataclass
class LeRobotV3LIBERODataLoaderFactory:
    """Bridge a Hydra-instantiated local dataset to the π0.5 batch contract."""

    dataset: Any
    repo_id: str
    model_path: str
    assets_path: str
    normalization_asset_id: str
    config_name: str
    batch_size: int
    action_horizon: int
    num_workers: int
    seed: int
    shuffle: bool = True

    def build(self, *, world_size: int, rank: int, eval_dataset: bool = False) -> Any:
        """Build batches using model-owned transforms and native dataset values."""
        from dreamervla.models.embodiment.pi05.sft_data import build_local_sft_loader

        if self.dataset.sequence_length != self.action_horizon:
            raise ValueError("LIBERO sequence_length must match the policy action_horizon")
        return build_local_sft_loader(
            dataset=self.dataset,
            repo_id=self.repo_id,
            model_path=self.model_path,
            assets_path=self.assets_path,
            normalization_asset_id=self.normalization_asset_id,
            config_name=self.config_name,
            batch_size=self.batch_size,
            action_horizon=self.action_horizon,
            num_workers=self.num_workers,
            seed=self.seed,
            shuffle=self.shuffle and not eval_dataset,
            world_size=world_size,
            rank=rank,
        )


def _task_from_path(path: str | Path) -> str:
    stem = Path(path).name
    if stem.endswith("_demo.hdf5"):
        stem = stem[: -len("_demo.hdf5")]
    else:
        stem = Path(stem).stem
    return stem.replace("_", " ").strip().lower()


class VLASFTHDF5Dataset(HDF5ActionChunkDataset):
    """LIBERO HDF5 reader with the OpenVLA-OFT model input transform."""

    def __init__(
        self,
        hdf5_dir: str | Path,
        processor: Any,
        action_tokenizer: Any,
        dataset_statistics: dict[str, Any],
        action_horizon: int = 8,
        image_keys: Sequence[str] = ("agentview_rgb",),
        use_wrist_image: bool = False,
        use_proprio: bool = False,
        **kwargs: Any,
    ) -> None:
        if tuple(image_keys) != ("agentview_rgb",):
            raise ValueError("OpenVLA-OFT mainline SFT requires image_keys=('agentview_rgb',)")
        if use_wrist_image:
            raise ValueError("OpenVLA-OFT mainline SFT does not include a wrist image")
        if use_proprio:
            raise ValueError("OpenVLA-OFT mainline SFT does not include VLA-side proprio")
        from dreamervla.models.embodiment.openvla_oft.sft_data import OpenVLAHDF5Transform

        self.transform = OpenVLAHDF5Transform(processor, action_tokenizer, dataset_statistics)
        super().__init__(hdf5_dir, action_horizon=action_horizon, image_keys=image_keys, **kwargs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = super().__getitem__(index)
        return self.transform(
            raw, task=_task_from_path(raw["file_path"]), image_key=self.image_keys[0]
        )


class VLASFTHDF5DatasetFactory:
    """Hydra bridge for LIBERO HDF5 supervision of OpenVLA-OFT."""

    def __init__(
        self,
        hdf5_dir: str | Path,
        dataset_statistics_path: str | Path | None = None,
        dataset_statistics_key: str = "libero_goal_no_noops",
        action_horizon: int = 8,
        image_keys: Sequence[str] = ("agentview_rgb",),
        use_wrist_image: bool = False,
        use_proprio: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        max_files: int | None = None,
        max_demos_per_file: int | None = None,
        demos_per_task: int | None = None,
        demo_selection_seed: int = 0,
        max_samples: int | None = None,
        **unexpected_kwargs: Any,
    ) -> None:
        if unexpected_kwargs:
            raise TypeError(
                "VLASFTHDF5DatasetFactory received unsupported arguments: "
                f"{sorted(unexpected_kwargs)!r}"
            )
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
        if self.image_keys != ("agentview_rgb",):
            raise ValueError("OpenVLA-OFT mainline SFT requires image_keys=('agentview_rgb',)")
        if self.use_wrist_image:
            raise ValueError("OpenVLA-OFT mainline SFT does not include a wrist image")
        if self.use_proprio:
            raise ValueError("OpenVLA-OFT mainline SFT does not include VLA-side proprio")
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

    def build(self, policy: Any, *, train: bool = True) -> DatasetLoaderBundle:
        """Build a policy-transformed HDF5 dataset and distributed loader."""
        ensure_openvla_oft_on_path()
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction
        from prismatic.vla.action_tokenizer import ActionTokenizer

        stats = self._load_statistics(policy)
        action_tokenizer = ActionTokenizer(policy.processor.tokenizer)
        dataset = VLASFTHDF5Dataset(
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
        return DatasetLoaderBundle(
            dataset=dataset,
            dataloader=dataloader,
            dataset_statistics={self.dataset_statistics_key: stats},
        )


def read_libero_proprio(demo: h5py.Group) -> np.ndarray:
    """Return the canonical LIBERO proprio sidecar as ``[T,8]`` float32."""

    obs = demo["obs"]
    return np.concatenate(
        [
            np.asarray(obs["ee_pos"][...], dtype=np.float32),
            np.asarray(obs["ee_ori"][...], dtype=np.float32),
            np.asarray(obs["gripper_states"][...], dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
