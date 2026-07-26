from __future__ import annotations

import json
import random
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path


@dataclass(frozen=True)
class VLASFTLeRobotSpec:
    """Resolved metadata for an OpenVLA-OFT LeRobot SFT dataset."""

    root_dir: str
    codebase_version: str
    num_episodes: int
    num_tasks: int
    num_samples: int
    action_horizon: int
    image_key: str
    action_key: str
    state_key: str
    one_trajectory_sft: bool
    episodes_per_task: int | None
    episode_selection: str
    episode_selection_seed: int | None
    selected_episode_indices: tuple[int, ...]


@dataclass(frozen=True)
class LeRobotEpisode:
    """One episode record resolved from LeRobot v2.1 metadata."""

    episode_index: int
    task_index: int
    task: str
    length: int


@dataclass(frozen=True)
class _LeRobotSample:
    episode_index: int
    frame_index: int


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object in {path}:{line_number}")
            records.append(value)
    return records


def _normalize_bounds_q99(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.clip(2.0 * (values - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)
    return np.where(mask, normalized, values).astype(np.float32, copy=False)


def load_lerobot_v21_episodes(
    root_dir: str | Path,
    *,
    required_features: Sequence[str],
    action_key: str = "action",
    expected_action_dim: int = 7,
) -> tuple[dict[str, Any], list[LeRobotEpisode]]:
    """Validate LeRobot v2.1 metadata and return its ordered episodes."""

    root = Path(root_dir).expanduser().resolve()
    info = _read_json(root / "meta" / "info.json")
    if str(info.get("codebase_version")) != "v2.1":
        raise ValueError(
            f"LeRobot source requires codebase_version='v2.1'; got {info.get('codebase_version')!r}"
        )
    features = dict(info.get("features", {}))
    for key in required_features:
        if key not in features:
            raise KeyError(f"{root}/meta/info.json missing feature {key!r}")
    action_shape = tuple(int(value) for value in features[action_key].get("shape", []))
    if action_shape != (int(expected_action_dim),):
        raise ValueError(
            f"Expected {int(expected_action_dim)}D LIBERO actions, got shape={action_shape}"
        )

    task_records = _read_jsonl(root / "meta" / "tasks.jsonl")
    tasks = {int(record["task_index"]): str(record["task"]) for record in task_records}
    task_indices_by_text: dict[str, list[int]] = defaultdict(list)
    for task_index, task in tasks.items():
        task_indices_by_text[task].append(task_index)

    episodes: list[LeRobotEpisode] = []
    for record in _read_jsonl(root / "meta" / "episodes.jsonl"):
        episode_tasks = list(record.get("tasks", []))
        if len(episode_tasks) != 1:
            raise ValueError(f"Episode {record.get('episode_index')} must contain exactly one task")
        task = str(episode_tasks[0])
        matching_indices = task_indices_by_text.get(task, [])
        if len(matching_indices) != 1:
            raise ValueError(f"Episode task {task!r} does not have one tasks.jsonl entry")
        length = int(record["length"])
        if length < 1:
            raise ValueError(f"Episode {record['episode_index']} has invalid length={length}")
        episodes.append(
            LeRobotEpisode(
                episode_index=int(record["episode_index"]),
                task_index=matching_indices[0],
                task=task,
                length=length,
            )
        )
    return info, sorted(episodes, key=lambda item: item.episode_index)


def select_lerobot_episodes(
    episodes: Sequence[LeRobotEpisode],
    *,
    episodes_per_task: int | None,
    episode_selection: str,
    episode_selection_seed: int,
) -> list[LeRobotEpisode]:
    """Select a deterministic number of LeRobot episodes per task."""

    if episodes_per_task is None:
        return list(episodes)
    count = int(episodes_per_task)
    if count < 1:
        raise ValueError("episodes_per_task must be >= 1 when set")
    if episode_selection not in {"first", "random"}:
        raise ValueError("episode_selection must be 'first' or 'random'")

    by_task: dict[int, list[LeRobotEpisode]] = defaultdict(list)
    for episode in episodes:
        by_task[episode.task_index].append(episode)

    selected: list[LeRobotEpisode] = []
    for task_index in sorted(by_task):
        candidates = sorted(by_task[task_index], key=lambda item: item.episode_index)
        if len(candidates) < count:
            raise ValueError(
                f"Task {task_index} has only {len(candidates)} episodes; cannot select {count}."
            )
        if episode_selection == "first":
            chosen = candidates[:count]
        else:
            rng = random.Random(f"{int(episode_selection_seed)}:{task_index}")
            chosen = sorted(rng.sample(candidates, k=count), key=lambda item: item.episode_index)
        selected.extend(chosen)
    return sorted(selected, key=lambda item: item.episode_index)


def format_lerobot_episode_path(
    root_dir: str | Path,
    info: dict[str, Any],
    template_key: str,
    episode_index: int,
    **kwargs: Any,
) -> Path:
    """Resolve one data or video path using LeRobot metadata templates."""

    values = {
        "episode_chunk": int(episode_index) // int(info["chunks_size"]),
        "episode_index": int(episode_index),
        **kwargs,
    }
    return Path(root_dir).expanduser().resolve() / str(info[template_key]).format(**values)


def decode_lerobot_video(path: str | Path, *, expected_length: int) -> np.ndarray:
    """Decode an AV1 LeRobot video to RGB uint8 frames with strict shape checks."""

    resolved = Path(path).expanduser().resolve()
    frames = np.asarray(iio.imread(resolved, plugin="FFMPEG"), dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[0] != int(expected_length) or frames.shape[-1] != 3:
        raise ValueError(
            f"{resolved} decoded shape {frames.shape}, expected [{int(expected_length)}, H, W, 3]"
        )
    return frames


class VLASFTLeRobotDataset(Dataset):
    """LeRobot v2.1 dataset emitting native OpenVLA-OFT SFT samples.

    LIBERO LeRobot exports already store the standardized gripper convention used
    by OpenVLA-OFT (``0=close, 1=open``). This loader therefore normalizes the
    relative action dimensions but deliberately does not invert the gripper.
    """

    def __init__(
        self,
        root_dir: str | Path,
        processor: Any,
        action_tokenizer: Any,
        dataset_statistics: dict[str, Any],
        action_horizon: int = 8,
        image_key: str = "observation.images.image",
        action_key: str = "action",
        state_key: str = "observation.state",
        use_wrist_image: bool = False,
        use_proprio: bool = False,
        episodes_per_task: int | None = None,
        episode_selection: str = "first",
        episode_selection_seed: int = 0,
        max_episodes: int | None = None,
        max_samples: int | None = None,
        video_cache_size: int = 10,
        dataset_name: str = "libero_goal_no_noops",
    ) -> None:
        ensure_openvla_oft_on_path()
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.vla.constants import IGNORE_INDEX

        self.root_dir = Path(root_dir).expanduser().resolve()
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.dataset_statistics = dataset_statistics
        self.action_horizon = int(action_horizon)
        self.image_key = str(image_key)
        self.action_key = str(action_key)
        self.state_key = str(state_key)
        self.use_wrist_image = bool(use_wrist_image)
        self.use_proprio = bool(use_proprio)
        self.episodes_per_task = None if episodes_per_task is None else int(episodes_per_task)
        self.episode_selection = str(episode_selection)
        self.episode_selection_seed = int(episode_selection_seed)
        self.video_cache_size = int(video_cache_size)
        self.dataset_name = str(dataset_name)
        self.prompt_builder_cls = PurePromptBuilder
        self.ignore_index = int(IGNORE_INDEX)
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be >= 1")
        if self.video_cache_size < 0:
            raise ValueError("video_cache_size must be >= 0")
        if self.use_wrist_image:
            raise ValueError("OpenVLA-OFT mainline SFT does not include a wrist image")
        if self.use_proprio:
            raise ValueError("OpenVLA-OFT mainline SFT does not include VLA-side proprio")

        self.info, episodes = load_lerobot_v21_episodes(
            self.root_dir,
            required_features=(self.image_key, self.action_key, self.state_key),
            action_key=self.action_key,
        )
        episodes = select_lerobot_episodes(
            episodes,
            episodes_per_task=self.episodes_per_task,
            episode_selection=self.episode_selection,
            episode_selection_seed=self.episode_selection_seed,
        )
        if max_episodes is not None:
            episodes = episodes[: int(max_episodes)]
        if not episodes:
            raise RuntimeError(f"No episodes selected under {self.root_dir}")
        self.episodes = {episode.episode_index: episode for episode in episodes}

        self.samples: list[_LeRobotSample] = []
        for episode in episodes:
            for frame_index in range(episode.length):
                self.samples.append(_LeRobotSample(episode.episode_index, frame_index))
                if max_samples is not None and len(self.samples) >= int(max_samples):
                    break
            if max_samples is not None and len(self.samples) >= int(max_samples):
                break

        self._action_cache: dict[int, np.ndarray] = {}
        self._video_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        selected_indices = tuple(episode.episode_index for episode in episodes)
        self._spec = VLASFTLeRobotSpec(
            root_dir=str(self.root_dir),
            codebase_version=str(self.info["codebase_version"]),
            num_episodes=len(episodes),
            num_tasks=len({episode.task_index for episode in episodes}),
            num_samples=len(self.samples),
            action_horizon=self.action_horizon,
            image_key=self.image_key,
            action_key=self.action_key,
            state_key=self.state_key,
            one_trajectory_sft=self.episodes_per_task == 1,
            episodes_per_task=self.episodes_per_task,
            episode_selection=self.episode_selection,
            episode_selection_seed=(
                self.episode_selection_seed if self.episode_selection == "random" else None
            ),
            selected_episode_indices=selected_indices,
        )

    @property
    def data_spec(self) -> VLASFTLeRobotSpec:
        return self._spec

    def __len__(self) -> int:
        return len(self.samples)

    def _format_path(self, template_key: str, episode_index: int, **kwargs: Any) -> Path:
        return format_lerobot_episode_path(
            self.root_dir,
            self.info,
            template_key,
            episode_index,
            **kwargs,
        )

    def _actions(self, episode_index: int) -> np.ndarray:
        cached = self._action_cache.get(episode_index)
        if cached is not None:
            return cached
        path = self._format_path("data_path", episode_index)
        table = pq.read_table(path, columns=[self.action_key])
        actions = np.asarray(table[self.action_key].to_pylist(), dtype=np.float32)
        expected = self.episodes[episode_index].length
        if actions.shape != (expected, 7):
            raise ValueError(
                f"{path}:{self.action_key} has shape {actions.shape}, expected {(expected, 7)}"
            )
        self._action_cache[episode_index] = actions
        return actions

    def _action_chunk(self, episode_index: int, frame_index: int) -> np.ndarray:
        actions = self._actions(episode_index)
        stop = min(frame_index + self.action_horizon, len(actions))
        chunk = actions[frame_index:stop]
        if len(chunk) < self.action_horizon:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], self.action_horizon - len(chunk), axis=0)],
                axis=0,
            )
        # LeRobot LIBERO exports already use 0=close and 1=open. In particular,
        # do not apply the raw-HDF5 gripper inversion used by VLASFTHDF5Dataset.
        return _normalize_bounds_q99(chunk, self.dataset_statistics["action"])

    def _video(self, episode_index: int) -> np.ndarray:
        cached = self._video_cache.pop(episode_index, None)
        if cached is not None:
            self._video_cache[episode_index] = cached
            return cached
        path = self._format_path("video_path", episode_index, video_key=self.image_key)
        expected = self.episodes[episode_index].length
        frames = decode_lerobot_video(path, expected_length=expected)
        if self.video_cache_size > 0:
            self._video_cache[episode_index] = frames
            while len(self._video_cache) > self.video_cache_size:
                self._video_cache.popitem(last=False)
        return frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[int(index)]
        episode = self.episodes[sample.episode_index]
        image = Image.fromarray(self._video(sample.episode_index)[sample.frame_index])
        pixel_values = self.processor.image_processor.apply_transform(image)

        actions = self._action_chunk(sample.episode_index, sample.frame_index)
        current_action_string = self.action_tokenizer(actions[0])
        future_actions_string = "".join(self.action_tokenizer(actions[1:]))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        prompt_builder = self.prompt_builder_cls("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {episode.task}?")
        prompt_builder.add_turn("gpt", action_chunk_string)
        input_ids = self.processor.tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        labels = list(input_ids)
        labels[: -(action_chunk_len + 1)] = [self.ignore_index] * (
            len(labels) - (action_chunk_len + 1)
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "actions": actions,
            "dataset_name": self.dataset_name,
        }


class VLASFTLeRobotDatasetFactory:
    """Build an OpenVLA-OFT dataloader from a local LeRobot v2.1 dataset."""

    def __init__(
        self,
        root_dir: str | Path,
        dataset_statistics_path: str | Path | None = None,
        dataset_statistics_key: str = "libero_goal_no_noops",
        action_horizon: int = 8,
        image_key: str = "observation.images.image",
        action_key: str = "action",
        state_key: str = "observation.state",
        use_wrist_image: bool = False,
        use_proprio: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        episodes_per_task: int | None = None,
        episode_selection: str = "first",
        episode_selection_seed: int = 0,
        max_episodes: int | None = None,
        max_samples: int | None = None,
        video_cache_size: int = 10,
        dataset_name: str = "libero_goal_no_noops",
        **unexpected_kwargs: Any,
    ) -> None:
        if unexpected_kwargs:
            raise TypeError(
                "VLASFTLeRobotDatasetFactory received unsupported arguments: "
                f"{sorted(unexpected_kwargs)!r}"
            )
        self.root_dir = str(Path(root_dir).expanduser().resolve())
        self.dataset_statistics_path = (
            None
            if dataset_statistics_path is None
            else str(Path(dataset_statistics_path).expanduser().resolve())
        )
        self.dataset_statistics_key = str(dataset_statistics_key)
        self.action_horizon = int(action_horizon)
        self.image_key = str(image_key)
        self.action_key = str(action_key)
        self.state_key = str(state_key)
        self.use_wrist_image = bool(use_wrist_image)
        self.use_proprio = bool(use_proprio)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.episodes_per_task = None if episodes_per_task is None else int(episodes_per_task)
        self.episode_selection = str(episode_selection)
        self.episode_selection_seed = int(episode_selection_seed)
        self.max_episodes = max_episodes
        self.max_samples = max_samples
        self.video_cache_size = int(video_cache_size)
        self.dataset_name = str(dataset_name)

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

        from dreamervla.dataset.vla_sft_rlds_dataset import VLASFTRLDSDatasetBundle

        stats = self._load_statistics(policy)
        action_tokenizer = ActionTokenizer(policy.processor.tokenizer)
        dataset = VLASFTLeRobotDataset(
            root_dir=self.root_dir,
            processor=policy.processor,
            action_tokenizer=action_tokenizer,
            dataset_statistics=stats,
            action_horizon=self.action_horizon,
            image_key=self.image_key,
            action_key=self.action_key,
            state_key=self.state_key,
            use_wrist_image=self.use_wrist_image,
            use_proprio=self.use_proprio,
            episodes_per_task=self.episodes_per_task,
            episode_selection=self.episode_selection,
            episode_selection_seed=self.episode_selection_seed,
            max_episodes=self.max_episodes,
            max_samples=self.max_samples,
            video_cache_size=self.video_cache_size,
            dataset_name=self.dataset_name,
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
        return VLASFTRLDSDatasetBundle(
            dataset=dataset,
            dataloader=dataloader,
            dataset_statistics={self.dataset_statistics_key: stats},
        )


__all__ = [
    "LeRobotEpisode",
    "VLASFTLeRobotDataset",
    "VLASFTLeRobotDatasetFactory",
    "VLASFTLeRobotSpec",
    "decode_lerobot_video",
    "format_lerobot_episode_path",
    "load_lerobot_v21_episodes",
    "select_lerobot_episodes",
]
