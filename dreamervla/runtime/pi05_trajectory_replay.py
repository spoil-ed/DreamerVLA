"""Streaming RGB replay with trajectory-local frozen π0.5 prefix caching."""

from __future__ import annotations

import bisect
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from dreamervla.models.embodiment.pi05.policy import _libero_eval_image


@dataclass(frozen=True)
class _TrajectoryRef:
    path: Path
    demo_key: str
    length: int
    task_id: int
    episode_id: int
    task_description: str
    success: bool
    collection_index: int

    @property
    def identity(self) -> tuple[str, str]:
        return str(self.path), self.demo_key


class Pi05TrajectoryReplay:
    """Stream full trajectory windows without materializing a latent replay.

    Whole trajectories are deterministically assigned to one DDP rank. A rank
    encodes one trajectory at a time with a frozen π0.5 producer, consumes all
    of its sequence windows, then releases that latent cache before moving on.
    The shorter DDP shards repeat only their final padding slots so every rank
    executes the same number of optimizer updates.
    """

    def __init__(
        self,
        *,
        data_dir: str | Path,
        sequence_length: int,
        encoder: Any,
        encode_batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        task_ids: tuple[int, ...] | None = None,
        rotate_images_180: bool = True,
        base_image_key: str = "agentview_rgb",
        wrist_image_key: str = "eye_in_hand_rgb",
        max_episodes_per_task: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.sequence_length = int(sequence_length)
        self.encoder = encoder
        self.encode_batch_size = int(encode_batch_size)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self.rotate_images_180 = bool(rotate_images_180)
        self.base_image_key = str(base_image_key)
        self.wrist_image_key = str(wrist_image_key)
        if self.sequence_length <= 0 or self.encode_batch_size <= 0:
            raise ValueError("sequence_length and encode_batch_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} is outside world_size {self.world_size}")

        refs = _scan_trajectories(
            self.data_dir,
            sequence_length=self.sequence_length,
            task_ids=task_ids,
            max_episodes_per_task=max_episodes_per_task,
        )
        if not refs:
            raise RuntimeError(
                f"RGB replay found no trajectory of length >= {self.sequence_length} "
                f"under {self.data_dir}"
            )
        self._all_refs = refs
        assignments, loads = _assign_trajectories(refs, self.world_size, self.sequence_length)
        self._refs = assignments[self.rank]
        self._rank_window_loads = tuple(loads)
        self._raw_window_count = sum(_window_count(ref, self.sequence_length) for ref in refs)
        self._effective_window_count = max(loads) * self.world_size
        self._num_transitions = sum(ref.length for ref in refs)
        self._task_episode_counts = Counter(ref.task_id for ref in refs)

        self._epoch = 0
        self._position = 0
        self._batch_size: int | None = None
        self._steps_per_epoch: int | None = None
        self._ordered_refs: list[_TrajectoryRef] = []
        self._cumulative_windows: list[int] = []
        self._cache_identity: tuple[str, str] | None = None
        self._cache: dict[str, np.ndarray] | None = None
        self._reset_epoch_order()

    @property
    def num_transitions(self) -> int:
        return int(self._num_transitions)

    @property
    def raw_window_count(self) -> int:
        return int(self._raw_window_count)

    def sampleable_window_count(self) -> int:
        """Return DDP-effective slots, including the minimum rank padding."""

        return int(self._effective_window_count)

    def task_episode_counts(self) -> Counter[int]:
        return Counter(self._task_episode_counts)

    def classifier_window_count(self, *, window: int, chunk_size: int) -> int:
        del window, chunk_size
        return 0

    def seek_update(self, update: int, *, batch_size: int) -> None:
        """Seek to a deterministic epoch/update boundary for checkpoint resume."""

        self._configure_batch_size(batch_size)
        assert self._steps_per_epoch is not None
        update = max(0, int(update))
        self._epoch = update // self._steps_per_epoch
        self._position = (update % self._steps_per_epoch) * int(batch_size)
        self._drop_cache()
        self._reset_epoch_order()

    def sample(
        self,
        batch_size: int,
        *,
        staleness_threshold: int | None = None,
        include_images: bool = False,
    ) -> dict[str, torch.Tensor]:
        del staleness_threshold, include_images
        self._configure_batch_size(batch_size)
        assert self._steps_per_epoch is not None
        epoch_capacity = self._steps_per_epoch * int(batch_size)
        local_windows = self._cumulative_windows[-1]
        slots = [int((self._position + offset) % local_windows) for offset in range(batch_size)]
        samples = [self._window_at(slot) for slot in slots]
        self._position += int(batch_size)
        if self._position >= epoch_capacity:
            self._epoch += 1
            self._position = 0
            self._drop_cache()
            self._reset_epoch_order()
        return _stack_samples(samples, self.sequence_length, source_rank=self.rank)

    def _configure_batch_size(self, batch_size: int) -> None:
        value = int(batch_size)
        if value <= 0:
            raise ValueError("batch_size must be positive")
        if self._batch_size is not None and self._batch_size != value:
            raise ValueError(
                f"streaming replay batch size changed from {self._batch_size} to {value}"
            )
        self._batch_size = value
        self._steps_per_epoch = max(
            1,
            (max(self._rank_window_loads) + value - 1) // value,
        )

    def _reset_epoch_order(self) -> None:
        self._ordered_refs = list(self._refs)
        random.Random(self.seed + self._epoch).shuffle(self._ordered_refs)
        cumulative: list[int] = []
        total = 0
        for ref in self._ordered_refs:
            total += _window_count(ref, self.sequence_length)
            cumulative.append(total)
        if total <= 0:
            raise RuntimeError(f"DDP rank {self.rank} received no sampleable RGB trajectory")
        self._cumulative_windows = cumulative

    def _window_at(self, slot: int) -> dict[str, Any]:
        ref_index = bisect.bisect_right(self._cumulative_windows, int(slot))
        previous = self._cumulative_windows[ref_index - 1] if ref_index > 0 else 0
        start = int(slot) - int(previous)
        ref = self._ordered_refs[ref_index]
        cache = self._trajectory_cache(ref)
        stop = start + self.sequence_length
        return {
            "obs_embedding": cache["obs_embedding"][start:stop],
            "current_actions": cache["actions"][start:stop],
            "rewards": cache["rewards"][start:stop],
            "dones": cache["dones"][start:stop],
            "proprio": cache["proprio"][start:stop],
            "task_id": ref.task_id,
            "episode_id": ref.episode_id,
            "collection_index": ref.collection_index,
            "episode_length": ref.length,
            "start": start,
            "success": ref.success,
        }

    def _trajectory_cache(self, ref: _TrajectoryRef) -> dict[str, np.ndarray]:
        if self._cache_identity == ref.identity and self._cache is not None:
            return self._cache
        self._drop_cache()
        with h5py.File(ref.path, "r") as handle:
            demo = handle["data"][ref.demo_key]
            obs = demo["obs"]
            base_images = np.asarray(obs[self.base_image_key][...], dtype=np.uint8)
            wrist_images = np.asarray(obs[self.wrist_image_key][...], dtype=np.uint8)
            proprio = np.concatenate(
                [
                    np.asarray(obs["ee_pos"][...], dtype=np.float32),
                    np.asarray(obs["ee_ori"][...], dtype=np.float32),
                    np.asarray(obs["gripper_states"][...], dtype=np.float32),
                ],
                axis=-1,
            )
            actions = np.asarray(demo["actions"][...], dtype=np.float32)
            rewards = np.asarray(demo["sparse_rewards"][...], dtype=np.float32)
            dones = np.asarray(demo["dones"][...], dtype=np.float32)

        encoded: list[np.ndarray] = []
        for begin in range(0, ref.length, self.encode_batch_size):
            end = min(ref.length, begin + self.encode_batch_size)
            raw = [
                {
                    "observation/image": _libero_eval_image(
                        base_images[index], rotate_180=self.rotate_images_180
                    ),
                    "observation/wrist_image": _libero_eval_image(
                        wrist_images[index], rotate_180=self.rotate_images_180
                    ),
                    "observation/state": proprio[index],
                    "prompt": ref.task_description,
                }
                for index in range(begin, end)
            ]
            prefix = self.encoder.encode_raw_observation_prefix_batch(raw)
            encoded.append(prefix.detach().to(dtype=torch.float16, device="cpu").numpy())
        self._cache_identity = ref.identity
        self._cache = {
            "obs_embedding": np.concatenate(encoded, axis=0),
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "proprio": proprio,
        }
        return self._cache

    def _drop_cache(self) -> None:
        self._cache_identity = None
        self._cache = None


def _window_count(ref: _TrajectoryRef, sequence_length: int) -> int:
    return max(0, int(ref.length) - int(sequence_length) + 1)


def _assign_trajectories(
    refs: list[_TrajectoryRef],
    world_size: int,
    sequence_length: int,
) -> tuple[list[list[_TrajectoryRef]], list[int]]:
    assignments: list[list[_TrajectoryRef]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    ordered = sorted(
        refs,
        key=lambda ref: (-_window_count(ref, sequence_length), str(ref.path), ref.demo_key),
    )
    for ref in ordered:
        rank = min(range(world_size), key=lambda index: (loads[index], index))
        assignments[rank].append(ref)
        loads[rank] += _window_count(ref, sequence_length)
    if any(not shard for shard in assignments):
        raise RuntimeError(
            f"RGB replay has {len(refs)} trajectories for {world_size} DDP ranks"
        )
    return assignments, loads


def _scan_trajectories(
    data_dir: Path,
    *,
    sequence_length: int,
    task_ids: tuple[int, ...] | None,
    max_episodes_per_task: int | None,
) -> list[_TrajectoryRef]:
    shards = sorted(data_dir.glob("*.hdf5"))
    if not shards:
        raise FileNotFoundError(f"no reward HDF5 shards under {data_dir}")
    requested = set(task_ids) if task_ids is not None else None
    cap = int(max_episodes_per_task) if max_episodes_per_task is not None else None
    counts: Counter[int] = Counter()
    refs: list[_TrajectoryRef] = []
    for path in shards:
        with h5py.File(path, "r") as handle:
            data = handle.get("data")
            if data is None:
                continue
            for demo_key in sorted(data.keys()):
                demo = data[demo_key]
                if not bool(demo.attrs.get("complete", True)):
                    continue
                task_id = int(demo.attrs.get("task_id", -1))
                if task_id < 0:
                    raise ValueError(f"{path.name}/{demo_key} has no task_id attr")
                if requested is not None and task_id not in requested:
                    continue
                if cap is not None and counts[task_id] >= cap:
                    continue
                length = int(demo.attrs.get("num_samples", demo["actions"].shape[0]))
                if length < sequence_length:
                    continue
                for key in ("actions", "dones", "sparse_rewards"):
                    if key not in demo or int(demo[key].shape[0]) < length:
                        raise ValueError(f"{path.name}/{demo_key} has invalid {key}")
                obs = demo.get("obs")
                required_obs = (
                    "agentview_rgb",
                    "eye_in_hand_rgb",
                    "ee_pos",
                    "ee_ori",
                    "gripper_states",
                )
                if obs is None or any(
                    key not in obs or int(obs[key].shape[0]) < length for key in required_obs
                ):
                    raise ValueError(f"{path.name}/{demo_key} has incomplete RGB/proprio data")
                sparse = np.asarray(demo["sparse_rewards"][...])
                refs.append(
                    _TrajectoryRef(
                        path=path,
                        demo_key=str(demo_key),
                        length=length,
                        task_id=task_id,
                        episode_id=int(demo.attrs.get("episode_id", len(refs))),
                        task_description=str(
                            demo.attrs.get(
                                "task_description",
                                demo.attrs.get("task_name", f"LIBERO task {task_id}"),
                            )
                        ),
                        success=bool(demo.attrs.get("success", np.any(sparse > 0.5))),
                        collection_index=len(refs),
                    )
                )
                counts[task_id] += 1
    return refs


def _stack_samples(
    samples: list[dict[str, Any]],
    sequence_length: int,
    *,
    source_rank: int,
) -> dict[str, torch.Tensor]:
    current_actions = np.stack([item["current_actions"] for item in samples], axis=0)
    actions = np.zeros_like(current_actions, dtype=np.float32)
    actions[:, 1:] = current_actions[:, :-1]
    rewards = np.stack([item["rewards"] for item in samples], axis=0).astype(np.float32)
    dones = np.stack([item["dones"] for item in samples], axis=0).astype(np.float32)
    is_first = np.zeros((len(samples), sequence_length), dtype=np.bool_)
    is_first[:, 0] = True
    return {
        "obs_embedding": torch.from_numpy(
            np.stack([item["obs_embedding"] for item in samples], axis=0)
        ),
        "actions": torch.from_numpy(actions),
        "current_actions": torch.from_numpy(current_actions.astype(np.float32, copy=False)),
        "rewards": torch.from_numpy(rewards),
        "dones": torch.from_numpy(dones),
        "is_terminal": torch.from_numpy((rewards > 0.5).astype(np.float32)),
        "is_last": torch.from_numpy(dones.copy()),
        "is_first": torch.from_numpy(is_first),
        "proprio": torch.from_numpy(
            np.stack([item["proprio"] for item in samples], axis=0).astype(np.float32)
        ),
        "task_ids": torch.tensor([item["task_id"] for item in samples], dtype=torch.long),
        "episode_success": torch.tensor(
            [item["success"] for item in samples], dtype=torch.bool
        ),
        "start_indices": torch.tensor([item["start"] for item in samples], dtype=torch.long),
        "episode_ids": torch.tensor(
            [item["episode_id"] for item in samples], dtype=torch.long
        ),
        "collection_indices": torch.tensor(
            [item["collection_index"] for item in samples], dtype=torch.long
        ),
        "episode_lengths": torch.tensor(
            [item["episode_length"] for item in samples], dtype=torch.long
        ),
        "source_ranks": torch.full((len(samples),), int(source_rank), dtype=torch.long),
        "replay_source_ids": torch.zeros((len(samples),), dtype=torch.long),
    }


__all__ = ["Pi05TrajectoryReplay"]
