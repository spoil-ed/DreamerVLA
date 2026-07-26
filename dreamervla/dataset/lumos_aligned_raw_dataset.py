"""Raw-image WMPO classifier datasets with lazy HDF5 reads.

These datasets preserve the trajectory-level split and sampling protocol of
``lumos_aligned_latent_dataset`` while returning only the selected raw frames.
The runner embeds those frames with a frozen VLA; no embedding sidecar is
created or required.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from dreamervla.dataset.lumos_aligned_latent_dataset import (
    _partition_demo_pairs,
    _rank_shard_demo_pairs,
)


@dataclass(frozen=True)
class _RawDemoRecord:
    path: Path
    demo_key: str
    finish_step: int
    complete: bool
    eid: str
    task_description: str


@dataclass(frozen=True)
class _ValSlot:
    demo_idx: int
    end_idx: int
    label: int
    is_end_window: bool


def _discover_raw_pairs(directory: str | Path) -> list[tuple[Path, Path, str]]:
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"raw rollout directory does not exist: {root}")
    pairs: list[tuple[Path, Path, str]] = []
    for path in sorted(root.glob("*.hdf5")):
        try:
            with h5py.File(path, "r") as handle:
                data = handle.get("data")
                if not isinstance(data, h5py.Group):
                    continue
                pairs.extend((path, path, f"data/{key}") for key in sorted(data.keys()))
        except OSError:
            # Interrupted collection shards live under .corrupt, but tolerate
            # an incomplete file in the source directory during discovery too.
            continue
    if not pairs:
        raise RuntimeError(f"no readable rollout demos found under {root}")
    return pairs


def _load_metadata(
    pairs: Sequence[tuple[Path, Path, str]],
    *,
    min_steps: int,
) -> list[_RawDemoRecord]:
    records: list[_RawDemoRecord] = []
    for raw_path, _unused, demo_key in pairs:
        with h5py.File(raw_path, "r") as handle:
            demo = handle[demo_key]
            reward_key = "sparse_rewards" if "sparse_rewards" in demo else "rewards"
            rewards = np.asarray(demo[reward_key][...])
            length = int(rewards.shape[0])
            dones = np.asarray(demo["dones"][...]) if "dones" in demo else None
            done_indices = np.flatnonzero(dones) if dones is not None else np.empty(0)
            finish_step = int(done_indices[0] + 1) if done_indices.size else length
            finish_step = min(finish_step, length)
            if finish_step < int(min_steps):
                continue
            task_description = str(
                demo.attrs.get(
                    "task_description",
                    demo.attrs.get("task_name", raw_path.stem.replace("_", " ")),
                )
            )
            records.append(
                _RawDemoRecord(
                    path=raw_path,
                    demo_key=demo_key,
                    finish_step=finish_step,
                    complete=bool(rewards[:finish_step].sum() > 0),
                    eid=f"{raw_path.stem}/{demo_key.rsplit('/', 1)[-1]}",
                    task_description=task_description,
                )
            )
    return records


class _RawDatasetMixin:
    online_vla_encoding = True

    W: int
    S: int
    K: int
    window_env: int
    chunk_pool: str
    image_key: str
    proprio_keys: tuple[str, ...]
    _demos: list[_RawDemoRecord]

    def _chunk_indices(self, start: int, end: int) -> np.ndarray:
        if self.chunk_pool == "last":
            return np.arange(start + self.K - 1, end, self.K, dtype=np.int64)
        if self.chunk_pool == "first":
            return np.arange(start, end, self.K, dtype=np.int64)
        raise ValueError(
            "raw online VLA encoding supports chunk_pool=first|last; "
            "pixel-space mean is not equivalent to latent-space mean"
        )

    def _read_indices(
        self,
        rec: _RawDemoRecord,
        indices: np.ndarray,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        with h5py.File(rec.path, "r") as handle:
            obs = handle[rec.demo_key]["obs"]
            images = np.asarray(obs[self.image_key][indices], dtype=np.uint8)
            proprio = np.concatenate(
                [
                    np.asarray(obs[key][indices], dtype=np.float32).reshape(len(indices), -1)
                    for key in self.proprio_keys
                ],
                axis=-1,
            )
        return torch.from_numpy(np.ascontiguousarray(images)), {
            "proprio": torch.from_numpy(np.ascontiguousarray(proprio)),
            "task_description": rec.task_description,
        }

    def _window_item(
        self,
        rec: _RawDemoRecord,
        *,
        end: int,
        label: int,
    ) -> tuple[torch.Tensor, int, dict[str, Any]]:
        indices = self._chunk_indices(int(end) - self.window_env, int(end))
        images, extra = self._read_indices(rec, indices)
        return images, int(label), extra

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, int, dict[str, Any]]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return (
            torch.stack([item[0] for item in batch]),
            torch.tensor([item[1] for item in batch], dtype=torch.long),
            {
                "proprio": torch.stack([item[2]["proprio"] for item in batch]),
                "task_description": [str(item[2]["task_description"]) for item in batch],
            },
        )

    def summary(self) -> dict[str, int | str | bool]:
        successes = sum(int(record.complete) for record in self._demos)
        return {
            "num_demos": len(self._demos),
            "num_success_demos": successes,
            "num_failure_demos": len(self._demos) - successes,
            "window": self.W,
            "stride": self.S,
            "chunk_subsample": self.K,
            "chunk_pool": self.chunk_pool,
            "input_storage": "raw_images",
            "online_vla_encoding": True,
        }


class LumosAlignedRawTrainDataset(_RawDatasetMixin, IterableDataset):
    """Infinite balanced WMPO stream backed by lazily read raw rollout frames."""

    def __init__(
        self,
        success_dir_raw: str | Path,
        success_dir_hidden: str | Path | None = None,
        failure_dir_raw: str | Path | None = None,
        failure_dir_hidden: str | Path | None = None,
        window: int = 8,
        stride: int = 8,
        seed: int = 0,
        verbose: bool = True,
        chunk_subsample: int = 8,
        chunk_pool: str = "last",
        proprio_keys: Sequence[str] | None = None,
        lang_emb_dir: str | Path | None = None,
        lang_emb_key: str = "lang_emb",
        sampling_protocol: str = "wmpo",
        balance_batches: bool = True,
        demo_split: str = "train",
        val_fraction: float = 0.2,
        split_seed: int = 0,
        stratify_by_complete: bool = True,
        distributed_rank: int = 0,
        distributed_world_size: int = 1,
        image_key: str = "agentview_rgb",
    ) -> None:
        super().__init__()
        del success_dir_hidden, failure_dir_hidden, lang_emb_dir, lang_emb_key
        if str(sampling_protocol) != "wmpo" or not bool(balance_batches):
            raise ValueError("raw classifier training currently requires balanced WMPO sampling")
        if chunk_pool not in {"first", "last"}:
            raise ValueError("raw classifier chunk_pool must be first or last")
        self.W, self.S, self.K = int(window), int(stride), int(chunk_subsample)
        self.window_env = self.W * self.K
        self.chunk_pool = str(chunk_pool)
        self.image_key = str(image_key)
        self.proprio_keys = tuple(proprio_keys or ("ee_pos", "ee_ori", "gripper_states"))
        self.seed = int(seed)
        self.sampling_protocol = "wmpo"
        self.balance_batches = True
        self.demo_split = str(demo_split)
        self.pre_sharded = int(distributed_world_size) > 1

        pairs = _discover_raw_pairs(success_dir_raw)
        if failure_dir_raw is not None:
            pairs.extend(_discover_raw_pairs(failure_dir_raw))
        pairs = _partition_demo_pairs(
            pairs,
            split=demo_split,
            val_fraction=val_fraction,
            split_seed=split_seed,
            stratify_by_complete=stratify_by_complete,
        )
        pairs = _rank_shard_demo_pairs(
            pairs,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
        )
        self._demos = _load_metadata(pairs, min_steps=self.window_env)
        self._positive_ids = [idx for idx, record in enumerate(self._demos) if record.complete]
        self._negative_slots = self._build_negative_slots()
        if not self._positive_ids or not self._negative_slots:
            raise RuntimeError(
                "each DDP raw-data shard needs successful trajectories and negative clips; "
                f"rank={distributed_rank} successes={len(self._positive_ids)} "
                f"negative_clips={len(self._negative_slots)}"
            )
        self._epoch_windows = 2 * len(self._positive_ids)
        if verbose:
            print(f"[lumos-raw:train] {self.summary()}", flush=True)

    def _build_negative_slots(self) -> list[tuple[int, int]]:
        slots: list[tuple[int, int]] = []
        for idx, rec in enumerate(self._demos):
            max_end = rec.finish_step - self.window_env if rec.complete else rec.finish_step
            slots.extend((idx, end) for end in range(self.window_env, max_end + 1, self.S))
        return slots

    def __len__(self) -> int:
        return self._epoch_windows

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int, dict[str, Any]]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        rng = np.random.default_rng(self.seed + 1000 * (worker_id + 1))
        while True:
            pos_idx = int(rng.choice(self._positive_ids))
            pos = self._demos[pos_idx]
            yield self._window_item(pos, end=pos.finish_step, label=1)
            neg_idx, neg_end = self._negative_slots[int(rng.integers(len(self._negative_slots)))]
            yield self._window_item(self._demos[neg_idx], end=neg_end, label=0)


class LumosAlignedRawValDataset(_RawDatasetMixin, Dataset):
    """Deterministic held-out raw windows and trajectory-level evaluation data."""

    def __init__(
        self,
        success_dir_raw: str | Path,
        success_dir_hidden: str | Path | None = None,
        failure_dir_raw: str | Path | None = None,
        failure_dir_hidden: str | Path | None = None,
        window: int = 8,
        stride: int = 8,
        verbose: bool = True,
        chunk_subsample: int = 8,
        chunk_pool: str = "last",
        proprio_keys: Sequence[str] | None = None,
        lang_emb_dir: str | Path | None = None,
        lang_emb_key: str = "lang_emb",
        sampling_protocol: str = "wmpo",
        demo_split: str = "val",
        val_fraction: float = 0.2,
        split_seed: int = 0,
        stratify_by_complete: bool = True,
        distributed_rank: int = 0,
        distributed_world_size: int = 1,
        image_key: str = "agentview_rgb",
    ) -> None:
        super().__init__()
        del success_dir_hidden, failure_dir_hidden, lang_emb_dir, lang_emb_key
        if str(sampling_protocol) != "wmpo":
            raise ValueError("raw classifier validation currently requires WMPO sampling")
        if chunk_pool not in {"first", "last"}:
            raise ValueError("raw classifier chunk_pool must be first or last")
        self.W, self.S, self.K = int(window), int(stride), int(chunk_subsample)
        self.window_env = self.W * self.K
        self.chunk_pool = str(chunk_pool)
        self.image_key = str(image_key)
        self.proprio_keys = tuple(proprio_keys or ("ee_pos", "ee_ori", "gripper_states"))
        self.sampling_protocol = "wmpo"
        self.demo_split = str(demo_split)
        self.pre_sharded = int(distributed_world_size) > 1

        pairs = _discover_raw_pairs(success_dir_raw)
        if failure_dir_raw is not None:
            pairs.extend(_discover_raw_pairs(failure_dir_raw))
        pairs = _partition_demo_pairs(
            pairs,
            split=demo_split,
            val_fraction=val_fraction,
            split_seed=split_seed,
            stratify_by_complete=stratify_by_complete,
        )
        pairs = _rank_shard_demo_pairs(
            pairs,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
        )
        self._demos = _load_metadata(pairs, min_steps=self.window_env)
        self._slots: list[_ValSlot] = []
        for idx, rec in enumerate(self._demos):
            self._slots.append(_ValSlot(idx, rec.finish_step, int(rec.complete), True))
            max_end = (
                rec.finish_step - self.window_env if rec.complete else rec.finish_step - self.S
            )
            self._slots.extend(
                _ValSlot(idx, end, 0, False) for end in range(self.window_env, max_end + 1, self.S)
            )
        if verbose:
            print(f"[lumos-raw:val] {self.summary()}", flush=True)

    def __len__(self) -> int:
        return len(self._slots)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        slot = self._slots[index]
        images, label, extra = self._window_item(
            self._demos[slot.demo_idx],
            end=slot.end_idx,
            label=slot.label,
        )
        return images, label, extra

    def trajectories(
        self,
    ) -> Iterator[tuple[np.ndarray, bool, int, str, dict[str, Any]]]:
        """Yield already chunk-sampled raw frames, loaded one trajectory at a time."""
        for rec in self._demos:
            usable = (rec.finish_step // self.K) * self.K
            indices = self._chunk_indices(0, usable)
            images, extra = self._read_indices(rec, indices)
            yield (
                images.numpy(),
                rec.complete,
                len(indices),
                rec.eid,
                {
                    "proprio": extra["proprio"].numpy(),
                    "task_description": rec.task_description,
                    "pre_pooled": True,
                },
            )


__all__ = ["LumosAlignedRawTrainDataset", "LumosAlignedRawValDataset"]
