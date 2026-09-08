# SPDX-License-Identifier: Apache-2.0
"""Local LeRobot v3 window reader.

The bounded Parquet cache, file-local episode offsets and keyframe-seek video
decoding are adapted from OpenWAM's LeRobot v3 reader and video_io helpers
(Apache-2.0). This implementation has no OpenWAM runtime dependency. It keeps
native feature dimensions and camera views, and never substitutes another
sample when reading fails.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dreamervla.dataset.base.base_dataloader import BaseDataset

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _read_table(path: str, columns: tuple[str, ...]) -> Any:
    """Bound decoded shard memory per DataLoader process."""
    import pyarrow.parquet as pq

    # ParquetFile avoids treating dotted feature names as nested dataset paths.
    return pq.ParquetFile(path).read(columns=list(columns))


def decode_video_frames(video_path: str, frame_indices: Sequence[int]) -> list[np.ndarray]:
    """Read RGB uint8 frames in requested order, including repeated indices.

    Seek to the preceding keyframe, then decode forwards. If seeking cannot
    recover every requested frame, reopen and count from frame zero. Reopening
    also resets decoder state after a failed seek. Neither path resizes frames.
    """
    import av

    if not frame_indices:
        return []
    targets = set(int(index) for index in frame_indices)
    if min(targets) < 0:
        raise ValueError("video frame indices must be nonnegative")
    found: dict[int, np.ndarray] = {}
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        if rate and stream.time_base and min(targets) > 0:
            origin = stream.start_time or 0
            ticks = 1 / (rate * stream.time_base)
            try:
                container.seek(
                    origin + max(0, int((min(targets) - 2) * ticks)),
                    stream=stream,
                    backward=True,
                    any_frame=False,
                )
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    index = round((frame.pts - origin) / ticks)
                    if index in targets:
                        found[index] = frame.to_ndarray(format="rgb24")
                    if index >= max(targets):
                        break
            except av.error.FFmpegError:
                logger.debug("Video seek failed for %s", video_path, exc_info=True)
    if not targets.issubset(found):
        found.clear()
        with av.open(video_path) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in targets:
                    found[index] = frame.to_ndarray(format="rgb24")
                if index >= max(targets):
                    break
    missing = targets - found.keys()
    if missing:
        raise RuntimeError(f"Missing frames {sorted(missing)} in {video_path}")
    return [found[int(index)] for index in frame_indices]


class LeRobotV3DataLoader(BaseDataset):
    """Read episode-contained windows from native v3 Parquet and image/video data.

    Numeric features have a leading ``sequence_length`` dimension. Cameras
    have a leading ``len(image_offsets)`` dimension and retain HWC uint8 RGB.
    Short tails repeat the final real row; ``time_mask`` identifies real rows.
    Offsets are computed on the full manifest before split/episode selection.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        sequence_length: int,
        feature_keys: Sequence[str],
        camera_keys: Sequence[str],
        image_offsets: Sequence[int] = (0,),
        window_stride: int = 1,
        split: str = "train",
        episode_indices: Sequence[int] | None = None,
        pad_end: bool = True,
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.root = Path(dataset_dir).expanduser().resolve()
        self.sequence_length = int(sequence_length)
        self.window_stride = int(window_stride)
        self.feature_keys = tuple(feature_keys)
        self.camera_keys = tuple(camera_keys)
        self.image_offsets = tuple(int(value) for value in image_offsets)
        if self.sequence_length <= 0 or self.window_stride <= 0:
            raise ValueError("sequence_length and window_stride must be positive")
        if not self.image_offsets or any(
            offset < 0 or offset >= self.sequence_length for offset in self.image_offsets
        ):
            raise ValueError("image_offsets must lie inside the sequence window")
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError(f"LeRobotV3DataLoader requires v3.0 data: {self.root}")
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("dataset fps must be finite and positive")
        features = self.info["features"]
        for key in (*self.feature_keys, *self.camera_keys):
            if key not in features:
                raise ValueError(f"Dataset metadata is missing feature {key!r}")
        for key in self.camera_keys:
            if features[key]["dtype"] not in {"image", "video"}:
                raise ValueError(f"Camera {key!r} must have dtype image or video")

        paths = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No episode metadata in {self.root}/meta/episodes")
        tables = []
        for path in paths:
            columns = [key for key in pq.read_schema(path).names if not key.startswith("stats/")]
            tables.append(pq.ParquetFile(path).read(columns=columns))
        episodes = pa.concat_tables(tables).to_pandas().sort_values("dataset_from_index")
        if episodes.episode_index.duplicated().any() or (episodes.length <= 0).any():
            raise ValueError("Episode metadata requires unique IDs and positive lengths")
        if not np.array_equal(
            episodes.dataset_to_index - episodes.dataset_from_index, episodes.length
        ):
            raise ValueError("Episode lengths disagree with dataset index ranges")
        # Physical offsets must not change when a split drops earlier episodes.
        episodes["_row_offset"] = (
            episodes.groupby(["data/chunk_index", "data/file_index"], sort=False).length.cumsum()
            - episodes.length
        )
        splits = self.info["splits"]
        if split not in splits:
            raise ValueError(f"Unknown split {split!r}; available splits: {list(splits)}")
        bounds = str(splits[split]).split(":")
        if len(bounds) != 2:
            raise ValueError(f"Unsupported LeRobot split range: {splits[split]!r}")
        start = int(bounds[0]) if bounds[0] else 0
        end = int(bounds[1]) if bounds[1] else int(self.info["total_episodes"])
        episodes = episodes[episodes.episode_index.between(start, end - 1)]
        if episode_indices is not None:
            requested = {int(value) for value in episode_indices}
            missing = requested - set(episodes.episode_index)
            if missing:
                raise ValueError(f"Episodes absent from split {split!r}: {sorted(missing)}")
            episodes = episodes[episodes.episode_index.isin(requested)]
        self.episodes = episodes.reset_index(drop=True)
        lengths = self.episodes.length.to_numpy(dtype=np.int64)
        starts = lengths if pad_end else np.maximum(0, lengths - self.sequence_length + 1)
        counts = (starts + self.window_stride - 1) // self.window_stride
        self._cumulative = np.concatenate(([0], np.cumsum(counts)))
        if not len(self):
            raise ValueError(f"No sampleable windows in {self.root} split {split!r}")
        tasks = pq.read_table(self.root / "meta/tasks.parquet").to_pandas()
        self.tasks = {int(row.task_index): str(prompt) for prompt, row in tasks.iterrows()}
        self.num_episodes = len(self.episodes)

    @property
    def data_spec(self) -> dict[str, Any]:
        """Return source metadata and selected window geometry."""
        return {
            "source": str(self.root),
            "features": self.info["features"],
            "sequence_length": self.sequence_length,
            "num_episodes": self.num_episodes,
            "num_windows": len(self),
        }

    def get_normalizer(self) -> None:
        """Return no normalizer: this reader exposes unmodified source values."""
        return None

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = int(np.searchsorted(self._cumulative, index, side="right") - 1)
        episode = self.episodes.iloc[position]
        offset = (index - int(self._cumulative[position])) * self.window_stride
        count = min(self.sequence_length, int(episode.length) - offset)
        columns = tuple(
            dict.fromkeys(
                (
                    *self.feature_keys,
                    "episode_index",
                    "frame_index",
                    "task_index",
                    *(
                        key
                        for key in self.camera_keys
                        if self.info["features"][key]["dtype"] == "image"
                    ),
                )
            )
        )
        path = self.root / self.info["data_path"].format(
            chunk_index=int(episode["data/chunk_index"]),
            file_index=int(episode["data/file_index"]),
        )
        table = _read_table(str(path), columns).slice(int(episode["_row_offset"]) + offset, count)
        if table.num_rows != count or any(
            value != int(episode.episode_index) for value in table["episode_index"].to_pylist()
        ):
            raise ValueError(f"Parquet rows disagree with episode {episode.episode_index}: {path}")
        if table["frame_index"].to_pylist() != list(range(offset, offset + count)):
            raise ValueError(f"Frame indices disagree with episode window in {path}")
        sample: dict[str, Any] = {}
        take = np.minimum(np.arange(self.sequence_length), count - 1)
        for key in self.feature_keys:
            values = np.asarray(table[key].to_pylist())
            feature = self.info["features"][key]
            values = values.astype(feature["dtype"], copy=False)
            if list(values.shape[1:]) != feature["shape"]:
                raise ValueError(f"Feature {key!r} shape disagrees with metadata: {values.shape}")
            sample[key] = values[take]
        image_rows = np.minimum(self.image_offsets, count - 1)
        for key in self.camera_keys:
            if self.info["features"][key]["dtype"] == "video":
                prefix = f"videos/{key}"
                video_path = self.root / self.info["video_path"].format(
                    video_key=key,
                    chunk_index=int(episode[f"{prefix}/chunk_index"]),
                    file_index=int(episode[f"{prefix}/file_index"]),
                )
                # Video shards can have different boundaries than data shards.
                base = round(float(episode[f"{prefix}/from_timestamp"]) * self.fps)
                frames = decode_video_frames(str(video_path), (base + offset + image_rows).tolist())
            else:
                frames = []
                for row in image_rows:
                    value = table[key][int(row)].as_py()
                    source = (
                        io.BytesIO(value["bytes"])
                        if value.get("bytes")
                        else self.root / value["path"]
                    )
                    with Image.open(source) as image:
                        frames.append(np.asarray(image.convert("RGB")))
            sample[key] = np.stack(frames)
        task_ids = table["task_index"].to_pylist()
        if len(set(task_ids)) != 1 or int(task_ids[0]) not in self.tasks:
            raise ValueError(f"Invalid task indices for episode {episode.episode_index}")
        sample.update(
            prompt=self.tasks[int(task_ids[0])],
            episode_index=int(episode.episode_index),
            frame_index=offset,
            time_mask=np.arange(self.sequence_length) < count,
        )
        return sample
