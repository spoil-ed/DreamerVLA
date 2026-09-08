"""Native v3 indexing, episode boundaries and image/video alignment."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dreamervla.dataset.base import lerobot_v3_dataloader as reader_module
from dreamervla.dataset.base.lerobot_v3_dataloader import LeRobotV3DataLoader, decode_video_frames
from dreamervla.dataset.libero import LiberoDataset

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pd = pytest.importorskip("pandas")

CAMERAS = ("observation.images.image", "observation.images.image2")


@pytest.fixture()
def native_v3(tmp_path: Path) -> Path:
    """Two episodes share a data shard but have independent video offsets."""
    meta = tmp_path / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "total_episodes": 3,
        "splits": {"train": "0:3", "val": "1:3"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [7]},
            "observation.state": {"dtype": "float32", "shape": [8]},
            **{key: {"dtype": "video", "shape": [4, 4, 3]} for key in CAMERAS},
        },
    }
    (meta / "info.json").write_text(json.dumps(info))
    pd.DataFrame({"task_index": [0, 1, 2]}, index=["task zero", "task one", "task two"]).to_parquet(
        meta / "tasks.parquet"
    )
    episodes = []
    shards: dict[int, list] = {0: [], 1: []}
    for eid, length in enumerate((3, 4, 2)):
        start = (0, 3, 7)[eid]
        episode = {
            "episode_index": eid,
            "length": length,
            "dataset_from_index": start,
            "dataset_to_index": start + length,
            "data/chunk_index": 0,
            "data/file_index": int(eid == 2),
        }
        for camera in CAMERAS:
            episode.update(
                {
                    f"videos/{camera}/chunk_index": 0,
                    f"videos/{camera}/file_index": int(eid > 0),
                    f"videos/{camera}/from_timestamp": (0, 0.8, 2.0)[eid],
                }
            )
        episodes.append(episode)
        for frame in range(length):
            shards[int(eid == 2)].append(
                {
                    "episode_index": eid,
                    "frame_index": frame,
                    "task_index": eid,
                    "action": [float(10 * eid + frame)] * 7,
                    "observation.state": [float(100 * eid + frame)] * 8,
                }
            )
    # Deliberately shuffled manifest: physical offsets follow dataset indices.
    pq.write_table(
        pa.Table.from_pylist(episodes[::-1]), meta / "episodes/chunk-000/file-000.parquet"
    )
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    for shard, rows in shards.items():
        pq.write_table(
            pa.Table.from_pylist(rows), tmp_path / f"data/chunk-000/file-{shard:03d}.parquet"
        )
    return tmp_path


def make_dataset(root: Path, **kwargs: object) -> LiberoDataset:
    return LiberoDataset(
        root,
        sequence_length=3,
        image_key=CAMERAS[0],
        wrist_image_key=CAMERAS[1],
        action_key="action",
        state_key="observation.state",
        **kwargs,
    )


@pytest.fixture()
def decoded_indices(monkeypatch: pytest.MonkeyPatch) -> list:
    calls = []

    def decode(path: str, indices: list[int]) -> list[np.ndarray]:
        calls.append((path, indices))
        return [np.full((4, 4, 3), index, dtype=np.uint8) for index in indices]

    monkeypatch.setattr(reader_module, "decode_video_frames", decode)
    return calls


def test_split_and_selection_preserve_physical_offsets(
    native_v3: Path, decoded_indices: list
) -> None:
    dataset = make_dataset(native_v3, split="val", episode_indices=[1])
    assert len(dataset) == 4
    sample = dataset[0]
    np.testing.assert_array_equal(sample["actions"][:, 0], [10, 11, 12])
    assert sample["state"].shape == (8,)
    assert sample["actions"].shape == (3, 7)
    assert sample["actions"].dtype == np.float32
    assert sample["prompt"] == "task one"
    assert sample["episode_index"] == 1
    assert decoded_indices[0][1] == [8]
    assert decoded_indices[0][0].endswith("file-001.mp4")


def test_tail_padding_never_crosses_episode(native_v3: Path, decoded_indices: list) -> None:
    dataset = make_dataset(native_v3)
    sample = dataset[2]
    np.testing.assert_array_equal(sample["actions"], np.full((3, 7), 2))
    np.testing.assert_array_equal(sample["action_mask"], [True, False, False])
    assert sample["episode_index"] == 0
    assert dataset[3]["episode_index"] == 1
    assert dataset[-1]["frame_index"] == 1
    assert decoded_indices[-1][1] == [21]
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_temporal_camera_windows_and_stride(native_v3: Path, decoded_indices: list) -> None:
    dataset = LeRobotV3DataLoader(
        native_v3,
        sequence_length=3,
        feature_keys=["action"],
        camera_keys=CAMERAS,
        image_offsets=[0, 2],
        episode_indices=[1],
        window_stride=2,
    )
    sample = dataset[1]
    assert decoded_indices[0][1] == [10, 11]
    np.testing.assert_array_equal(sample["time_mask"], [True, True, False])
    assert len(make_dataset(native_v3, pad_end=False)) == 3


def test_image_payload_without_video_decoder(
    native_v3: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info_path = native_v3 / "meta/info.json"
    info = json.loads(info_path.read_text())
    for camera in CAMERAS:
        info["features"][camera]["dtype"] = "image"
    info_path.write_text(json.dumps(info))
    for path in (native_v3 / "data").rglob("*.parquet"):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            stream = io.BytesIO()
            Image.fromarray(np.full((4, 4, 3), row["frame_index"], dtype=np.uint8)).save(
                stream, format="PNG"
            )
            for camera in CAMERAS:
                row[camera] = {"bytes": stream.getvalue(), "path": None}
        pq.write_table(pa.Table.from_pylist(rows), path)

    def no_video(*args: object) -> None:
        pytest.fail("Image features must not invoke video decoding")

    monkeypatch.setattr(reader_module, "decode_video_frames", no_video)
    sample = make_dataset(native_v3)[1]
    np.testing.assert_array_equal(sample["image"], np.ones((4, 4, 3), dtype=np.uint8))


def test_corrupt_alignment_fails_without_sample_substitution(
    native_v3: Path, decoded_indices: list
) -> None:
    path = native_v3 / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["episode_index"] = 42
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="Parquet rows disagree"):
        make_dataset(native_v3)[0]
    assert decoded_indices == []


def test_reject_v2_metadata(native_v3: Path) -> None:
    path = native_v3 / "meta/info.json"
    info = json.loads(path.read_text())
    info["codebase_version"] = "v2.0"
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="requires v3.0"):
        make_dataset(native_v3)


def test_video_seek_matches_sequential_decode(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    path = tmp_path / "video.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "10"}
        for index in range(40):
            pixels = np.full((32, 32, 3), index * 5, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        reference = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    for indices in ([0, 2], [21, 17, 21, 39]):
        actual = decode_video_frames(str(path), indices)
        for frame, index in zip(actual, indices, strict=True):
            np.testing.assert_array_equal(frame, reference[index])
    with pytest.raises(RuntimeError, match="Missing frames"):
        decode_video_frames(str(path), [40])
