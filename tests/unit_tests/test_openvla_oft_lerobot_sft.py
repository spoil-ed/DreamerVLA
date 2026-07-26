from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


class _TinyImageProcessor:
    def apply_transform(self, image):
        array = np.asarray(image)
        return torch.from_numpy(array.copy()).permute(2, 0, 1)


class _TinyTokenizer:
    model_max_length = 128
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = True):
        _ = add_special_tokens
        return SimpleNamespace(input_ids=list(range(1, min(len(text), 120) + 1)))


class _TinyProcessor:
    image_processor = _TinyImageProcessor()
    tokenizer = _TinyTokenizer()


class _TinyActionTokenizer:
    def __call__(self, actions):
        array = np.asarray(actions)
        if array.ndim == 1:
            return "A" * int(array.shape[0])
        return ["A" * int(array.shape[1]) for _ in range(array.shape[0])]


def _write_metadata(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 4,
        "total_frames": 8,
        "total_tasks": 2,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.image": {"dtype": "video", "shape": [2, 2, 3]},
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [7]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    tasks = [
        {"task_index": 0, "task": "task zero"},
        {"task_index": 1, "task": "task one"},
    ]
    (root / "meta" / "tasks.jsonl").write_text(
        "\n".join(json.dumps(record) for record in tasks) + "\n", encoding="utf-8"
    )
    episodes = [
        {"episode_index": 0, "tasks": ["task zero"], "length": 2},
        {"episode_index": 1, "tasks": ["task zero"], "length": 2},
        {"episode_index": 2, "tasks": ["task one"], "length": 2},
        {"episode_index": 3, "tasks": ["task one"], "length": 2},
    ]
    (root / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(record) for record in episodes) + "\n",
        encoding="utf-8",
    )


def _write_episode(root: Path, episode_index: int) -> None:
    path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    actions = [
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    table = pa.Table.from_pydict(
        {
            "observation.state": [[0.0] * 8, [0.0] * 8],
            "action": actions,
        }
    )
    pq.write_table(table, path)


def test_lerobot_dataset_selects_one_episode_per_task_without_gripper_flip(
    tmp_path: Path, monkeypatch
) -> None:
    from dreamervla.dataset.vla_sft_lerobot_dataset import VLASFTLeRobotDataset

    _write_metadata(tmp_path)
    for episode_index in range(4):
        _write_episode(tmp_path, episode_index)

    def fake_read(path: Path, *, plugin: str):
        assert plugin == "FFMPEG"
        episode_index = int(Path(path).stem.removeprefix("episode_"))
        return np.full((2, 2, 2, 3), episode_index, dtype=np.uint8)

    monkeypatch.setattr("dreamervla.dataset.vla_sft_lerobot_dataset.iio.imread", fake_read)
    stats = {
        "action": {
            "q01": [-1.0] * 6 + [0.0],
            "q99": [1.0] * 7,
            "mask": [True] * 6 + [False],
        }
    }
    dataset = VLASFTLeRobotDataset(
        root_dir=tmp_path,
        processor=_TinyProcessor(),
        action_tokenizer=_TinyActionTokenizer(),
        dataset_statistics=stats,
        action_horizon=3,
        episodes_per_task=1,
        episode_selection="first",
    )

    assert len(dataset) == 4
    assert dataset.data_spec.one_trajectory_sft is True
    assert dataset.data_spec.selected_episode_indices == (0, 2)
    first = dataset[0]
    assert tuple(first["pixel_values"].shape) == (3, 2, 2)
    assert first["actions"].shape == (3, 7)
    assert first["actions"][:, -1].tolist() == [1.0, 0.0, 0.0]
    tail = dataset[1]
    assert tail["actions"][:, -1].tolist() == [0.0, 0.0, 0.0]


def test_lerobot_dataset_rejects_unknown_schema_version(tmp_path: Path) -> None:
    from dreamervla.dataset.vla_sft_lerobot_dataset import VLASFTLeRobotDataset

    _write_metadata(tmp_path)
    info_path = tmp_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["codebase_version"] = "v3.0"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    stats = {"action": {"q01": [-1.0] * 7, "q99": [1.0] * 7}}
    try:
        VLASFTLeRobotDataset(
            root_dir=tmp_path,
            processor=_TinyProcessor(),
            action_tokenizer=_TinyActionTokenizer(),
            dataset_statistics=stats,
        )
    except ValueError as error:
        assert "codebase_version='v2.1'" in str(error)
    else:
        raise AssertionError("expected an unsupported schema error")
