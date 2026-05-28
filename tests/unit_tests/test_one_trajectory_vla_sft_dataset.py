from __future__ import annotations

import json
import pickle
import sys
import types
from pathlib import Path


if "h5py" not in sys.modules:
    h5py_stub = types.ModuleType("h5py")
    h5py_stub.File = object
    h5py_stub.Group = object
    sys.modules["h5py"] = h5py_stub


def _write_payload(
    root: Path, task: str, trj: int, frame: int, item_id: int
) -> dict[str, object]:
    path = root / f"{task}_trj{trj}_frame{frame}.pkl"
    image_base = f"/dataset/{task}/trj_{trj}"
    payload = {
        "token": [11, 22, 1000 + item_id, 8710],
        "label": [-100, -100, 1000 + item_id, 8710],
        "id": item_id,
        "meta": {"task_name": task, "task_text": task.replace("_", " ")},
        "task_name": task,
        "image": [
            f"{image_base}/imgs_third_view/image_{frame}.png",
            f"{image_base}/imgs_wrist/image_{frame}.png",
        ],
        "action": [[float(item_id), float(frame)]],
        "state": [],
        "reward": 0.0,
        "next_obs": {},
        "wm_obs_input_ids": [11, 22],
        "wm_next_obs_input_ids": [33, 44],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return {
        "file": str(path),
        "len": len(payload["token"]),
        "id": item_id,
        "meta": payload["meta"],
        "reward": 0.0,
        "next_obs": {},
    }


def test_one_trajectory_action_chunk_dataset_keeps_one_traj_per_task(
    tmp_path: Path,
) -> None:
    from src.dataloader.one_trajectory_pretokenize_dataset import (
        OneTrajectoryPretokenizeActionChunkDataset,
    )

    records: list[dict[str, object]] = []
    item_id = 0
    for task, trajectories in {"task_a": {0: 3, 1: 3}, "task_b": {0: 3, 1: 2}}.items():
        for trj, frames in trajectories.items():
            for frame in range(frames):
                records.append(_write_payload(tmp_path, task, trj, frame, item_id))
                item_id += 1

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(records), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "META": [{"path": str(manifest_path)}],
                "prompt_text": "Finish the task: {task_text}.",
            }
        ),
        encoding="utf-8",
    )

    dataset = OneTrajectoryPretokenizeActionChunkDataset(
        config_path=config_path,
        action_horizon=2,
        trajectories_per_task=1,
        trajectory_offset=0,
    )

    assert len(dataset) == 4
    assert dataset.selected_trajectory_keys == ("task_a/trj_0", "task_b/trj_0")

    seen = set()
    for idx in range(len(dataset)):
        item = dataset[idx]
        seen.add(item["meta"]["trajectory_key"])
        assert item["meta"]["one_trajectory_sft"] is True
        assert item["meta"]["trajectories_per_task"] == 1
        assert item["wm_action"].shape == (2, 2)

    assert seen == {"task_a/trj_0", "task_b/trj_0"}


def test_one_trajectory_action_chunk_dataset_can_select_offset(tmp_path: Path) -> None:
    from src.dataloader.one_trajectory_pretokenize_dataset import (
        OneTrajectoryPretokenizeActionChunkDataset,
    )

    records: list[dict[str, object]] = []
    item_id = 0
    for trj in (0, 1):
        for frame in range(3):
            records.append(_write_payload(tmp_path, "task_a", trj, frame, item_id))
            item_id += 1

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(records), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"META": [{"path": str(manifest_path)}]}), encoding="utf-8"
    )

    dataset = OneTrajectoryPretokenizeActionChunkDataset(
        config_path=config_path,
        action_horizon=2,
        trajectories_per_task=1,
        trajectory_offset=1,
    )

    assert dataset.selected_trajectory_keys == ("task_a/trj_1",)
    assert {dataset[idx]["meta"]["trajectory_key"] for idx in range(len(dataset))} == {
        "task_a/trj_1"
    }
