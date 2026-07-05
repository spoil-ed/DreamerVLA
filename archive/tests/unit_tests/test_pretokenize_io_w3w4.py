"""Tests for PERF W3 (manifest-first sequence index) and W4 (per-worker frame
cache) in ``pretokenize_dataset``.

Both optimizations must be byte-identical to the pre-optimization pickle-scan /
per-frame-load behavior; the spies on ``pickle.load`` are the RED drivers that
the IO shape actually changed.
"""

from __future__ import annotations

import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

if "h5py" not in sys.modules:
    try:
        import h5py  # noqa: F401
    except ImportError:
        h5py_stub = types.ModuleType("h5py")
        h5py_stub.File = object
        h5py_stub.Group = object
        sys.modules["h5py"] = h5py_stub


_TASK = "lift_the_block"
_TRJ = 3


def _trj_dir(root: Path) -> Path:
    return root / "frames" / _TASK / f"trj_{_TRJ}"


def _image_path(root: Path, frame: int) -> str:
    return str(_trj_dir(root) / "imgs_third_view" / f"image_{frame}.png")


def _wrist_path(root: Path, frame: int) -> str:
    return str(_trj_dir(root) / "imgs_wrist" / f"image_{frame}.png")


def _write_frame(root: Path, frame: int, *, with_current_image_in_manifest: bool) -> dict:
    """Write one his=1 pkl frame plus its on-disk action/reward siblings.

    Returns the manifest record. ``with_current_image_in_manifest`` controls
    whether the manifest carries the *current* frame image (so the manifest-first
    W3 path is byte-identical) under ``meta.next_obs.image`` / ``next_obs.image``.
    """
    trj = _trj_dir(root)
    (trj / "imgs_third_view").mkdir(parents=True, exist_ok=True)
    (trj / "imgs_wrist").mkdir(parents=True, exist_ok=True)
    (trj / "action").mkdir(parents=True, exist_ok=True)
    (trj / "reward").mkdir(parents=True, exist_ok=True)

    # Distinct, deterministic action/reward per frame so a wrong sibling path
    # would change the emitted tensors.
    np.save(trj / "action" / f"action_{frame}.npy", np.array([frame + 0.5, frame - 0.5], dtype=np.float32))
    np.save(trj / "reward" / f"reward_{frame}.npy", np.array([float(frame) * 0.1], dtype=np.float32))

    images = [_image_path(root, frame), _wrist_path(root, frame)]
    payload = {
        "token": [11, 22, 100 + frame, 8710],
        "label": [-100, -100, 100 + frame, 8710],
        "id": frame,
        "meta": {"task_name": _TASK, "task_text": _TASK.replace("_", " ")},
        "task_name": _TASK,
        "image": images,
        "action": [[float(frame + 0.5), float(frame - 0.5)]],
        "state": [],
        "reward": float(frame) * 0.1,
        "next_obs": {},
        "wm_obs_input_ids": [11, 22, 100 + frame],
        "wm_next_obs_input_ids": [33, 44, 200 + frame],
    }
    pkl_path = root / "pkls" / f"frame_{frame}.pkl"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle)

    record: dict = {
        "file": str(pkl_path),
        "len": len(payload["token"]),
        "id": frame,
        "meta": dict(payload["meta"]),
        "reward": payload["reward"],
        "next_obs": {},
    }
    if with_current_image_in_manifest:
        # Mirror one_trajectory_pretokenize_dataset's preferred manifest field,
        # but carrying the CURRENT frame image so the index is byte-identical.
        record["meta"]["next_obs"] = {"image": list(images)}
        record["next_obs"] = {"image": list(images)}
    return record


def _build_dataset_files(
    root: Path, n_frames: int, *, with_current_image_in_manifest: bool
) -> Path:
    records = [
        _write_frame(root, frame, with_current_image_in_manifest=with_current_image_in_manifest)
        for frame in range(n_frames)
    ]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(records), encoding="utf-8")
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps({"META": [{"path": str(manifest_path)}]}), encoding="utf-8"
    )
    return config_path


def _make_sequence_dataset(config_path: Path):
    from dreamervla.dataset.pretokenize_dataset import PretokenizeDataset

    # sequence_length=2, history=1 -> obs_count=2; stride=1 -> overlapping windows.
    return PretokenizeDataset(
        config_path=config_path,
        history=1,
        batch_length=1,
        replay_context=1,
        sequence_length=2,
        stride=1,
    )


def _frame_index_signature(dataset) -> list[tuple]:
    """Stable signature of the built per-frame index (the W3 artifact)."""
    sig: list[tuple] = []
    for key in sorted(dataset._frames_by_key):
        frame_map = dataset._frames_by_key[key]
        for fidx in sorted(frame_map):
            frame = frame_map[fidx]
            sig.append(
                (
                    key,
                    fidx,
                    frame.file,
                    frame.task_name,
                    frame.trajectory_key,
                    frame.frame_index,
                    frame.image_path,
                    frame.action_path,
                    frame.reward_path,
                )
            )
    return sig


def _window_signature(dataset) -> list[tuple]:
    return [tuple(rec.file for rec in window.records) for window in dataset._windows]


def _spy_pickle_load(monkeypatch) -> dict:
    import dreamervla.dataset.pretokenize_dataset as mod

    state = {"calls": 0, "by_path": {}}
    real_load = pickle.load

    def _counting_load(handle, *args, **kwargs):
        state["calls"] += 1
        name = getattr(handle, "name", None)
        if name is not None:
            state["by_path"][name] = state["by_path"].get(name, 0) + 1
        return real_load(handle, *args, **kwargs)

    monkeypatch.setattr(mod.pickle, "load", _counting_load)
    return state


# --------------------------------------------------------------------------- W3


def test_w3_manifest_index_equals_pickle_scan_index(tmp_path: Path) -> None:
    scan_cfg = _build_dataset_files(
        tmp_path / "scan", 4, with_current_image_in_manifest=False
    )
    man_cfg = _build_dataset_files(
        tmp_path / "man", 4, with_current_image_in_manifest=True
    )

    scan_ds = _make_sequence_dataset(scan_cfg)
    man_ds = _make_sequence_dataset(man_cfg)

    # The two roots differ only in their tmp prefix; compare the index relative
    # to each root so the comparison is path-prefix independent.
    def _strip(sig, root):
        prefix = str(root)
        return [
            tuple(c.replace(prefix, "<ROOT>") if isinstance(c, str) else c for c in row)
            for row in sig
        ]

    assert _strip(_window_signature(man_ds), tmp_path / "man") == _strip(
        _window_signature(scan_ds), tmp_path / "scan"
    )
    assert _strip(_frame_index_signature(man_ds), tmp_path / "man") == _strip(
        _frame_index_signature(scan_ds), tmp_path / "scan"
    )

    assert len(man_ds) == len(scan_ds)
    for idx in range(len(scan_ds)):
        scan_item = scan_ds[idx]
        man_item = man_ds[idx]
        assert man_item["wm_obs_input_ids_seq"] == scan_item["wm_obs_input_ids_seq"]
        assert torch.equal(man_item["action_seq"], scan_item["action_seq"])
        assert torch.equal(man_item["reward_seq"], scan_item["reward_seq"])
        assert [m["frame_index"] for m in man_item["meta_seq"]] == [
            m["frame_index"] for m in scan_item["meta_seq"]
        ]


def test_w3_init_does_not_pickle_load_when_manifest_has_current_image(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = _build_dataset_files(
        tmp_path, 4, with_current_image_in_manifest=True
    )
    state = _spy_pickle_load(monkeypatch)
    _make_sequence_dataset(config_path)
    assert state["calls"] == 0, (
        "manifest-first index must not pickle.load any frame in __init__ "
        f"when the manifest carries the current-frame image (got {state['calls']})"
    )


def test_w3_init_falls_back_to_pickle_scan_without_manifest_field(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = _build_dataset_files(
        tmp_path, 4, with_current_image_in_manifest=False
    )
    state = _spy_pickle_load(monkeypatch)
    _make_sequence_dataset(config_path)
    assert state["calls"] >= 4, "fallback path must still pickle-scan every frame"


# --------------------------------------------------------------------------- W4


def test_w4_overlapping_windows_return_identical_shared_frame(tmp_path: Path) -> None:
    config_path = _build_dataset_files(
        tmp_path, 3, with_current_image_in_manifest=True
    )
    dataset = _make_sequence_dataset(config_path)
    assert len(dataset) >= 2

    w0 = dataset[0]  # frames (0, 1)
    w1 = dataset[1]  # frames (1, 2) -> shares frame 1

    # frame 1 is index 1 in w0 and index 0 in w1.
    assert w0["wm_obs_input_ids_seq"][1] == w1["wm_obs_input_ids_seq"][0]
    assert w0["meta_seq"][1]["file"] == w1["meta_seq"][0]["file"]
    assert w0["meta_seq"][1]["frame_index"] == w1["meta_seq"][0]["frame_index"]


def test_w4_shared_frame_loaded_from_disk_once(tmp_path: Path, monkeypatch) -> None:
    config_path = _build_dataset_files(
        tmp_path, 3, with_current_image_in_manifest=True
    )
    dataset = _make_sequence_dataset(config_path)

    state = _spy_pickle_load(monkeypatch)
    shared_pkl = str(tmp_path / "pkls" / "frame_1.pkl")

    _ = dataset[0]  # loads frames 0, 1
    _ = dataset[1]  # loads frames 1 (cached), 2

    assert state["by_path"].get(shared_pkl, 0) == 1, (
        "frame shared by two overlapping windows must be pickle.load'ed once "
        f"(per-worker cache hit); got {state['by_path'].get(shared_pkl, 0)}"
    )
