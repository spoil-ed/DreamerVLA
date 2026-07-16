from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dreamervla.runtime.reproduction import (
    ReproductionError,
    atomic_write_json,
    decide_stage,
    select_metric_checkpoint,
    sha256_file,
)


def test_sha256_file_hashes_file_content(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"dreamervla")

    assert sha256_file(path) == hashlib.sha256(b"dreamervla").hexdigest()


def test_atomic_write_json_replaces_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    atomic_write_json(path, {"schema_version": 1, "status": "complete"})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "status": "complete",
    }
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_select_metric_checkpoint_uses_minimum_loss(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    for name in (
        "epoch=0001-loss=0.400000.ckpt",
        "epoch=0002-loss=0.200000.ckpt",
        "epoch=0003-loss=0.300000.ckpt",
    ):
        (root / name).write_bytes(name.encode())

    selected = select_metric_checkpoint(root, metric_name="loss", mode="min")

    assert selected.path.name == "epoch=0002-loss=0.200000.ckpt"
    assert selected.epoch == 2
    assert selected.value == pytest.approx(0.2)
    assert selected.sha256 == sha256_file(selected.path)


def test_select_metric_checkpoint_uses_maximum_f1_and_latest_tie(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    for name in (
        "epoch=0003-f1=0.800000.ckpt",
        "epoch=0007-f1=0.900000.ckpt",
        "epoch=0008-f1=0.900000.ckpt",
    ):
        (root / name).write_bytes(name.encode())

    selected = select_metric_checkpoint(root, metric_name="f1", mode="max")

    assert selected.path.name == "epoch=0008-f1=0.900000.ckpt"


def test_select_metric_checkpoint_rejects_missing_candidates(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "latest.ckpt").touch()

    with pytest.raises(ReproductionError, match="no loss metric checkpoints"):
        select_metric_checkpoint(root, metric_name="loss", mode="min")


def test_decide_stage_starts_fresh_when_run_root_is_absent(tmp_path: Path) -> None:
    decision = decide_stage({}, stage="world_model", run_root=tmp_path / "world_model", budget=30)

    assert decision.action == "fresh"
    assert decision.resume_source is None


def test_decide_stage_resumes_when_latest_exists(tmp_path: Path) -> None:
    run_root = tmp_path / "world_model"
    latest = run_root / "checkpoints" / "latest.ckpt"
    latest.parent.mkdir(parents=True)
    latest.touch()

    decision = decide_stage({}, stage="world_model", run_root=run_root, budget=30)

    assert decision.action == "resume"
    assert decision.resume_source == run_root.resolve()


def test_decide_stage_skips_valid_completed_stage(tmp_path: Path) -> None:
    selected = tmp_path / "world_model" / "checkpoints" / "epoch=0030-loss=0.2.ckpt"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"wm")
    state = {
        "stages": {
            "world_model": {
                "status": "completed",
                "budget": 30,
                "selected_checkpoint": str(selected),
                "sha256": sha256_file(selected),
            }
        }
    }

    decision = decide_stage(
        state, stage="world_model", run_root=tmp_path / "world_model", budget=30
    )

    assert decision.action == "skip"
    assert decision.selected_checkpoint == selected.resolve()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("budget", 100, "budget mismatch"),
        ("sha256", "0" * 64, "hash mismatch"),
    ],
)
def test_decide_stage_rejects_completed_state_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    selected = tmp_path / "run" / "checkpoints" / "epoch=0001-loss=0.2.ckpt"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"checkpoint")
    record: dict[str, object] = {
        "status": "completed",
        "budget": 30,
        "selected_checkpoint": str(selected),
        "sha256": sha256_file(selected),
    }
    record[field] = value

    with pytest.raises(ReproductionError, match=match):
        decide_stage(
            {"stages": {"world_model": record}},
            stage="world_model",
            run_root=tmp_path / "run",
            budget=30,
        )
