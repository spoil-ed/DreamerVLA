from __future__ import annotations

from pathlib import Path

import pytest

from dreamervla.utils.run_paths import infer_run_root, resolve_resume_checkpoint


def test_infer_run_root_accepts_run_and_canonical_checkpoint_paths(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "dreamer-wm" / "20260714_120000"
    checkpoint = run_dir / "checkpoints" / "global_step_12" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert infer_run_root(run_dir) == run_dir.resolve()
    assert infer_run_root(checkpoint) == run_dir.resolve()
    assert infer_run_root(checkpoint.parent) == run_dir.resolve()


def test_infer_run_root_keeps_legacy_checkpoint_layout_readable(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "legacy"
    checkpoint = run_dir / "ckpt" / "warmup_progress" / "wm_step_8.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert infer_run_root(checkpoint) == run_dir.resolve()


def test_resolve_resume_checkpoint_prefers_canonical_latest(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    canonical = run_dir / "checkpoints" / "latest.ckpt"
    legacy = run_dir / "ckpt" / "latest.ckpt"
    canonical.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    canonical.touch()
    legacy.touch()
    (run_dir / "stray.ckpt").touch()

    assert resolve_resume_checkpoint(run_dir) == canonical.resolve()
    assert resolve_resume_checkpoint(legacy) == legacy.resolve()


def test_resolve_resume_checkpoint_accepts_canonical_checkpoint_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    latest = checkpoint_dir / "latest.ckpt"
    checkpoint_dir.mkdir(parents=True)
    latest.touch()

    assert resolve_resume_checkpoint(checkpoint_dir) == latest.resolve()


def test_resolve_resume_checkpoint_accepts_checkpoint_hf_directory(
    tmp_path: Path,
) -> None:
    checkpoint_hf = tmp_path / "run" / "checkpoint_hf"
    checkpoint_hf.mkdir(parents=True)
    (checkpoint_hf / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_hf / "model.safetensors").touch()

    assert resolve_resume_checkpoint(checkpoint_hf) == checkpoint_hf.resolve()
    assert infer_run_root(checkpoint_hf) == checkpoint_hf.parent.resolve()


def test_resolve_resume_checkpoint_reports_expected_latest(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    with pytest.raises(
        FileNotFoundError,
        match=r"requested directory: .*checkpoints.*expected latest: .*latest\.ckpt",
    ):
        resolve_resume_checkpoint(checkpoint_dir)


def test_resolve_resume_checkpoint_finds_latest_global_step(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    older = run_dir / "checkpoints" / "global_step_2" / "manual_cotrain.ckpt"
    latest = run_dir / "checkpoints" / "global_step_10" / "manual_cotrain.ckpt"
    older.parent.mkdir(parents=True)
    latest.parent.mkdir(parents=True)
    older.touch()
    latest.touch()

    assert resolve_resume_checkpoint(run_dir) == latest.resolve()


def test_resolve_resume_checkpoint_prefers_manual_then_canonical_warmups(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    wm = run_dir / "checkpoints" / "wm_warmup.ckpt"
    classifier = run_dir / "checkpoints" / "classifier_warmup.ckpt"
    legacy_wm = run_dir / "ckpt" / "wm_warmup.ckpt"
    legacy_progress = run_dir / "ckpt" / "warmup_progress" / "wm_step_99.ckpt"
    for path in (wm, classifier, legacy_wm, legacy_progress):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert resolve_resume_checkpoint(run_dir) == wm.resolve()
    wm.unlink()
    assert resolve_resume_checkpoint(run_dir) == classifier.resolve()
    classifier.unlink()
    assert resolve_resume_checkpoint(run_dir) == legacy_wm.resolve()
    legacy_wm.unlink()
    assert resolve_resume_checkpoint(run_dir) == legacy_progress.resolve()
