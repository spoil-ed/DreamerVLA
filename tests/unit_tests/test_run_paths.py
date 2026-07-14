from __future__ import annotations

from pathlib import Path

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

    assert resolve_resume_checkpoint(run_dir) == canonical.resolve()
    assert resolve_resume_checkpoint(legacy) == legacy.resolve()


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
