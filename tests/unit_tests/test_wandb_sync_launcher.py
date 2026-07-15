from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from dreamervla.launchers import wandb_sync


def _write_stream(
    wandb_dir: Path,
    *,
    run_id: str,
    timestamp: str = "20260715_120000",
    legacy: bool = False,
    synced: bool = False,
    run_prefix: str = "offline-run",
) -> Path:
    root = wandb_dir / "wandb" if legacy else wandb_dir
    run_dir = root / f"{run_prefix}-{timestamp}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stream = run_dir / f"run-{run_id}.wandb"
    stream.write_bytes(f"offline stream {timestamp}".encode())
    if synced:
        stream.with_name(f"{stream.name}.synced").touch()
    return stream


def _install_fake_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    create_marker: bool = True,
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "wandb-argv.jsonl"
    executable = bin_dir / "wandb"
    executable.write_text(
        f"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

with Path(os.environ["WANDB_FAKE_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
status = int(os.environ.get("WANDB_FAKE_EXIT", "0"))
if status == 0 and {create_marker!r}:
    stream = Path(sys.argv[-1])
    stream.with_name(stream.name + ".synced").touch()
raise SystemExit(status)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("WANDB_FAKE_LOG", str(log_path))
    return log_path


def _calls(log_path: Path) -> list[list[str]]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def _snapshot(directory: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_syncs_one_canonical_segment_without_extra_remote_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    (wandb_dir / "run_id.txt").write_text("stable_id\n", encoding="utf-8")
    stream = _write_stream(wandb_dir, run_id="stable_id")
    before = _snapshot(wandb_dir)
    log_path = _install_fake_wandb(tmp_path, monkeypatch)

    assert wandb_sync.main([str(wandb_dir)]) == 0

    assert _calls(log_path) == [["sync", "--id", "stable_id", str(stream)]]
    for relative_path, contents in before.items():
        assert (wandb_dir / relative_path).read_bytes() == contents


def test_syncs_segments_chronologically_and_appends_only_after_first_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    later = _write_stream(wandb_dir, run_id="same-id", timestamp="20260715_130000")
    earlier = _write_stream(wandb_dir, run_id="same-id", timestamp="20260715_120000")
    log_path = _install_fake_wandb(tmp_path, monkeypatch)

    wandb_sync.main([str(wandb_dir)])

    assert _calls(log_path) == [
        ["sync", "--id", "same-id", str(earlier)],
        ["sync", "--id", "same-id", "--append", str(later)],
    ]


def test_sorts_run_and_offline_run_segments_by_timestamp_not_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    later_offline = _write_stream(
        wandb_dir,
        run_id="same-id",
        timestamp="20260715_130000",
        run_prefix="offline-run",
    )
    earlier_online = _write_stream(
        wandb_dir,
        run_id="same-id",
        timestamp="20260715_120000",
        run_prefix="run",
    )
    log_path = _install_fake_wandb(tmp_path, monkeypatch)

    wandb_sync.main([str(wandb_dir)])

    assert _calls(log_path) == [
        ["sync", "--id", "same-id", str(earlier_online)],
        ["sync", "--id", "same-id", "--append", str(later_offline)],
    ]


def test_discovers_legacy_nested_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    stream = _write_stream(wandb_dir, run_id="legacy", legacy=True)
    log_path = _install_fake_wandb(tmp_path, monkeypatch)

    wandb_sync.main([str(wandb_dir)])

    assert _calls(log_path) == [["sync", "--id", "legacy", str(stream)]]


def test_synced_markers_skip_upload_and_make_reruns_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    already_synced = _write_stream(
        wandb_dir,
        run_id="stable",
        timestamp="20260715_120000",
        synced=True,
    )
    pending = _write_stream(
        wandb_dir,
        run_id="stable",
        timestamp="20260715_130000",
    )
    log_path = _install_fake_wandb(tmp_path, monkeypatch)

    wandb_sync.main([str(wandb_dir)])
    wandb_sync.main([str(wandb_dir)])

    assert _calls(log_path) == [
        ["sync", "--id", "stable", "--append", str(pending)],
    ]
    assert already_synced.exists()
    assert pending.exists()


def test_launcher_creates_marker_after_success_for_idempotent_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    stream = _write_stream(wandb_dir, run_id="stable")
    marker = stream.with_name(f"{stream.name}.synced")
    log_path = _install_fake_wandb(tmp_path, monkeypatch, create_marker=False)

    wandb_sync.main([str(wandb_dir)])
    wandb_sync.main([str(wandb_dir)])

    assert _calls(log_path) == [["sync", "--id", "stable", str(stream)]]
    assert stream.exists()
    assert marker.is_file()


def test_requires_exactly_one_directory_argument() -> None:
    with pytest.raises(SystemExit):
        wandb_sync.main([])
    with pytest.raises(SystemExit):
        wandb_sync.main(["first", "second"])


def test_rejects_missing_wandb_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    _write_stream(wandb_dir, run_id="stable")
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

    with pytest.raises(RuntimeError, match="wandb CLI"):
        wandb_sync.main([str(wandb_dir)])


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_rejects_invalid_wandb_directory(tmp_path: Path, kind: str) -> None:
    path = tmp_path / kind
    if kind == "file":
        path.touch()

    with pytest.raises((FileNotFoundError, NotADirectoryError), match="W&B directory"):
        wandb_sync.main([str(path)])


def test_rejects_directory_without_supported_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    unsupported = wandb_dir / "arbitrary" / "offline-run-20260715_120000-hidden"
    unsupported.mkdir(parents=True)
    (unsupported / "run-hidden.wandb").touch()
    _install_fake_wandb(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="offline W&B segment"):
        wandb_sync.main([str(wandb_dir)])


@pytest.mark.parametrize("source", ["run_id", "filename"])
def test_rejects_invalid_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    if source == "run_id":
        (wandb_dir / "run_id.txt").write_text("bad/id", encoding="utf-8")
        _write_stream(wandb_dir, run_id="valid")
    else:
        _write_stream(wandb_dir, run_id="bad!")
    _install_fake_wandb(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="invalid W&B run ID"):
        wandb_sync.main([str(wandb_dir)])


def test_rejects_conflicting_segment_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    _write_stream(wandb_dir, run_id="first", timestamp="20260715_120000")
    _write_stream(wandb_dir, run_id="second", timestamp="20260715_130000")
    _install_fake_wandb(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="conflicting W&B run IDs"):
        wandb_sync.main([str(wandb_dir)])


def test_rejects_symlink_stream_even_when_target_is_a_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    run_dir = wandb_dir / "offline-run-20260715_120000-stable"
    run_dir.mkdir(parents=True)
    outside = tmp_path / "run-stable.wandb"
    outside.write_bytes(b"outside")
    (run_dir / "run-stable.wandb").symlink_to(outside)
    _install_fake_wandb(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="symlink"):
        wandb_sync.main([str(wandb_dir)])


def test_rejects_symlink_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    outside_run_dir = tmp_path / "outside-run"
    outside_run_dir.mkdir()
    (outside_run_dir / "run-stable.wandb").write_bytes(b"outside")
    (wandb_dir / "offline-run-20260715_120000-stable").symlink_to(
        outside_run_dir,
        target_is_directory=True,
    )
    _install_fake_wandb(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="symlink"):
        wandb_sync.main([str(wandb_dir)])


def test_propagates_cli_failure_without_modifying_local_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    stream = _write_stream(wandb_dir, run_id="stable")
    before = _snapshot(wandb_dir)
    log_path = _install_fake_wandb(tmp_path, monkeypatch)
    monkeypatch.setenv("WANDB_FAKE_EXIT", "23")

    with pytest.raises(subprocess.CalledProcessError) as error:
        wandb_sync.main([str(wandb_dir)])

    assert error.value.returncode == 23
    assert _calls(log_path) == [["sync", "--id", "stable", str(stream)]]
    assert _snapshot(wandb_dir) == before
