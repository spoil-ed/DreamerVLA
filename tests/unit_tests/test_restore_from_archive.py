"""Tests for scripts/restore_from_archive.sh against the deprecation manifest."""

from __future__ import annotations

import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "restore_from_archive.sh"
MANIFEST = REPO / "docs" / "superpowers" / "DEPRECATION-manifest.md"

_ROW = re.compile(r"^\|\s*(?P<orig>[^|]+?)\s*\|\s*(?P<arch>archive/[^|]+?)\s*\|")


def _manifest_rows(text: str) -> list[tuple[str, str]]:
    rows = []
    for line in text.splitlines():
        m = _ROW.match(line)
        if m:
            rows.append((m.group("orig"), m.group("arch")))
    return rows


def test_dry_run_lists_one_action_per_manifest_row() -> None:
    rows = _manifest_rows(MANIFEST.read_text())
    assert rows, "manifest has no archive rows"

    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        capture_output=True, text=True, check=True,
    ).stdout
    planned = [ln for ln in out.splitlines() if ln.startswith("git mv ")]
    # every in-place original counts as skipped, not planned; on a fresh checkout
    # all 74 are archived (orig absent) so planned == rows.
    assert len(planned) + out.count("skip (already in place)") == len(rows)
    assert f"{len(planned)} restore action(s) planned" in out


def _mini_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A throwaway git repo mirroring the script's manifest+archive layout."""
    repo = tmp_path / "repo"
    (repo / "docs" / "superpowers").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "archive" / "pkg").mkdir(parents=True)
    (repo / "scripts" / "restore_from_archive.sh").write_bytes(SCRIPT.read_bytes())

    (repo / "archive" / "pkg" / "moved.py").write_text("# archived\n")
    (repo / "docs" / "superpowers" / "DEPRECATION-manifest.md").write_text(
        "| Original Path | Archive Path | Reason | Migration Commit |\n"
        "| --- | --- | --- | --- |\n"
        "| pkg/moved.py | archive/pkg/moved.py | test | staged |\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=repo, check=True,
    )
    return repo


def test_restore_moves_file_and_is_idempotent(tmp_path: pathlib.Path) -> None:
    repo = _mini_repo(tmp_path)
    script = repo / "scripts" / "restore_from_archive.sh"

    assert not (repo / "pkg" / "moved.py").exists()
    subprocess.run(["bash", str(script), "--all"], cwd=repo, capture_output=True,
                   text=True, check=True)
    assert (repo / "pkg" / "moved.py").exists()
    assert not (repo / "archive" / "pkg" / "moved.py").exists()

    # second run: original already in place → skipped, no error
    out = subprocess.run(["bash", str(script), "--all"], cwd=repo, capture_output=True,
                         text=True, check=True).stdout
    assert "skip (already in place): pkg/moved.py" in out


def test_restore_selects_named_path_only(tmp_path: pathlib.Path) -> None:
    repo = _mini_repo(tmp_path)
    script = repo / "scripts" / "restore_from_archive.sh"
    # a non-matching name restores nothing
    out = subprocess.run(["bash", str(script), "pkg/other.py"], cwd=repo,
                         capture_output=True, text=True, check=True).stdout
    assert "restored 0 file(s)" in out
    assert (repo / "archive" / "pkg" / "moved.py").exists()
