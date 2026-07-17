"""Download one LIBERO suite from an immutable Hugging Face revision."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import snapshot_download

SOURCE_MARKER = ".dreamervla-source.json"


def download_libero(*, repo: str, revision: str, suite: str, target: Path) -> None:
    """Download ``suite`` at ``revision`` and persist its source identity."""

    destination = Path(target).expanduser().resolve()
    if destination.name != str(suite):
        raise ValueError(f"LIBERO target must end with the suite name {suite!r}: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(repo),
        repo_type="dataset",
        revision=str(revision),
        local_dir=str(destination.parent),
        allow_patterns=f"{suite}/*",
    )
    if not destination.is_dir():
        raise RuntimeError(f"LIBERO download did not create {destination}")
    marker = {
        "repo": str(repo),
        "revision": str(revision),
        "suite": str(suite),
    }
    marker_path = destination / SOURCE_MARKER
    temporary = marker_path.with_suffix(f"{marker_path.suffix}.tmp")
    temporary.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--target", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pinned LIBERO downloader."""

    args = _parser().parse_args(argv)
    download_libero(
        repo=args.repo,
        revision=args.revision,
        suite=args.suite,
        target=args.target,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SOURCE_MARKER", "download_libero", "main"]
