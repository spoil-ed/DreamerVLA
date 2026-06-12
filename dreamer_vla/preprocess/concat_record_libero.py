from __future__ import annotations

import argparse
from pathlib import Path

from dreamer_vla.preprocess.concat_record import concat_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Concatenate LIBERO token record shards.")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--task-prefix", default="libero")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    directories = sorted(
        path
        for path in args.base_dir.iterdir()
        if path.is_dir() and path.name.startswith(args.task_prefix)
    )
    if not directories:
        print(f"No subdirectories starting with {args.task_prefix!r} under {args.base_dir}")
        return 0
    for directory in directories:
        save_path = directory / "record.json"
        print(f"concat {directory} -> {save_path}")
        concat_records(directory, save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
