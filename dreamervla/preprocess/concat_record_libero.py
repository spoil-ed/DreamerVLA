from __future__ import annotations

from pathlib import Path

from dreamervla.preprocess.concat_record import concat_records
from dreamervla.utils.hydra_config import script_namespace


def main() -> int:
    args = script_namespace("concat_record_libero")
    base_dir = Path(args.base_dir).expanduser()
    directories = sorted(
        path
        for path in base_dir.iterdir()
        if path.is_dir() and path.name.startswith(args.task_prefix)
    )
    if not directories:
        print(f"No subdirectories starting with {args.task_prefix!r} under {base_dir}")
        return 0
    for directory in directories:
        save_path = directory / "record.json"
        print(f"concat {directory} -> {save_path}")
        concat_records(directory, save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
