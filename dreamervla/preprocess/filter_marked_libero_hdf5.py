#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from dreamervla.preprocess.libero_utils.noop_marking import filter_marked_hdf5_dir
from dreamervla.utils.hydra_config import script_namespace


def parse_args() -> SimpleNamespace:
    return script_namespace("filter_marked_libero_hdf5")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    records = filter_marked_hdf5_dir(
        input_dir,
        output_dir,
        filter_noops=bool(args.filter_noops),
        threshold=float(args.threshold),
        overwrite=bool(args.overwrite),
    )
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "filter_noops": bool(args.filter_noops),
        "files": len(records),
        "demos": sum(int(record["demos"]) for record in records),
        "frames_in": sum(int(record["frames_in"]) for record in records),
        "frames_out": sum(int(record["frames_out"]) for record in records),
        "noop_frames": sum(int(record["noop_frames"]) for record in records),
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "noop_filter_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[filter-marked-libero] "
        f"files={summary['files']} demos={summary['demos']} "
        f"frames={summary['frames_in']}->{summary['frames_out']} "
        f"noop={summary['noop_frames']} out={output_dir}"
    )


if __name__ == "__main__":
    main()
