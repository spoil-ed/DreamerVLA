"""Summarize the artifacts produced by classifier evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dreamervla.utils.paths import data_path

DEFAULT_CLASSIFIER_EXPERIMENT = "wmpo_token_classifier_openvla_onetraj_libero_goal_h1"


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _latest_child(root: Path) -> Path:
    _require_path(root, "run family directory")
    children = [item for item in root.iterdir() if item.is_dir()]
    if not children:
        raise FileNotFoundError(f"no run directories under {root}")
    return max(children, key=lambda item: item.stat().st_mtime)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def cls_eval(args: argparse.Namespace) -> int:
    """Write a compact summary of one completed classifier run."""

    run_dir = (
        Path(args.run_dir).expanduser()
        if args.run_dir
        else _latest_child(data_path("outputs/classifier", args.family))
    )
    _require_path(run_dir, "classifier run directory")
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "log" / "train_log.jsonl"
    ckpt_dir = run_dir / "checkpoints"
    _require_path(summary_path, "classifier summary")
    _require_path(log_path, "classifier train log")
    _require_path(ckpt_dir, "classifier checkpoint directory")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = _read_jsonl(log_path)
    val_window = [record for record in records if record.get("event") == "val_window"]
    val_episode = [record for record in records if record.get("event") == "val_episode"]
    ckpts = sorted(str(path) for path in ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no classifier checkpoints under {ckpt_dir}")

    payload = {
        "status": "ok",
        "stage": "cls-eval",
        "run_dir": str(run_dir),
        "summary": summary,
        "num_val_window_records": len(val_window),
        "num_val_episode_records": len(val_episode),
        "checkpoints": ckpts,
    }
    out = Path(args.out).expanduser() if args.out else run_dir / "classifier_eval_summary.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["out"] = str(out)
    _print_json(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cls_eval_parser = subparsers.add_parser("cls-eval")
    cls_eval_parser.add_argument("--run-dir", default=None)
    cls_eval_parser.add_argument("--family", default=DEFAULT_CLASSIFIER_EXPERIMENT)
    cls_eval_parser.add_argument("--out", default=None)
    cls_eval_parser.set_defaults(func=cls_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
