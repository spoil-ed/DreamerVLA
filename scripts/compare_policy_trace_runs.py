#!/usr/bin/env python3
"""Compare LIBERO policy traces from pure VLA and DreamerVLA rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "policy_trace.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            record["_arrays"] = np.load(record["array_path"])
            records.append(record)
    return records


def _array(record: dict[str, Any], *names: str) -> np.ndarray | None:
    arrays = record["_arrays"]
    for name in names:
        if name in arrays.files:
            return np.asarray(arrays[name], dtype=np.float32)
    return None


def _flat(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _mse(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    left_f, right_f = _flat(left), _flat(right)
    if left_f is None or right_f is None or left_f.shape != right_f.shape:
        return None
    return float(np.mean(np.square(left_f - right_f)))


def _mae(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    left_f, right_f = _flat(left), _flat(right)
    if left_f is None or right_f is None or left_f.shape != right_f.shape:
        return None
    return float(np.mean(np.abs(left_f - right_f)))


def _cos(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    left_f, right_f = _flat(left), _flat(right)
    if left_f is None or right_f is None or left_f.shape != right_f.shape:
        return None
    denom = float(np.linalg.norm(left_f) * np.linalg.norm(right_f))
    if denom <= 1.0e-12:
        return None
    return float(np.dot(left_f, right_f) / denom)


def _first_env(record: dict[str, Any]) -> np.ndarray | None:
    chunk = _array(record, "action_chunk_env")
    if chunk is None:
        return None
    chunk = chunk.reshape(-1, chunk.shape[-1])
    return chunk[0, :7]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return float(np.mean(values))


def _equal_ids(left: np.ndarray | None, right: np.ndarray | None) -> bool | None:
    if left is None or right is None:
        return None
    return bool(left.shape == right.shape and np.array_equal(left.astype(np.int64), right.astype(np.int64)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vla", required=True, type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    traces = {
        "vla": _read_records(args.vla),
        "original": _read_records(args.original),
        "trained": _read_records(args.trained),
    }
    n = min(len(records) for records in traces.values())
    rows: list[dict[str, Any]] = []
    for idx in range(n):
        vla = traces["vla"][idx]
        orig = traces["original"][idx]
        trained = traces["trained"][idx]
        vla_hidden = _array(vla, "wm_style_action_hidden", "action_hidden", "obs_embedding")
        orig_live = _array(orig, "live_action_hidden", "obs_embedding")
        orig_recon = _array(orig, "recon_action_hidden", "actor_input")
        trained_live = _array(trained, "live_action_hidden", "obs_embedding")
        trained_recon = _array(trained, "recon_action_hidden", "actor_input")
        row = {
            "index": idx,
            "vla_env_step": vla.get("context", {}).get("env_step"),
            "original_env_step": orig.get("context", {}).get("env_step"),
            "trained_env_step": trained.get("context", {}).get("env_step"),
            "state_mse_vla_original": _mse(_array(vla, "state"), _array(orig, "state")),
            "state_mse_vla_trained": _mse(_array(vla, "state"), _array(trained, "state")),
            "input_ids_equal_vla_original": _equal_ids(_array(vla, "input_ids"), _array(orig, "input_ids")),
            "input_ids_equal_vla_trained": _equal_ids(_array(vla, "input_ids"), _array(trained, "input_ids")),
            "hidden_mse_vla_original_live": _mse(vla_hidden, orig_live),
            "hidden_cos_vla_original_live": _cos(vla_hidden, orig_live),
            "hidden_mse_vla_trained_live": _mse(vla_hidden, trained_live),
            "hidden_cos_vla_trained_live": _cos(vla_hidden, trained_live),
            "hidden_mse_original_trained_live": _mse(orig_live, trained_live),
            "hidden_cos_original_trained_live": _cos(orig_live, trained_live),
            "recon_mse_original_live": _mse(orig_live, orig_recon),
            "recon_cos_original_live": _cos(orig_live, orig_recon),
            "recon_mse_trained_live": _mse(trained_live, trained_recon),
            "recon_cos_trained_live": _cos(trained_live, trained_recon),
            "recon_mse_original_trained": _mse(orig_recon, trained_recon),
            "rssm_deter_mse_original_trained": _mse(_array(orig, "rssm_deter"), _array(trained, "rssm_deter")),
            "rssm_stoch_mse_original_trained": _mse(_array(orig, "rssm_stoch"), _array(trained, "rssm_stoch")),
            "action_mse_vla_original_env": _mse(_first_env(vla), _first_env(orig)),
            "action_mse_vla_trained_env": _mse(_first_env(vla), _first_env(trained)),
            "action_mse_original_trained_env": _mse(_first_env(orig), _first_env(trained)),
            "action_mae_original_trained_env": _mae(_first_env(orig), _first_env(trained)),
            "vla_first_action_env": _first_env(vla).tolist() if _first_env(vla) is not None else None,
            "original_first_action_env": _first_env(orig).tolist() if _first_env(orig) is not None else None,
            "trained_first_action_env": _first_env(trained).tolist() if _first_env(trained) is not None else None,
        }
        rows.append(row)

    summary = {
        "num_compared_trace_points": n,
        "runs": {name: str(path) for name, path in {"vla": args.vla, "original": args.original, "trained": args.trained}.items()},
        "means": {
            key: _mean(rows, key)
            for key in rows[0].keys()
            if rows and key not in {"index", "vla_first_action_env", "original_first_action_env", "trained_first_action_env"}
        },
        "first_step": rows[0] if rows else {},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    md = args.out.with_suffix(".md")
    lines = [
        "# Policy Trace Comparison",
        "",
        f"- VLA: `{args.vla}`",
        f"- VLA+original actor: `{args.original}`",
        f"- VLA+trained actor: `{args.trained}`",
        f"- Compared trace points: `{n}`",
        "",
        "## First Step",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in (rows[0] if rows else {}).items():
        if key.endswith("_first_action_env"):
            continue
        lines.append(f"| `{key}` | {_fmt(value)} |")
    lines += [
        "",
        "## Per Trace Point",
        "",
        "| idx | env step | state MSE VLA/orig | state MSE VLA/trained | hidden MSE VLA/orig live | hidden MSE VLA/trained live | recon MSE orig | recon MSE trained | action MSE orig/trained | RSSM deter MSE orig/trained |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["index"]),
                    _fmt(row["vla_env_step"]),
                    _fmt(row["state_mse_vla_original"]),
                    _fmt(row["state_mse_vla_trained"]),
                    _fmt(row["hidden_mse_vla_original_live"]),
                    _fmt(row["hidden_mse_vla_trained_live"]),
                    _fmt(row["recon_mse_original_live"]),
                    _fmt(row["recon_mse_trained_live"]),
                    _fmt(row["action_mse_original_trained_env"]),
                    _fmt(row["rssm_deter_mse_original_trained"]),
                ]
            )
            + " |"
        )
    md.write_text("\n".join(lines) + "\n")
    print(args.out)
    print(md)


if __name__ == "__main__":
    main()
