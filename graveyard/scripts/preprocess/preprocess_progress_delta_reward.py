#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEME_NAME = "pi06_progress_delta_v1"


def _project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _demo_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("demo_"):
        try:
            return int(name.split("_")[-1]), name
        except ValueError:
            pass
    return 10**9, name


def _task_key_from_file(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith("_demo.hdf5"):
        return name[: -len("_demo.hdf5")]
    return Path(name).stem


def _load_metainfo(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    resolved = _project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"metainfo JSON does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compression(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = str(name).lower()
    if normalized in {"", "none", "null", "false", "0"}:
        return None
    if normalized not in {"lzf", "gzip"}:
        raise ValueError(f"Unsupported HDF5 compression: {name}")
    return normalized


def _metainfo_success(
    metainfo: dict[str, Any], source_path: Path, demo_key: str
) -> bool | None:
    if not metainfo:
        return None
    task_key = _task_key_from_file(source_path)
    task_info = metainfo.get(task_key)
    if not isinstance(task_info, dict):
        return None
    demo_info = task_info.get(str(demo_key))
    if not isinstance(demo_info, dict) or "success" not in demo_info:
        return None
    return bool(demo_info["success"])


def progress_delta_reward(
    sparse_rewards: np.ndarray,
    *,
    success: bool | None = None,
    success_threshold: float = 0.5,
    failure_value: float = 0.0,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build per-step telescoping progress reward from sparse terminal rewards.

    For successful episodes with first-positive index ``s``, define
    success-to-go ``S(t) = min_value + clip(t/s, 0, 1) * (max_value - min_value)``.
    The emitted per-step reward is the delta ``r(t) = S(t+1) - S(t)``, padded
    with 0 at the terminal step so the array length matches the input.

    Properties:
      sum(r) == max_value - min_value  for successful episodes
      sum(r) == 0                       for failed episodes (or s == 0)
      r(t) is constant 1/s within [0, s) and 0 elsewhere (range [0,1] default)
    """
    rewards = np.asarray(sparse_rewards).reshape(-1)
    length = int(rewards.shape[0])
    if length <= 0:
        raise ValueError("cannot build progress-delta rewards for an empty episode")

    positive = np.flatnonzero(rewards > float(success_threshold))
    inferred_success = bool(positive.size > 0)
    is_success = inferred_success if success is None else bool(success)

    if not is_success:
        target = np.full(length, float(failure_value), dtype=np.float32)
        return target, {
            "success": False,
            "success_index": -1,
            "source_positive_rewards": int(positive.size),
            "reward_sum": float(target.sum()),
        }

    success_index = int(positive[0]) if positive.size else length - 1
    span = float(max_value) - float(min_value)

    if success_index <= 0 or length == 1:
        # Degenerate: success at t=0 (no progress to make) or length-1 episode
        target = np.zeros(length, dtype=np.float32)
    else:
        steps = np.arange(length, dtype=np.float32)
        progress = np.clip(steps / float(success_index), 0.0, 1.0)
        s_to_go = float(min_value) + progress * span  # length T
        delta = np.diff(s_to_go)  # length T-1
        # Pad terminal step with 0 (no reward after final transition)
        target = np.concatenate([delta, np.zeros(1, dtype=np.float32)]).astype(
            np.float32, copy=False
        )

    return target, {
        "success": True,
        "success_index": success_index,
        "source_positive_rewards": int(positive.size),
        "reward_sum": float(target.sum()),
    }


def _copy_file_with_progress_delta(
    source_path: Path,
    output_path: Path,
    *,
    metainfo: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    demos = 0
    successes = 0
    failures = 0
    frames = 0
    reward_min = float("inf")
    reward_max = float("-inf")
    reward_sum_total = 0.0
    success_reward_sums: list[float] = []

    with (
        h5py.File(source_path, "r", swmr=True, libver="latest") as src,
        h5py.File(tmp_path, "w", libver="latest") as dst,
    ):
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for key in src.keys():
            src.copy(key, dst)

        data = dst["data"]
        for demo_key in sorted(data.keys(), key=_demo_sort_key):
            demo = data[demo_key]
            if "rewards" not in demo and "sparse_rewards" not in demo:
                raise KeyError(
                    f"{source_path}:{demo_key} missing rewards/sparse_rewards"
                )
            # Prefer pre-existing sparse_rewards (from prior pi06_remaining run);
            # fall back to raw rewards in the original dataset
            if "sparse_rewards" in demo:
                sparse_rewards = np.asarray(demo["sparse_rewards"], dtype=np.float32)
            else:
                sparse_rewards = np.asarray(demo["rewards"], dtype=np.float32)
            success_from_meta = _metainfo_success(metainfo, source_path, demo_key)
            shaped, info = progress_delta_reward(
                sparse_rewards,
                success=success_from_meta,
                success_threshold=float(args.success_threshold),
                failure_value=float(args.failure_value),
                min_value=float(args.min_value),
                max_value=float(args.max_value),
            )
            # Persist a copy of sparse_rewards (idempotent)
            if "sparse_rewards" not in demo:
                demo.copy("rewards", "sparse_rewards")
            if "rewards" in demo:
                del demo["rewards"]
            reward_dset = demo.create_dataset(
                "rewards",
                data=shaped.astype(np.float32, copy=False),
                dtype=np.float32,
                compression=_compression(args.compression),
            )
            reward_dset.attrs["scheme"] = SCHEME_NAME
            reward_dset.attrs["source"] = "episode_success_progress_delta"
            reward_dset.attrs["min_value"] = float(args.min_value)
            reward_dset.attrs["max_value"] = float(args.max_value)
            reward_dset.attrs["failure_value"] = float(args.failure_value)
            reward_dset.attrs["success"] = bool(info["success"])
            reward_dset.attrs["success_index"] = int(info["success_index"])
            reward_dset.attrs["source_positive_rewards"] = int(
                info["source_positive_rewards"]
            )
            reward_dset.attrs["reward_sum"] = float(info["reward_sum"])
            demo.attrs["reward_scheme"] = SCHEME_NAME
            demo.attrs["reward_success"] = bool(info["success"])
            demo.attrs["reward_success_index"] = int(info["success_index"])

            demos += 1
            frames += int(shaped.shape[0])
            successes += int(bool(info["success"]))
            failures += int(not bool(info["success"]))
            reward_min = min(reward_min, float(shaped.min()))
            reward_max = max(reward_max, float(shaped.max()))
            reward_sum_total += float(shaped.sum())
            if info["success"]:
                success_reward_sums.append(float(info["reward_sum"]))

        dst.attrs["reward_scheme"] = SCHEME_NAME
        dst.attrs["reward_source_hdf5"] = str(source_path)
        dst.attrs["reward_min_value"] = float(args.min_value)
        dst.attrs["reward_max_value"] = float(args.max_value)
        dst.attrs["reward_failure_value"] = float(args.failure_value)

    tmp_path.replace(output_path)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "demos": demos,
        "successes": successes,
        "failures": failures,
        "frames": frames,
        "reward_min": reward_min if demos else None,
        "reward_max": reward_max if demos else None,
        "reward_mean": reward_sum_total / max(frames, 1),
        "success_reward_sum_mean": (
            sum(success_reward_sums) / len(success_reward_sums)
            if success_reward_sums
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite LIBERO HDF5 rewards into per-step telescoping progress deltas "
            "(success-to-go increments). Successful episodes get sum(r)=max-min; "
            "failed episodes get all zeros."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(
            PROJECT_ROOT / "data" / "processed_data" / "libero_goal_no_noops_t_256"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT
            / "data"
            / "processed_data"
            / "libero_goal_no_noops_t_256_pi06_progress_delta"
        ),
    )
    parser.add_argument("--metainfo-json", default=None)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--failure-value", type=float, default=0.0)
    parser.add_argument("--min-value", type=float, default=0.0)
    parser.add_argument("--max-value", type=float, default=1.0)
    parser.add_argument(
        "--compression", default="none", choices=["none", "lzf", "gzip"]
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = _project_path(args.input_dir)
    output_dir = _project_path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input HDF5 directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metainfo = _load_metainfo(args.metainfo_json)
    files = sorted(input_dir.glob("*.hdf5"))
    if args.max_files is not None:
        files = files[: int(args.max_files)]
    if not files:
        raise RuntimeError(f"No HDF5 files found under {input_dir}")

    records: list[dict[str, Any]] = []
    for source_path in tqdm(files, desc="progress-delta rewards"):
        output_path = output_dir / source_path.name
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"output exists, pass --overwrite to replace: {output_path}"
            )
        if output_path.exists():
            output_path.unlink()
        records.append(
            _copy_file_with_progress_delta(
                source_path,
                output_path,
                metainfo=metainfo,
                args=args,
            )
        )

    summary = {
        "scheme": SCHEME_NAME,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": len(records),
        "demos": sum(int(r["demos"]) for r in records),
        "successes": sum(int(r["successes"]) for r in records),
        "failures": sum(int(r["failures"]) for r in records),
        "frames": sum(int(r["frames"]) for r in records),
        "reward_min": min(
            float(r["reward_min"]) for r in records if r["reward_min"] is not None
        ),
        "reward_max": max(
            float(r["reward_max"]) for r in records if r["reward_max"] is not None
        ),
        "reward_mean": (
            sum(float(r["reward_mean"]) * int(r["frames"]) for r in records)
            / max(sum(int(r["frames"]) for r in records), 1)
        ),
        "records": records,
    }
    (output_dir / "progress_delta_reward_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[progress-delta] "
        f"wrote {len(records)} files demos={summary['demos']} "
        f"success={summary['successes']} failure={summary['failures']} "
        f"reward=[{summary['reward_min']:.4f}, {summary['reward_max']:.4f}] "
        f"mean={summary['reward_mean']:.4f} out={output_dir}"
    )


if __name__ == "__main__":
    main()
