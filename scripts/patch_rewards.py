"""
Patch reward field in pretokenized pkl files.

The pkl files were generated without reward because the conv generation script
was an older version. This script reads rewards from the original HDF5 files
and patches each pkl in-place.

Reward logic (matching action_state_model_conv_generation.py):
  reward = hdf5['data']['demo_{trj_idx}']['rewards'][last_action_step]
  where last_action_step = int of last filename in payload['action'] list

Usage:
  conda run -n wmpo python scripts/patch_rewards.py
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import json
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HDF5_DIR = PROJECT_ROOT / "data/processed_data/libero_goal_no_noops_t_256"
TOKENS_DIRS = [
    PROJECT_ROOT / "data/processed_data/tokens/libero_goal_his_2_train_third_view_wrist_w_state_10_256",
    PROJECT_ROOT / "data/processed_data/tokens/libero_goal_his_2_val_ind_third_view_wrist_w_state_10_256",
    PROJECT_ROOT / "data/processed_data/tokens/libero_goal_his_2_val_ood_third_view_wrist_w_state_10_256",
]
CONCATE_JSON = PROJECT_ROOT / "data/processed_data/concate_tokens/libero_goal_his_2_third_view_wrist_w_state_10_256.json"
NUM_WORKERS = 16


def _load_hdf5_rewards(hdf5_dir: Path) -> dict[tuple[str, int], np.ndarray]:
    """Load all rewards from all HDF5 files. Returns {(task_name, trj_idx): rewards_array}."""
    import h5py
    cache: dict[tuple[str, int], np.ndarray] = {}
    for hdf5_path in sorted(hdf5_dir.glob("*_demo.hdf5")):
        # filename: <task_name>_demo.hdf5
        task_name = hdf5_path.stem[: -len("_demo")]
        with h5py.File(hdf5_path, "r") as f:
            for demo_key in f["data"].keys():
                m = re.match(r"demo_(\d+)$", demo_key)
                if m is None:
                    continue
                trj_idx = int(m.group(1))
                rewards = f["data"][demo_key]["rewards"][()].astype(np.float32)
                cache[(task_name, trj_idx)] = rewards
    return cache


def _extract_info_from_pkl(payload: dict) -> tuple[str, int, int] | None:
    """
    Extract (task_name, trj_idx, last_step) from the action file paths in payload.
    Action paths look like:
      .../libero_goal_image_state_action_t_256/<task>/trj_<N>/action/action_<S>.npy
    Returns the step index of the LAST action in the sequence (used for reward lookup).
    """
    action_list = payload.get("action", [])
    if not action_list:
        return None
    # Take the last action file to get the highest step index
    last_action_path = Path(action_list[-1])
    # Extract step index from filename: action_<S>.npy
    m_step = re.match(r"action_(\d+)\.npy$", last_action_path.name)
    if m_step is None:
        return None
    last_step = int(m_step.group(1))
    # Extract trj_idx from parent dirs: .../trj_<N>/action/action_S.npy
    trj_dir = last_action_path.parent.parent  # .../trj_<N>
    m_trj = re.match(r"trj_(\d+)$", trj_dir.name)
    if m_trj is None:
        return None
    trj_idx = int(m_trj.group(1))
    # Extract task_name: parent of trj_dir
    task_name = trj_dir.parent.name
    return task_name, trj_idx, last_step


def _patch_one(pkl_path_str: str, reward_cache: dict) -> tuple[str, float | None, str]:
    """Load, patch, and save one pkl file. Returns (path, reward, status)."""
    pkl_path = Path(pkl_path_str)
    try:
        with pkl_path.open("rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        return pkl_path_str, None, f"load_error: {e}"

    # Skip if reward already set
    existing = payload.get("reward")
    if existing is not None:
        return pkl_path_str, float(existing), "already_set"

    info = _extract_info_from_pkl(payload)
    if info is None:
        return pkl_path_str, None, "no_action_paths"

    task_name, trj_idx, last_step = info
    rewards_arr = reward_cache.get((task_name, trj_idx))
    if rewards_arr is None:
        # Fallback: sparse terminal reward (shouldn't happen if HDF5s are complete)
        reward_value = 1.0  # assume terminal
        status = f"hdf5_missing:{task_name}/{trj_idx} fallback=1.0"
    else:
        step = min(last_step, len(rewards_arr) - 1)
        reward_value = float(rewards_arr[step])
        status = "ok"

    payload["reward"] = reward_value
    if isinstance(payload.get("meta"), dict):
        payload["meta"]["reward"] = reward_value

    with pkl_path.open("wb") as f:
        pickle.dump(payload, f)

    return pkl_path_str, reward_value, status


def collect_pkl_paths(tokens_dirs: list[Path]) -> list[str]:
    paths = []
    for d in tokens_dirs:
        if not d.exists():
            print(f"  Warning: directory not found: {d}")
            continue
        for jsonl_file in sorted(d.glob("*-record.jsonl")):
            with jsonl_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    paths.append(record["file"])
        # Also check record.json
        record_json = d / "record.json"
        if record_json.exists():
            with record_json.open() as f:
                records = json.load(f)
            for r in records:
                paths.append(r["file"])
    return list(set(paths))


def main() -> None:
    print("Loading HDF5 reward cache...")
    reward_cache = _load_hdf5_rewards(HDF5_DIR)
    print(f"  Loaded rewards for {len(reward_cache)} (task, trj) pairs.")

    print("Collecting pkl paths...")
    pkl_paths = collect_pkl_paths(TOKENS_DIRS)
    print(f"  Found {len(pkl_paths)} pkl files.")

    if not pkl_paths:
        print("No pkl files found. Check TOKENS_DIRS paths.")
        return

    # Patch in parallel
    print(f"Patching with {NUM_WORKERS} workers...")
    stats = {"ok": 0, "already_set": 0, "error": 0, "fallback": 0}
    nonzero_rewards = 0

    from functools import partial
    patch_fn = partial(_patch_one, reward_cache=reward_cache)

    with Pool(NUM_WORKERS) as pool:
        for _, reward, status in tqdm.tqdm(
            pool.imap_unordered(patch_fn, pkl_paths, chunksize=64),
            total=len(pkl_paths),
        ):
            if status == "ok":
                stats["ok"] += 1
                if reward and reward > 0:
                    nonzero_rewards += 1
            elif status == "already_set":
                stats["already_set"] += 1
            elif "fallback" in status:
                stats["fallback"] += 1
            else:
                stats["error"] += 1

    print("\nDone!")
    print(f"  Patched:      {stats['ok']}")
    print(f"  Already set:  {stats['already_set']}")
    print(f"  Fallback:     {stats['fallback']}")
    print(f"  Errors:       {stats['error']}")
    print(f"  Non-zero rewards patched: {nonzero_rewards}")

    # Verify a sample
    sample_path = pkl_paths[0]
    with open(sample_path, "rb") as f:
        p = pickle.load(f)
    print(f"\nVerification sample: {Path(sample_path).name} -> reward={p.get('reward')}")


if __name__ == "__main__":
    main()
