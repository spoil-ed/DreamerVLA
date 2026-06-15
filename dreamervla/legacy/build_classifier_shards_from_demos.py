"""Build webdataset shards for the LatentSuccessClassifier from precomputed
LIBERO demo action-hiddens.

Schema per sample (matches WMPO's reward_model/videomae.py expectations):
    <key>.latent.npy  shape [T, latent_dim] (float16)
    <key>.meta.json   {"finish_step": T-1, "complete": true, "task_id": int, "demo_id": int}

All demo episodes are successful, so this is a v1 dataset of positives.
Same-episode earlier-window negatives are sampled at classifier-training
time by the shard reader (LatentSuccessShardDataset), mirroring WMPO's
SuccessWindowDataset scheme.

To add failure-class negatives, a follow-up script must run pi0 SFT in
LIBERO sim and append failed episodes (complete=False, finish_step=max_steps-1).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
import webdataset as wds

from dreamervla.utils.paths import data_path, processed_data_path

DEFAULT_HIDDEN_DIR = str(
    processed_data_path(
        "libero_goal/no_noops_t_256_legacy_action_hidden_vla_policy_h2"
    )
)


def _iter_demos(hdf5_path: Path):
    with h5py.File(str(hdf5_path), "r") as h:
        if "data" not in h:
            return
        for demo_name, demo_group in h["data"].items():
            if "obs_embedding" not in demo_group:
                continue
            obs = np.asarray(demo_group["obs_embedding"][...], dtype=np.float16)
            yield demo_name, obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default=DEFAULT_HIDDEN_DIR)
    parser.add_argument(
        "--out_dir",
        default=str(data_path("legacy", "classifier_shards", "libero_goal")),
    )
    parser.add_argument("--episodes_per_shard", type=int, default=64)
    parser.add_argument("--prefix", default="libero_goal_demos")
    args = parser.parse_args()

    hidden_dir = Path(args.hidden_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(hidden_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise SystemExit(f"no .hdf5 files in {hidden_dir}")

    shard_idx = 0
    key_idx = 0
    sink = wds.TarWriter(str(out_dir / f"{args.prefix}_{shard_idx:06d}.tar"))
    total = 0
    success_count = 0
    fail_count = 0
    task_id = 0
    try:
        for hdf5_path in hdf5_files:
            print(f"reading {hdf5_path.name}")
            for demo_name, obs in _iter_demos(hdf5_path):
                T = int(obs.shape[0])
                if T < 16:  # too short for a window
                    continue
                finish_step = T - 1
                complete = True

                latent_buf = io.BytesIO()
                np.save(latent_buf, obs.astype(np.float16))
                meta = {
                    "finish_step": int(finish_step),
                    "complete": bool(complete),
                    "task_id": int(task_id),
                    "demo_id": demo_name,
                    "source_file": hdf5_path.name,
                }
                sink.write(
                    {
                        "__key__": f"{args.prefix}_{shard_idx:06d}_{key_idx:06d}",
                        "latent.npy": latent_buf.getvalue(),
                        "meta.json": json.dumps(meta).encode(),
                    }
                )
                key_idx += 1
                total += 1
                if complete:
                    success_count += 1
                else:
                    fail_count += 1
                if key_idx >= args.episodes_per_shard:
                    sink.close()
                    shard_idx += 1
                    key_idx = 0
                    sink = wds.TarWriter(
                        str(out_dir / f"{args.prefix}_{shard_idx:06d}.tar")
                    )
            task_id += 1
    finally:
        sink.close()

    print(
        f"wrote {total} samples across {shard_idx + 1} shards "
        f"({success_count} success / {fail_count} failed)"
    )


if __name__ == "__main__":
    main()
