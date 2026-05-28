"""
Smoke-test helper: extract a small subset from a single raw LIBERO HDF5 file
into the image/action/state directory structure expected by the rest of the
pretokenize pipeline.

Unlike the full pipeline (which replays demos in MuJoCo at 256x256), this
script reads the raw HDF5 directly and resizes 128x128 images to the target
resolution via PIL.  No robosuite / MuJoCo dependency required.

Usage:
    python smoke_extract_hdf5.py \
        --hdf5 /path/to/task_demo.hdf5 \
        --save_dir /path/to/output \
        --max_demos 2 \
        --resolution 256
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def save_png(image_array: np.ndarray, output_path: Path, target_size: int) -> None:
    """Save a numpy image array as PNG, flipping and resizing as needed."""
    # LIBERO raw images are upside-down; flip to match the full pipeline
    image = image_array[::-1, ::-1]
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    img = Image.fromarray(image)
    if img.size[0] != target_size or img.size[1] != target_size:
        img = img.resize((target_size, target_size), Image.LANCZOS)
    img.save(output_path)


def extract_demo(
    demo_data: h5py.Group,
    trj_dir: Path,
    resolution: int,
) -> None:
    """Extract a single demo into the expected directory structure."""
    actions = demo_data["actions"][()]
    rewards = (
        demo_data["rewards"][()]
        if "rewards" in demo_data
        else np.zeros(
            actions.shape[0],
            dtype=np.float32,
        )
    )
    ee_states = demo_data["obs"]["ee_states"][()]
    gripper_states = demo_data["obs"]["gripper_states"][()]
    robot_states = demo_data["robot_states"][()]
    rgb_third = demo_data["obs"]["agentview_rgb"][()]
    rgb_wrist = demo_data["obs"]["eye_in_hand_rgb"][()]

    subdirs = {
        "action": trj_dir / "action",
        "ee_state": trj_dir / "ee_state",
        "gripper_state": trj_dir / "gripper_state",
        "eef_gripper_state": trj_dir / "eef_gripper_state",
        "robot_state": trj_dir / "robot_state",
        "reward": trj_dir / "reward",
        "imgs_third_view": trj_dir / "imgs_third_view",
        "imgs_wrist": trj_dir / "imgs_wrist",
    }
    for d in subdirs.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    num_steps = actions.shape[0]
    for step in range(num_steps):
        np.save(subdirs["action"] / f"action_{step}.npy", actions[step])
        np.save(
            subdirs["reward"] / f"reward_{step}.npy",
            np.asarray(rewards[step], dtype=np.float32),
        )
        np.save(subdirs["ee_state"] / f"ee_state_{step}.npy", ee_states[step])
        np.save(
            subdirs["gripper_state"] / f"gripper_state_{step}.npy", gripper_states[step]
        )
        np.save(
            subdirs["eef_gripper_state"] / f"eef_gripper_state_{step}.npy",
            np.concatenate([ee_states[step], gripper_states[step]]),
        )
        np.save(subdirs["robot_state"] / f"robot_state_{step}.npy", robot_states[step])
        save_png(
            rgb_third[step],
            subdirs["imgs_third_view"] / f"image_{step}.png",
            resolution,
        )
        save_png(
            rgb_wrist[step], subdirs["imgs_wrist"] / f"image_{step}.png", resolution
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--hdf5", type=Path, required=True, help="Path to a single *_demo.hdf5 file"
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Output base directory (task subdir created automatically)",
    )
    parser.add_argument(
        "--max_demos",
        type=int,
        default=2,
        help="Max number of demos to extract (default: 2)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Target image resolution (default: 256)",
    )
    args = parser.parse_args()

    hdf5_path: Path = args.hdf5
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

    # Derive task name from filename: "turn_on_the_stove_demo.hdf5" -> "turn_on_the_stove"
    task_name = hdf5_path.stem.replace("_demo", "")
    task_dir = args.save_dir / task_name

    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
        selected = demo_keys[: args.max_demos]
        print(f"Task: {task_name}")
        print(f"Total demos in file: {len(demo_keys)}, extracting: {len(selected)}")

        for demo_key in selected:
            trj_idx = int(demo_key.split("_")[1])
            trj_dir = task_dir / f"trj_{trj_idx}"
            print(f"  Extracting {demo_key} -> {trj_dir}")
            extract_demo(data[demo_key], trj_dir, args.resolution)

    print(f"Done. Output at: {task_dir}")


if __name__ == "__main__":
    main()
