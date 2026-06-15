"""
Re-runs LIBERO simulator on the demos that the original no-op filter run
REJECTED (i.e. the replays where `done=True` was never reached and so the
demo was thrown away by regenerate_libero_dataset_filter_no_op.py).

Why we need this: the original filter writes only successful replays to
HDF5, but it DOES record `success: False` + `initial_state` for every
rejected replay in `<suite>_metainfo.json`. With those initial states
and the original raw-data action sequences, we can deterministically
reproduce the failed trajectories on a CPU simulator and save them with
a clear `is_failure=True` marker so downstream training can use them
as negative samples.

Output layout (mirrors the original):
  <libero_target_dir>/<task>_demo.hdf5
    └─ data/
       └─ demo_<i>/        with .attrs['is_failure'] = True
                            and  .attrs['failure_source'] = 'no_op_replay_did_not_reach_done'

The script ONLY processes demos that metainfo marks `success: False`,
so it never overwrites or duplicates the success set.

Example:
    python regenerate_libero_failure_demos.py \
        --libero_task_suite libero_goal \
        --libero_raw_data_dir /path/data/libero/datasets/libero_goal \
        --libero_target_dir /path/data/processed_data/libero_goal/no_noops_t_256_failures \
        --libero_metainfo_json /path/libero_goal_metainfo.json \
        --image_resolution 256
"""

import argparse
import json
import os

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark

from dreamervla.preprocess.libero_utils.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from dreamervla.preprocess.libero_utils.noop_marking import is_noop_action


def _load_metainfo(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _is_failure_demo(metainfo: dict, task_key: str, demo_key: str) -> bool:
    """Return True iff metainfo records this demo as success=False."""
    task_info = metainfo.get(task_key)
    if not isinstance(task_info, dict):
        return False
    demo_info = task_info.get(demo_key)
    if not isinstance(demo_info, dict):
        return False
    if "success" not in demo_info:
        return False
    return not bool(demo_info["success"])


def main(args):
    print(f"Re-running failure demos for {args.libero_task_suite}")
    print(f"  raw dir:     {args.libero_raw_data_dir}")
    print(f"  target dir:  {args.libero_target_dir}")
    print(f"  metainfo:    {args.libero_metainfo_json}")
    print(f"  resolution:  {args.image_resolution}")

    os.makedirs(args.libero_target_dir, exist_ok=True)
    metainfo = _load_metainfo(args.libero_metainfo_json)

    # Counter aggregations across the whole suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks
    num_demos_attempted = 0
    num_demos_written = 0
    num_done_at_replay = 0  # sanity: should be 0 if metainfo+seeding are deterministic
    num_noops = 0

    summary = {
        "suite": args.libero_task_suite,
        "target_dir": args.libero_target_dir,
        "metainfo": args.libero_metainfo_json,
        "image_resolution": int(args.image_resolution),
        "tasks": {},
    }

    IMAGE_RESOLUTION = args.image_resolution

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="tasks"):
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, resolution=IMAGE_RESOLUTION)
        task_key = task_description.replace(" ", "_")

        # Identify which demos in metainfo are flagged failure for THIS task
        task_failure_keys = []
        if task_key in metainfo:
            for demo_key, info in metainfo[task_key].items():
                if isinstance(info, dict) and info.get("success") is False:
                    task_failure_keys.append(demo_key)
        task_failure_keys = sorted(
            task_failure_keys,
            key=lambda k: int(k.split("_")[-1]) if k.startswith("demo_") else 10**9,
        )

        per_task = {
            "metainfo_failures": len(task_failure_keys),
            "attempted": 0,
            "written": 0,
            "done_at_replay_unexpected": 0,
            "skipped_missing_raw": 0,
        }

        if not task_failure_keys:
            print(
                f"[task {task_id}] {task_description}: no failure demos in metainfo, skipping"
            )
            summary["tasks"][task_key] = per_task
            continue

        # Open raw source for actions/states
        orig_data_path = os.path.join(
            args.libero_raw_data_dir, f"{task.name}_demo.hdf5"
        )
        if not os.path.exists(orig_data_path):
            print(f"[task {task_id}] missing raw data {orig_data_path}, skipping")
            per_task["skipped_missing_raw"] = len(task_failure_keys)
            summary["tasks"][task_key] = per_task
            continue
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # New HDF5 just for the failure subset of this task
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")
        grp.attrs["is_failure_dir"] = True
        grp.attrs["origin"] = "no_op_filter_rejected_replays"
        grp.attrs["suite"] = args.libero_task_suite
        grp.attrs["task_description"] = task_description

        for demo_key in tqdm.tqdm(
            task_failure_keys, desc=f"  failures in task_{task_id}", leave=False
        ):
            if demo_key not in orig_data:
                print(f"  [{demo_key}] missing in raw HDF5, skipping")
                continue
            num_demos_attempted += 1
            per_task["attempted"] += 1

            demo_data = orig_data[demo_key]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            env.reset()
            env.set_init_state(orig_states[0])
            obs, reward, done, info = None, 0.0, False, {}
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action())

            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []

            for _, action in enumerate(orig_actions):
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop_action(action, prev_action):
                    num_noops += 1
                    continue

                if states == []:
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate(
                            [
                                obs["robot0_gripper_qpos"],
                                obs["robot0_eef_pos"],
                                obs["robot0_eef_quat"],
                            ]
                        )
                    )

                actions.append(action)
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )
                agentview_images.append(obs["agentview_image"])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                obs, reward, done, info = env.step(action.tolist())

            # The whole point of this script: keep these regardless of done.
            # If `done` came back True here it means our replay diverged from
            # the original metainfo recording (something nondeterministic) —
            # record it but don't pretend it's a failure.
            unexpectedly_done = bool(done)
            if unexpectedly_done:
                num_done_at_replay += 1
                per_task["done_at_replay_unexpected"] += 1
                print(
                    f"  [{demo_key}] WARNING: replay returned done=True; "
                    "metainfo said failure — likely sim nondeterminism. Still saving as failure."
                )

            n = len(actions)
            if n == 0:
                print(f"  [{demo_key}] empty replay (all no-ops?), skipping write")
                continue

            dones = np.zeros(n, dtype=np.uint8)
            dones[-1] = 1  # mark terminal step
            rewards = np.zeros(n, dtype=np.uint8)  # failure: no positive reward ever

            ep = grp.create_group(demo_key)
            obs_grp = ep.create_group("obs")
            if gripper_states:
                obs_grp.create_dataset(
                    "gripper_states", data=np.stack(gripper_states, axis=0)
                )
            obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
            obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
            obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
            obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
            obs_grp.create_dataset(
                "agentview_rgb", data=np.stack(agentview_images, axis=0)
            )
            obs_grp.create_dataset(
                "eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0)
            )
            ep.create_dataset("actions", data=actions)
            ep.create_dataset("states", data=np.stack(states))
            ep.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
            ep.create_dataset("rewards", data=rewards)
            ep.create_dataset("dones", data=dones)

            # ── Mark this demo as failure ─────────────────────────────
            ep.attrs["is_failure"] = True
            ep.attrs["failure_source"] = "no_op_replay_did_not_reach_done"
            ep.attrs["source_metainfo_json"] = args.libero_metainfo_json
            ep.attrs["replay_unexpectedly_done"] = bool(unexpectedly_done)

            num_demos_written += 1
            per_task["written"] += 1

        orig_data_file.close()
        new_data_file.close()
        summary["tasks"][task_key] = per_task
        print(
            f"[task {task_id}] {task_description}: "
            f"failures_in_meta={per_task['metainfo_failures']} "
            f"written={per_task['written']}/{per_task['attempted']}"
        )

    summary_path = os.path.join(args.libero_target_dir, "failure_regen_summary.json")
    summary["totals"] = {
        "demos_attempted": num_demos_attempted,
        "demos_written": num_demos_written,
        "done_at_replay_unexpected": num_done_at_replay,
        "noop_actions_skipped": num_noops,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== summary ===")
    print(f"  attempted:  {num_demos_attempted}")
    print(f"  written:    {num_demos_written}")
    print(f"  unexpected done at replay (saved anyway): {num_done_at_replay}")
    print(f"  noop actions skipped (cumulative):        {num_noops}")
    print(f"  summary json: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
        required=True,
    )
    parser.add_argument("--libero_raw_data_dir", type=str, required=True)
    parser.add_argument("--libero_target_dir", type=str, required=True)
    parser.add_argument(
        "--libero_metainfo_json",
        type=str,
        required=True,
        help="Path to <suite>_metainfo.json written by the original "
        "regenerate_libero_dataset_filter_no_op.py run.",
    )
    parser.add_argument("--image_resolution", type=int, default=256)
    args = parser.parse_args()
    main(args)
