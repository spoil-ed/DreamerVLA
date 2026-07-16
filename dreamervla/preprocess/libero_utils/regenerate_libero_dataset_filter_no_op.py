"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px or 512x512px resolution.
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - In the LIBERO HDF5 data -> RLDS data conversion (not shown here), we rotate the images by
    180 degrees because we observe that the environments return images that are upside down
    on our platform.

Usage:
    Example (LIBERO-Spatial):
        python -m dreamervla.preprocess.libero_utils.regenerate_libero_dataset_filter_no_op \
            libero_task_suite=libero_spatial \
            libero_raw_data_dir=./third_party/LIBERO/libero/datasets/libero_spatial \
            libero_target_dir=./third_party/LIBERO/libero/datasets/libero_spatial_no_noops \
            image_resolution=512

"""

import json
import os

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
from libero.libero import benchmark

from dreamervla.envs.libero.utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from dreamervla.preprocess.libero_utils.noop_marking import (
    SCHEME_NAME,
    is_noop_action,
)
from dreamervla.utils.hydra_config import script_namespace
from dreamervla.utils.progress import ProgressReporter

NOOP_MARKING_SCHEME = SCHEME_NAME


def _complete_demo_group(group):
    required = (
        "actions",
        "states",
        "robot_states",
        "rewards",
        "dones",
        "noop_mask",
        "source_indices",
        "obs",
    )
    if any(key not in group for key in required):
        return False
    obs = group["obs"]
    return "agentview_rgb" in obs and "eye_in_hand_rgb" in obs


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")
    keep_noops = bool(getattr(args, "keep_noops", False))
    resume = bool(getattr(args, "resume", False))
    if keep_noops:
        print("No-op actions will be kept and marked with data/demo_*/noop_mask.")
    else:
        print("No-op actions will be filtered after marking.")

    # Create target directory
    if os.path.isdir(args.libero_target_dir):
        if resume:
            print(f"Resuming existing target directory: {args.libero_target_dir}")
        else:
            user_input = input(
                f"Target directory already exists at path: {args.libero_target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: "
            )
            if user_input != "y":
                exit()
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_out_path = (
        args.metainfo_json_out
        if args.metainfo_json_out
        else f"{args.libero_task_suite}_metainfo.json"
    )
    if resume and os.path.isfile(metainfo_json_out_path):
        try:
            with open(metainfo_json_out_path) as f:
                metainfo_json_dict = json.load(f)
            if not isinstance(metainfo_json_dict, dict):
                raise ValueError(f"metainfo JSON must be an object: {metainfo_json_out_path}")
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[resume] ignoring invalid metainfo JSON {metainfo_json_out_path}: {exc}")
            metainfo_json_dict = {}
    else:
        metainfo_json_dict = {}
        with open(metainfo_json_out_path, "w") as f:
            # Just test that we can write to this file (we overwrite it later)
            json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    IMAGE_RESOLUTION = args.image_resolution
    print(IMAGE_RESOLUTION)

    tasks_pbar = ProgressReporter(num_tasks_in_suite, "regenerate no-op filter", unit="task")
    for task_id in range(num_tasks_in_suite):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, resolution=IMAGE_RESOLUTION)

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "a" if resume else "w")
        new_data_file.attrs["noop_marking_scheme"] = NOOP_MARKING_SCHEME
        new_data_file.attrs["noop_keep_noops"] = bool(keep_noops)
        grp = new_data_file.require_group("data")
        task_key = task_description.replace(" ", "_")

        for i in range(len(orig_data.keys())):
            episode_key = f"demo_{i}"
            hdf5_complete = episode_key in grp and _complete_demo_group(grp[episode_key])
            existing_episode = metainfo_json_dict.get(task_key, {}).get(episode_key)
            if resume and existing_episode is not None:
                success = bool(existing_episode.get("success", False))
                if (not success) or hdf5_complete:
                    num_replays += 1
                    num_success += int(success)
                    print(
                        f"[resume] skip {task_key}/{episode_key}: "
                        f"success={success} hdf5_complete={hdf5_complete}"
                    )
                    continue

            if resume and episode_key in grp and not hdf5_complete:
                del grp[episode_key]

            # Get demo data
            demo_data = orig_data[episode_key]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action())

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []
            noop_mask = []
            source_indices = []
            prev_kept_action = None

            # Replay original demo actions in environment and record observations
            for action_idx, action in enumerate(orig_actions):
                action_is_noop = is_noop_action(action, prev_kept_action)
                if action_is_noop:
                    num_noops += 1
                    if not keep_noops:
                        print(f"\tSkipping no-op action: {action}")
                        continue
                else:
                    prev_kept_action = action

                noop_mask.append(bool(action_is_noop))
                source_indices.append(int(action_idx))

                if action_is_noop and keep_noops:
                    print(f"\tMarking no-op action: {action}")

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    # For all other timesteps, get state from environment and record it
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

                # Record original action (from demo)
                actions.append(action)

                # Record data returned by environment
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

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done and len(actions) > 0:
                if episode_key in grp:
                    del grp[episode_key]
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                ep_data_grp = grp.create_group(episode_key)
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)
                ep_data_grp.create_dataset("noop_mask", data=np.asarray(noop_mask, dtype=np.bool_))
                ep_data_grp.create_dataset(
                    "source_indices", data=np.asarray(source_indices, dtype=np.int64)
                )
                ep_data_grp.attrs["noop_marking_scheme"] = NOOP_MARKING_SCHEME
                ep_data_grp.attrs["noop_filtered"] = not bool(keep_noops)
                ep_data_grp.attrs["source_episode_length"] = int(len(orig_actions))
                ep_data_grp.attrs["source_noop_frames"] = int(
                    np.asarray(noop_mask, dtype=np.bool_).sum()
                )

                num_success += 1

            num_replays += 1

            # Record success/false and initial environment state in metainfo dict
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

            # Write metainfo dict to JSON file
            # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        # Close HDF5 files
        orig_data_file.close()
        new_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")
        tasks_pbar.update()

    tasks_pbar.close()
    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")


if __name__ == "__main__":
    args = script_namespace("regenerate_libero_dataset_filter_no_op")
    main(args)
