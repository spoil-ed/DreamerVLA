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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from dreamervla.preprocess.libero_utils.parallel_replay import (
    ReplayTaskResult,
    ReplayTotals,
    atomic_write_json,
    commit_task_result,
    iter_task_results,
    load_resume_metadata,
    write_task_metadata_shard,
)
from dreamervla.utils.hydra_config import script_namespace
from dreamervla.utils.progress import ProgressReporter

NOOP_MARKING_SCHEME = SCHEME_NAME


@dataclass(frozen=True)
class ReplayTaskRequest:
    """Pickle-safe configuration for one task-exclusive replay worker."""

    task_id: int
    libero_task_suite: str
    libero_raw_data_dir: str
    libero_target_dir: str
    image_resolution: int
    keep_noops: bool
    resume: bool
    existing_metadata: dict[str, Any]
    shard_dir: str


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


def _write_successful_episode(
    group: h5py.Group,
    episode_key: str,
    *,
    orig_actions: np.ndarray,
    states: list[np.ndarray],
    actions: list[np.ndarray],
    robot_states: list[np.ndarray],
    gripper_states: list[np.ndarray],
    joint_states: list[np.ndarray],
    ee_states: list[np.ndarray],
    agentview_images: list[np.ndarray],
    eye_in_hand_images: list[np.ndarray],
    noop_mask: list[bool],
    source_indices: list[int],
    keep_noops: bool,
) -> None:
    if episode_key in group:
        del group[episode_key]
    dones = np.zeros(len(actions), dtype=np.uint8)
    dones[-1] = 1
    rewards = np.zeros(len(actions), dtype=np.uint8)
    rewards[-1] = 1
    assert len(actions) == len(agentview_images)

    episode = group.create_group(episode_key)
    observations = episode.create_group("obs")
    observations.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
    observations.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
    observations.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
    observations.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
    observations.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
    observations.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
    observations.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
    episode.create_dataset("actions", data=actions)
    episode.create_dataset("states", data=np.stack(states))
    episode.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
    episode.create_dataset("rewards", data=rewards)
    episode.create_dataset("dones", data=dones)
    episode.create_dataset("noop_mask", data=np.asarray(noop_mask, dtype=np.bool_))
    episode.create_dataset("source_indices", data=np.asarray(source_indices, dtype=np.int64))
    episode.attrs["noop_marking_scheme"] = NOOP_MARKING_SCHEME
    episode.attrs["noop_filtered"] = not keep_noops
    episode.attrs["source_episode_length"] = int(len(orig_actions))
    episode.attrs["source_noop_frames"] = int(np.asarray(noop_mask, dtype=np.bool_).sum())


def _replay_task(request: ReplayTaskRequest) -> ReplayTaskResult:
    """Replay one LIBERO task in a worker that exclusively owns its files and env."""

    task_suite = benchmark.get_benchmark_dict()[request.libero_task_suite]()
    task = task_suite.get_task(request.task_id)
    env, task_description = get_libero_env(task, resolution=request.image_resolution)
    task_key = task_description.replace(" ", "_")
    metadata = {task_key: dict(request.existing_metadata.get(task_key, {}))}
    source_path = Path(request.libero_raw_data_dir) / f"{task.name}_demo.hdf5"
    if not source_path.is_file():
        raise FileNotFoundError(f"Cannot find raw data file {source_path}.")
    output_path = Path(request.libero_target_dir) / f"{task.name}_demo.hdf5"
    num_replays = num_successes = num_noops = 0

    try:
        with (
            h5py.File(source_path, "r") as source_file,
            h5py.File(output_path, "a" if request.resume else "w") as output_file,
        ):
            source = source_file["data"]
            output_file.attrs["noop_marking_scheme"] = NOOP_MARKING_SCHEME
            output_file.attrs["noop_keep_noops"] = request.keep_noops
            destination = output_file.require_group("data")

            for index in range(len(source.keys())):
                episode_key = f"demo_{index}"
                complete = episode_key in destination and _complete_demo_group(
                    destination[episode_key]
                )
                existing = metadata[task_key].get(episode_key)
                if request.resume and existing is not None:
                    success = bool(existing.get("success", False))
                    if not success or complete:
                        num_replays += 1
                        num_successes += int(success)
                        continue
                if request.resume and episode_key in destination and not complete:
                    del destination[episode_key]

                demo_data = source[episode_key]
                orig_actions = demo_data["actions"][()]
                orig_states = demo_data["states"][()]
                env.reset()
                env.set_init_state(orig_states[0])
                done = False
                for _ in range(10):
                    obs, _, done, _ = env.step(get_libero_dummy_action())

                states: list[np.ndarray] = []
                actions: list[np.ndarray] = []
                ee_states: list[np.ndarray] = []
                gripper_states: list[np.ndarray] = []
                joint_states: list[np.ndarray] = []
                robot_states: list[np.ndarray] = []
                agentview_images: list[np.ndarray] = []
                eye_in_hand_images: list[np.ndarray] = []
                noop_mask: list[bool] = []
                source_indices: list[int] = []
                previous_action = None

                for action_index, action in enumerate(orig_actions):
                    action_is_noop = is_noop_action(action, previous_action)
                    if action_is_noop:
                        num_noops += 1
                        if not request.keep_noops:
                            continue
                    else:
                        previous_action = action
                    noop_mask.append(bool(action_is_noop))
                    source_indices.append(int(action_index))

                    if not states:
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
                    obs, _, done, _ = env.step(action.tolist())

                if done and actions:
                    _write_successful_episode(
                        destination,
                        episode_key,
                        orig_actions=orig_actions,
                        states=states,
                        actions=actions,
                        robot_states=robot_states,
                        gripper_states=gripper_states,
                        joint_states=joint_states,
                        ee_states=ee_states,
                        agentview_images=agentview_images,
                        eye_in_hand_images=eye_in_hand_images,
                        noop_mask=noop_mask,
                        source_indices=source_indices,
                        keep_noops=request.keep_noops,
                    )
                    num_successes += 1
                num_replays += 1
                metadata[task_key][episode_key] = {
                    "success": bool(done),
                    "initial_state": orig_states[0].tolist(),
                }
                write_task_metadata_shard(
                    request.shard_dir,
                    task_id=request.task_id,
                    metadata=metadata,
                )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return ReplayTaskResult(
        task_id=request.task_id,
        task_description=task_description,
        output_path=str(output_path),
        metadata=metadata,
        num_replays=num_replays,
        num_successes=num_successes,
        num_noops=num_noops,
    )


def main(args) -> None:
    """Regenerate one suite, optionally replaying independent tasks in parallel."""

    suite_name = str(args.libero_task_suite)
    target_dir = Path(args.libero_target_dir)
    keep_noops = bool(getattr(args, "keep_noops", False))
    resume = bool(getattr(args, "resume", False))
    num_workers = int(getattr(args, "num_workers", 1))
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    print(f"Regenerating {suite_name} dataset with {num_workers} task worker(s)!")
    if keep_noops:
        print("No-op actions will be kept and marked with data/demo_*/noop_mask.")
    else:
        print("No-op actions will be filtered after marking.")
    if target_dir.is_dir() and not resume:
        user_input = input(
            f"Target directory already exists at path: {target_dir}\n"
            "Enter 'y' to overwrite the directory, or anything else to exit: "
        )
        if user_input != "y":
            return
    elif target_dir.is_dir():
        print(f"Resuming existing target directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    metainfo_path = Path(args.metainfo_json_out or f"{suite_name}_metainfo.json")
    shard_dir = target_dir / ".metainfo_shards"
    metadata = load_resume_metadata(metainfo_path, shard_dir) if resume else {}
    atomic_write_json(metainfo_path, metadata)

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        valid = ", ".join(sorted(benchmark_dict))
        raise ValueError(f"unknown LIBERO suite {suite_name!r}; valid: {valid}")
    num_tasks = int(benchmark_dict[suite_name]().n_tasks)
    requests = [
        ReplayTaskRequest(
            task_id=task_id,
            libero_task_suite=suite_name,
            libero_raw_data_dir=str(args.libero_raw_data_dir),
            libero_target_dir=str(target_dir),
            image_resolution=int(args.image_resolution),
            keep_noops=keep_noops,
            resume=resume,
            existing_metadata=metadata,
            shard_dir=str(shard_dir),
        )
        for task_id in range(num_tasks)
    ]
    totals = ReplayTotals()
    progress = ProgressReporter(num_tasks, "regenerate no-op filter", unit="task")
    for result in iter_task_results(requests, num_workers=num_workers, worker=_replay_task):
        commit_task_result(metainfo_path, shard_dir, result, metadata)
        totals.add(result)
        print(
            f"Saved regenerated demos for task '{result.task_description}' at: {result.output_path}"
        )
        print(totals.summary(), flush=True)
        progress.update()
    progress.close()
    if shard_dir.is_dir() and not any(shard_dir.iterdir()):
        shard_dir.rmdir()
    print(f"Dataset regeneration complete! Saved new dataset at: {target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_path}")


if __name__ == "__main__":
    args = script_namespace("regenerate_libero_dataset_filter_no_op")
    main(args)
