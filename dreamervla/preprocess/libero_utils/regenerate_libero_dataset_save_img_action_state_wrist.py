from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from dreamervla.utils.hydra_config import script_namespace
from dreamervla.utils.progress import ProgressReporter


def recreate_directory(path: Path) -> None:
    if path.exists():
        print(f"Warning: Directory '{path}' already exists. Deleting and recreating it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    print(f"Directory '{path}' created successfully.")


def save_png(image_array: np.ndarray, output_path: Path) -> None:
    image = image_array[::-1, ::-1]
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    Image.fromarray(image).save(output_path)


def parse_args() -> SimpleNamespace:
    return script_namespace("regenerate_libero_dataset_save_img_action_state_wrist")


def main(args: SimpleNamespace) -> None:
    import h5py
    from libero.libero import benchmark

    print(f"Regenerating {args.libero_task_suite} dataset!")
    print(f"Image resolution argument: {args.image_resolution}")
    print(f"Raw data dir: {args.raw_data_dir}")
    print(f"Save dir: {args.save_dir}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    save_dir = Path(args.save_dir)
    raw_data_dir = Path(args.raw_data_dir)
    recreate_directory(save_dir)

    tasks_pbar = ProgressReporter(num_tasks_in_suite, "save img/action/state", unit="task")
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        orig_data_path = raw_data_dir / f"{task.name}_demo.hdf5"
        if not orig_data_path.exists():
            raise FileNotFoundError(f"Cannot find raw data file: {orig_data_path}")

        with h5py.File(orig_data_path, "r") as orig_data_file:
            orig_data = orig_data_file["data"]

            cur_task_dir = save_dir / task.name
            recreate_directory(cur_task_dir)

            for demo_name in sorted(orig_data.keys()):
                if not demo_name.startswith("demo_"):
                    continue

                demo_data = orig_data[demo_name]
                trj_idx = int(demo_name.split("_")[1])
                orig_actions = demo_data["actions"][()]
                orig_rewards = (
                    demo_data["rewards"][()]
                    if "rewards" in demo_data
                    else np.zeros(orig_actions.shape[0], dtype=np.float32)
                )
                orig_ee_states = demo_data["obs"]["ee_states"][()]
                orig_gripper_states = demo_data["obs"]["gripper_states"][()]
                orig_robot_states = demo_data["robot_states"][()]
                orig_rgb = demo_data["obs"]["agentview_rgb"][()]
                orig_rgb_wrist = demo_data["obs"]["eye_in_hand_rgb"][()]

                cur_trial_dir = cur_task_dir / f"trj_{trj_idx}"
                action_dir = cur_trial_dir / "action"
                ee_state_dir = cur_trial_dir / "ee_state"
                gripper_state_dir = cur_trial_dir / "gripper_state"
                eef_gripper_state_dir = cur_trial_dir / "eef_gripper_state"
                robot_state_dir = cur_trial_dir / "robot_state"
                reward_dir = cur_trial_dir / "reward"
                img_dir_third_view = cur_trial_dir / "imgs_third_view"
                img_dir_wrist = cur_trial_dir / "imgs_wrist"

                recreate_directory(action_dir)
                recreate_directory(ee_state_dir)
                recreate_directory(gripper_state_dir)
                recreate_directory(eef_gripper_state_dir)
                recreate_directory(robot_state_dir)
                recreate_directory(reward_dir)
                recreate_directory(img_dir_third_view)
                recreate_directory(img_dir_wrist)

                for step_idx in range(orig_actions.shape[0]):
                    action = orig_actions[step_idx]
                    reward = np.asarray(orig_rewards[step_idx], dtype=np.float32)
                    ee_state = orig_ee_states[step_idx]
                    gripper_state = orig_gripper_states[step_idx]
                    robot_state = orig_robot_states[step_idx]
                    combined_state = np.concatenate([ee_state, gripper_state])

                    np.save(action_dir / f"action_{step_idx}.npy", action)
                    np.save(ee_state_dir / f"ee_state_{step_idx}.npy", ee_state)
                    np.save(
                        gripper_state_dir / f"gripper_state_{step_idx}.npy",
                        gripper_state,
                    )
                    np.save(
                        eef_gripper_state_dir / f"eef_gripper_state_{step_idx}.npy",
                        combined_state,
                    )
                    np.save(robot_state_dir / f"robot_state_{step_idx}.npy", robot_state)
                    np.save(reward_dir / f"reward_{step_idx}.npy", reward)

                    save_png(
                        orig_rgb[step_idx],
                        img_dir_third_view / f"image_{step_idx}.png",
                    )
                    save_png(
                        orig_rgb_wrist[step_idx],
                        img_dir_wrist / f"image_{step_idx}.png",
                    )

        tasks_pbar.update()
    tasks_pbar.close()


if __name__ == "__main__":
    main(parse_args())
