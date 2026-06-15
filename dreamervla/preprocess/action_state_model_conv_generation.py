import argparse  # Import the argparse module
import copy
import json
import math
import os

import numpy as np  # Still included, though not directly used for numpy operations on data here


def _collect_indexed_files(directory: str, prefix: str, suffix: str) -> dict[int, str]:
    indexed_files: dict[int, str] = {}
    if not os.path.isdir(directory):
        return indexed_files
    for filename in os.listdir(directory):
        if not (filename.startswith(prefix) and filename.endswith(suffix)):
            continue
        idx_text = filename[len(prefix) : -len(suffix)]
        try:
            idx = int(idx_text)
        except ValueError:
            continue
        indexed_files[idx] = os.path.join(directory, filename)
    return indexed_files


def process_libero_data(
    base_dir: str,
    his: int,
    len_action: int,
    task_name_for_output: str,
    resolution: int,
    with_state: bool,
    img_names: list,
    output_dir: str,
    with_world_model: bool = False,
):
    """
    Processes Libero robot trajectory data to create conversational datasets for
    training and validation (in-distribution and out-of-distribution).

    Args:
        base_dir (str): The base directory where the Libero datasets are located.
        his (int): The number of historical image frames to include in each conversation.
        len_action (int): The number of future action steps to predict.
        task_name_for_output (str): A string used in the output JSON file names to
                                    identify the task type (e.g., 'goal', 'object').
        resolution (int): The image resolution, used in the output JSON file names.
        output_dir (str): The directory where the generated JSON dataset files and
                          the summary JSON file will be saved.
    """

    train_convs = []
    val_convs_ind = []
    val_convs_ood = []
    all_convs = []

    train_traj_count = 0
    val_ind_traj_count = 0
    val_ood_traj_count = 0

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    task_list = sorted(os.listdir(base_dir))
    # Split for OOD tasks (10% for OOD validation, i.e., first 90% for train/val_ind)
    split_index_ood = math.ceil(len(task_list) * 0.9)

    print(f"Processing data from: {base_dir}")
    print(f"Historical frames (his): {his}")
    print(f"Action prediction length (len_action): {len_action}")
    print(f"Output task name: {task_name_for_output}")
    print(f"Resolution: {resolution}")
    print(f"With state: {with_state}")
    print(f"Image list: {img_names}")
    print(f"Output directory: {output_dir}")
    print("-" * 30)

    for task_id, task in enumerate(task_list):
        task_path = os.path.join(base_dir, task)
        # Assuming task names are like "put_apple_in_bowl" -> "put apple in bowl"
        task_name_readable = task.replace("_", " ")

        trj_list = sorted(os.listdir(task_path))
        # Split for In-Distribution validation within each task (10% for IND validation)
        split_index_ind = math.ceil(len(trj_list) * 0.9)

        for i, trj in enumerate(trj_list):
            trj_path = os.path.join(task_path, trj)
            action_path = os.path.join(trj_path, "action")
            # imgs_path = os.path.join(trj_path, 'imgs')
            imgs_paths = [os.path.join(trj_path, name) for name in img_names]
            reward_path = os.path.join(trj_path, "reward")
            if with_state:
                state_path = os.path.join(
                    trj_path, "eef_gripper_state"
                )  # TODO: 根据需要统一修改 eef_gripper_state

            skip_flag = False
            # Check if action and imgs directories exist
            if not os.path.exists(action_path):
                print(
                    f"    Warning: Missing 'action' directory in {trj_path}. Skipping."
                )
                skip_flag = True

            for imgs_path in imgs_paths:
                if not os.path.exists(imgs_path):
                    print(
                        f"    Warning: Missing 'imgs' directory in {trj_path}. Skipping."
                    )
                    skip_flag = True

            if with_state and (not os.path.exists(state_path)):
                print(
                    f"    Warning: Missing 'state' directory in {trj_path}. Skipping."
                )
                skip_flag = True

            if skip_flag:
                continue

            action_map = _collect_indexed_files(action_path, "action_", ".npy")
            image_maps = [
                _collect_indexed_files(imgs_path, "image_", ".png")
                for imgs_path in imgs_paths
            ]
            state_map = (
                _collect_indexed_files(state_path, "eef_gripper_state_", ".npy")
                if with_state
                else {}
            )
            reward_map = _collect_indexed_files(reward_path, "reward_", ".npy")

            common_indices_sets = [set(action_map.keys())]
            common_indices_sets.extend(
                set(image_map.keys()) for image_map in image_maps
            )
            if with_state:
                common_indices_sets.append(set(state_map.keys()))
            if reward_map:
                common_indices_sets.append(set(reward_map.keys()))
            common_indices = sorted(list(set.intersection(*common_indices_sets)))

            if not common_indices:
                print(
                    f"    Warning: No matching action/image file pairs found in {trj_path}. Skipping."
                )
                continue

            action_list: list[str] = []
            image_steps: list[list[str]] = []
            state_list: list[str] = []
            reward_list: list[float] = []
            for idx in common_indices:
                action_file = action_map[idx]
                img_files = [image_map[idx] for image_map in image_maps]
                action_list.append(action_file)
                image_steps.append(img_files)
                if with_state:
                    state_list.append(state_map[idx])
                if reward_map:
                    reward_value = float(np.load(reward_map[idx]).astype(np.float32))
                else:
                    # Fallback when reward files are absent: sparse terminal reward.
                    reward_value = 1.0 if idx == common_indices[-1] else 0.0
                reward_list.append(reward_value)

            if not image_steps or not action_list:
                print(
                    f"    Warning: No valid image/action pairs found in {trj_path} after filtering. Skipping."
                )
                continue

            # Generate conversation samples for each step in the trajectory
            for j in range(len(action_list)):
                prompt_text = f"Finish the task: {task_name_readable}."

                # Action task samples
                img_history_start_idx = max(0, j - his + 1)
                action_c = copy.deepcopy(
                    action_list[j : min(j + len_action, len(action_list))]
                )
                if len(action_c) < len_action:
                    continue
                img_c = []
                for step_idx in range(img_history_start_idx, j + 1):
                    img_c.extend(copy.deepcopy(image_steps[step_idx]))
                target_step_idx = min(j + len_action, len(image_steps) - 1)
                next_obs_images = copy.deepcopy(image_steps[target_step_idx])
                reward_value = float(
                    reward_list[min(j + len_action - 1, len(reward_list) - 1)]
                )

                if with_state:
                    state_c = copy.deepcopy(state_list[j : j + 1])
                    human_val = (
                        prompt_text
                        + "<|state|>" * len(state_c)
                        + "<|image|>" * len(img_c)
                    )
                else:
                    state_c = []
                    human_val = prompt_text + "<|image|>" * len(img_c)

                conv = {
                    "task_name": task,
                    "task_text": task_name_readable,
                    "prompt_text": prompt_text,
                    "conversations": [
                        {"from": "human", "value": human_val},
                        {"from": "gpt", "value": "<|action|>" * len(action_c)},
                    ],
                    "image": img_c,
                    "action": action_c,
                    "state": state_c,
                    "input_image": copy.deepcopy(img_c),
                    "input_action": [],
                    "input_state": copy.deepcopy(state_c),
                    "target_image": [],
                    "target_action": copy.deepcopy(action_c),
                    "target_state": [],
                    "next_obs": {
                        "image": copy.deepcopy(next_obs_images),
                        "state": [],
                    },
                    "reward": reward_value,
                }

                # Assign to appropriate dataset split based on task_id and trajectory_id
                if task_id < split_index_ood and i < split_index_ind:
                    train_convs.append(conv)
                elif task_id < split_index_ood and i >= split_index_ind:
                    val_convs_ind.append(conv)
                else:
                    val_convs_ood.append(conv)
                all_convs.append(conv)

                # World-model task samples
                if not with_world_model:
                    continue
                if j > len(action_list) - his - 1:
                    continue
                historical_indices = list(range(max(j - his + 1, 0), j + 1))
                future_index = j + 1
                if future_index >= len(image_steps):
                    continue

                world_input_images: list[str] = []
                for step_idx in historical_indices:
                    world_input_images.extend(copy.deepcopy(image_steps[step_idx]))
                world_target_images = copy.deepcopy(image_steps[future_index])
                world_action = [
                    copy.deepcopy(action_list[idx]) for idx in historical_indices
                ]
                world_reward = float(reward_list[future_index])
                if with_state:
                    world_state = copy.deepcopy(state_list[j : j + 1])
                    world_human_val = (
                        prompt_text
                        + "<|state|>" * len(world_state)
                        + "<|image|>" * len(world_input_images)
                        + "<|action|>" * len(world_action)
                    )
                else:
                    world_state = []
                    world_human_val = (
                        prompt_text
                        + "<|image|>" * len(world_input_images)
                        + "<|action|>" * len(world_action)
                    )

                world_conv = {
                    "task_name": task,
                    "task_text": task_name_readable,
                    "prompt_text": prompt_text,
                    "conversations": [
                        {
                            "from": "human",
                            "value": world_human_val,
                        },
                        {
                            "from": "gpt",
                            "value": "<|image|>" * len(world_target_images),
                        },
                    ],
                    "image": copy.deepcopy(world_input_images + world_target_images),
                    "action": copy.deepcopy(world_action),
                    "state": copy.deepcopy(world_state),
                    "input_image": copy.deepcopy(world_input_images),
                    "input_action": copy.deepcopy(world_action),
                    "input_state": copy.deepcopy(world_state),
                    "target_image": copy.deepcopy(world_target_images),
                    "target_action": [],
                    "target_state": [],
                    "next_obs": {
                        "image": copy.deepcopy(world_target_images),
                        "state": [],
                    },
                    "reward": world_reward,
                }

                if task_id < split_index_ood and i < split_index_ind:
                    train_convs.append(world_conv)
                elif task_id < split_index_ood and i >= split_index_ind:
                    val_convs_ind.append(world_conv)
                else:
                    val_convs_ood.append(world_conv)
                all_convs.append(world_conv)

            # Increment trajectory counts for statistics
            if task_id < split_index_ood and i < split_index_ind:
                train_traj_count += 1
            elif task_id < split_index_ood and i >= split_index_ind:
                val_ind_traj_count += 1
            else:
                val_ood_traj_count += 1

    print("-" * 30)
    print("Saving datasets...")

    # Define output file names using the parameters
    img_item = "_".join([item.replace("imgs_", "") for item in img_names])
    state_item = "w_state" if with_state else "wo_state"
    train_output_path = os.path.join(
        output_dir,
        f"libero_{task_name_for_output}_his_{his}_train_{img_item}_{state_item}_{len_action}_{resolution}.json",
    )
    val_ind_output_path = os.path.join(
        output_dir,
        f"libero_{task_name_for_output}_his_{his}_val_ind_{img_item}_{state_item}_{len_action}_{resolution}.json",
    )
    val_ood_output_path = os.path.join(
        output_dir,
        f"libero_{task_name_for_output}_his_{his}_val_ood_{img_item}_{state_item}_{len_action}_{resolution}.json",
    )
    # Save training set
    with open(train_output_path, "w") as f:
        json.dump(train_convs, f, indent=2)  # Use indent for readability
    print(f"Saved train conversations to: {train_output_path}")

    # Save validation in-distribution set
    with open(val_ind_output_path, "w") as f:
        json.dump(val_convs_ind, f, indent=2)
    print(f"Saved val_ind conversations to: {val_ind_output_path}")

    # Save validation out-of-distribution set
    with open(val_ood_output_path, "w") as f:
        json.dump(val_convs_ood, f, indent=2)
    print(f"Saved val_ood conversations to: {val_ood_output_path}")

    print("\n--- Final Summary ---")
    print(f"Train trajectories: {train_traj_count}, conversations: {len(train_convs)}")
    print(
        f"Validation In-Distribution trajectories: {val_ind_traj_count}, conversations: {len(val_convs_ind)}"
    )
    print(
        f"Validation Out-of-Distribution trajectories: {val_ood_traj_count}, conversations: {len(val_convs_ood)}"
    )
    print("---------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Process Libero robot trajectory data to create conversational datasets for LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows default values in help message
    )

    # Required argument
    parser.add_argument(
        "--base_dir",
        "-b",
        type=str,
        required=True,
        help="The base directory where the Libero datasets are located",
    )

    # Optional arguments with default values
    parser.add_argument(
        "--his",
        "-H",
        type=int,
        default=2,
        help="The number of historical image frames to include in each conversation (for observation history).",
    )
    parser.add_argument(
        "--len_action",
        "-L",
        type=int,
        default=5,
        help="The number of future action steps to predict.",
    )
    parser.add_argument(
        "--task_name",
        "-T",
        type=str,
        default="goal",
        help="A string used in the output JSON file names to identify the task type (e.g., 'goal', 'object').",
    )
    parser.add_argument(
        "--resolution",
        "-R",
        type=int,
        default=512,
        help="The image resolution, used in the output JSON file names (e.g., 256, 512).",
    )
    parser.add_argument(
        "--with_state", action="store_true", help="If True, with state."
    )
    parser.add_argument(
        "--img_names",
        nargs="+",
        default=["imgs_third_view"],
        choices=["imgs_wrist", "imgs_third_view"],
        help="List of image names to include (imgs_wrist and/or imgs_third_view)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="./generated_libero_convs/",
        help="The directory where the generated JSON dataset files and the summary JSON file will be saved. Will be created if it does not exist.",
    )
    parser.add_argument(
        "--with_world_model",
        action="store_true",
        help="If set, also generate world-model samples.",
    )

    args = parser.parse_args()

    # Call the processing function with parsed arguments
    process_libero_data(
        base_dir=args.base_dir,
        his=args.his,
        len_action=args.len_action,
        task_name_for_output=args.task_name,  # Map 'task_name' from args to 'task_name_for_output' in function
        resolution=args.resolution,
        with_state=args.with_state,
        img_names=args.img_names,
        output_dir=args.output_dir,
        with_world_model=bool(args.with_world_model),
    )


if __name__ == "__main__":
    main()
