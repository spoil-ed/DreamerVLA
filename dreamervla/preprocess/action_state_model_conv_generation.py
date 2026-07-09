from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from dreamervla.utils.hydra_config import script_namespace


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


def _load_reward(path: str) -> float:
    return float(np.asarray(np.load(path), dtype=np.float32).item())


def _write_json(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def process_libero_data(
    base_dir: str,
    his: int,
    len_action: int,
    task_name_for_output: str,
    resolution: int,
    with_state: bool,
    img_names: list[str],
    output_dir: str,
    with_world_model: bool = False,
) -> None:
    train_convs: list[dict[str, Any]] = []
    val_convs_ind: list[dict[str, Any]] = []
    val_convs_ood: list[dict[str, Any]] = []
    all_convs: list[dict[str, Any]] = []

    train_traj_count = 0
    val_ind_traj_count = 0
    val_ood_traj_count = 0

    os.makedirs(output_dir, exist_ok=True)

    task_list = sorted(os.listdir(base_dir))
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
        if not os.path.isdir(task_path):
            continue

        task_name_readable = task.replace("_", " ")
        trj_list = sorted(os.listdir(task_path))
        split_index_ind = math.ceil(len(trj_list) * 0.9)

        for traj_idx, trj in enumerate(trj_list):
            trj_path = os.path.join(task_path, trj)
            if not os.path.isdir(trj_path):
                continue

            action_path = os.path.join(trj_path, "action")
            imgs_paths = [os.path.join(trj_path, name) for name in img_names]
            reward_path = os.path.join(trj_path, "reward")
            state_path = os.path.join(trj_path, "eef_gripper_state")

            skip_flag = False
            if not os.path.exists(action_path):
                print(f"    Warning: Missing 'action' directory in {trj_path}. Skipping.")
                skip_flag = True

            for imgs_path in imgs_paths:
                if not os.path.exists(imgs_path):
                    print(f"    Warning: Missing 'imgs' directory in {trj_path}. Skipping.")
                    skip_flag = True

            if with_state and not os.path.exists(state_path):
                print(f"    Warning: Missing 'state' directory in {trj_path}. Skipping.")
                skip_flag = True

            if skip_flag:
                continue

            action_map = _collect_indexed_files(action_path, "action_", ".npy")
            image_maps = [
                _collect_indexed_files(imgs_path, "image_", ".png") for imgs_path in imgs_paths
            ]
            state_map = (
                _collect_indexed_files(state_path, "eef_gripper_state_", ".npy")
                if with_state
                else {}
            )
            reward_map = _collect_indexed_files(reward_path, "reward_", ".npy")

            common_indices_sets = [set(action_map.keys())]
            common_indices_sets.extend(set(image_map.keys()) for image_map in image_maps)
            if with_state:
                common_indices_sets.append(set(state_map.keys()))
            if reward_map:
                common_indices_sets.append(set(reward_map.keys()))
            common_indices = sorted(list(set.intersection(*common_indices_sets)))

            if not common_indices:
                print(
                    "    Warning: No matching action/image file pairs found "
                    f"in {trj_path}. Skipping."
                )
                continue

            action_list: list[str] = []
            image_steps: list[list[str]] = []
            state_list: list[str] = []
            reward_list: list[float] = []
            for idx in common_indices:
                action_list.append(action_map[idx])
                image_steps.append([image_map[idx] for image_map in image_maps])
                if with_state:
                    state_list.append(state_map[idx])
                reward_list.append(
                    _load_reward(reward_map[idx])
                    if reward_map
                    else float(idx == common_indices[-1])
                )

            if not image_steps or not action_list:
                print(
                    "    Warning: No valid image/action pairs found in "
                    f"{trj_path} after filtering. Skipping."
                )
                continue

            for step_idx in range(len(action_list)):
                prompt_text = f"Finish the task: {task_name_readable}."

                img_history_start_idx = max(0, step_idx - his + 1)
                action_c = copy.deepcopy(
                    action_list[step_idx : min(step_idx + len_action, len(action_list))]
                )
                if len(action_c) < len_action:
                    continue
                img_c: list[str] = []
                for hist_idx in range(img_history_start_idx, step_idx + 1):
                    img_c.extend(copy.deepcopy(image_steps[hist_idx]))
                target_step_idx = min(step_idx + len_action, len(image_steps) - 1)
                next_obs_images = copy.deepcopy(image_steps[target_step_idx])
                reward_value = float(
                    reward_list[min(step_idx + len_action - 1, len(reward_list) - 1)]
                )

                if with_state:
                    state_c = copy.deepcopy(state_list[step_idx : step_idx + 1])
                    human_val = prompt_text + "<|state|>" * len(state_c) + "<|image|>" * len(img_c)
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

                if task_id < split_index_ood and traj_idx < split_index_ind:
                    train_convs.append(conv)
                elif task_id < split_index_ood and traj_idx >= split_index_ind:
                    val_convs_ind.append(conv)
                else:
                    val_convs_ood.append(conv)
                all_convs.append(conv)

                if not with_world_model:
                    continue
                if step_idx > len(action_list) - his - 1:
                    continue
                historical_indices = list(range(max(step_idx - his + 1, 0), step_idx + 1))
                future_index = step_idx + 1
                if future_index >= len(image_steps):
                    continue

                world_input_images: list[str] = []
                for hist_idx in historical_indices:
                    world_input_images.extend(copy.deepcopy(image_steps[hist_idx]))
                world_target_images = copy.deepcopy(image_steps[future_index])
                world_action = [copy.deepcopy(action_list[idx]) for idx in historical_indices]
                world_reward = float(reward_list[future_index])
                if with_state:
                    world_state = copy.deepcopy(state_list[step_idx : step_idx + 1])
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
                        {"from": "human", "value": world_human_val},
                        {"from": "gpt", "value": "<|image|>" * len(world_target_images)},
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

                if task_id < split_index_ood and traj_idx < split_index_ind:
                    train_convs.append(world_conv)
                elif task_id < split_index_ood and traj_idx >= split_index_ind:
                    val_convs_ind.append(world_conv)
                else:
                    val_convs_ood.append(world_conv)
                all_convs.append(world_conv)

            if task_id < split_index_ood and traj_idx < split_index_ind:
                train_traj_count += 1
            elif task_id < split_index_ood and traj_idx >= split_index_ind:
                val_ind_traj_count += 1
            else:
                val_ood_traj_count += 1

    print("-" * 30)
    print("Saving datasets...")

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

    _write_json(train_output_path, train_convs)
    print(f"Saved train conversations to: {train_output_path}")

    _write_json(val_ind_output_path, val_convs_ind)
    print(f"Saved val_ind conversations to: {val_ind_output_path}")

    _write_json(val_ood_output_path, val_convs_ood)
    print(f"Saved val_ood conversations to: {val_ood_output_path}")

    print("\n--- Final Summary ---")
    print(f"Train trajectories: {train_traj_count}, conversations: {len(train_convs)}")
    print(
        "Validation In-Distribution trajectories: "
        f"{val_ind_traj_count}, conversations: {len(val_convs_ind)}"
    )
    print(
        "Validation Out-of-Distribution trajectories: "
        f"{val_ood_traj_count}, conversations: {len(val_convs_ood)}"
    )
    print("---------------------")


def main() -> None:
    args = script_namespace("action_state_model_conv_generation")
    process_libero_data(
        base_dir=args.base_dir,
        his=args.his,
        len_action=args.len_action,
        task_name_for_output=args.task_name,
        resolution=args.resolution,
        with_state=args.with_state,
        img_names=args.img_names,
        output_dir=args.output_dir,
        with_world_model=bool(args.with_world_model),
    )


if __name__ == "__main__":
    main()
