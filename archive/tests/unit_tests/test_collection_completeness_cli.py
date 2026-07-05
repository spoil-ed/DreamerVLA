import json

import h5py
import numpy as np


def _write_reward_hidden_pair(reward_path, hidden_path, episodes) -> None:
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(reward_path), "w") as rf, h5py.File(str(hidden_path), "w") as hf:
        rdata = rf.create_group("data")
        hdata = hf.create_group("data")
        for idx, spec in enumerate(episodes):
            key = f"demo_{idx}"
            length = int(spec.get("length", 3))
            rgrp = rdata.create_group(key)
            rgrp.create_dataset("actions", data=np.zeros((length, 7), dtype=np.float32))
            rgrp.create_dataset("dones", data=np.zeros((length,), dtype=np.uint8))
            rgrp.create_dataset("rewards", data=np.zeros((length,), dtype=np.float32))
            rgrp.create_dataset("sparse_rewards", data=np.zeros((length,), dtype=np.uint8))
            rgrp.create_dataset("robot_states", data=np.zeros((length, 9), dtype=np.float32))
            rgrp.create_dataset("states", data=np.zeros((length, 5), dtype=np.float32))
            obs = rgrp.create_group("obs")
            obs.create_dataset("agentview_rgb", data=np.zeros((length, 4, 4, 3), dtype=np.uint8))
            obs.create_dataset("eye_in_hand_rgb", data=np.zeros((length, 4, 4, 3), dtype=np.uint8))
            obs.create_dataset("ee_pos", data=np.zeros((length, 3), dtype=np.float32))
            obs.create_dataset("ee_ori", data=np.zeros((length, 3), dtype=np.float32))
            obs.create_dataset("ee_states", data=np.zeros((length, 6), dtype=np.float32))
            obs.create_dataset("gripper_states", data=np.zeros((length, 2), dtype=np.float32))
            obs.create_dataset("joint_states", data=np.zeros((length, 7), dtype=np.float32))
            rgrp.attrs["num_samples"] = str(length)
            rgrp.attrs["task_id"] = int(spec["task_id"])
            rgrp.attrs["episode_id"] = int(spec["episode_id"])
            rgrp.attrs["complete"] = bool(spec.get("complete", True))
            if spec.get("hidden", True):
                hgrp = hdata.create_group(key)
                hidden_length = int(spec.get("hidden_length", length))
                hgrp.create_dataset(
                    "obs_embedding",
                    data=np.zeros((hidden_length, 8), dtype=np.float16),
                )
                hgrp.attrs["complete"] = bool(spec.get("hidden_complete", True))
        rdata.attrs["num_demos"] = len(episodes)
        hdata.attrs["num_demos"] = len(episodes)


def test_collection_completeness_report_lists_missing_episode_ids(tmp_path) -> None:
    from dreamervla.diagnostics.check_collection_completeness import (
        build_collection_completeness_report,
    )

    reward = tmp_path / "reward"
    hidden = tmp_path / "hidden"
    _write_reward_hidden_pair(
        reward / "ray_shard_000.hdf5",
        hidden / "ray_shard_000.hdf5",
        [
            {"task_id": 0, "episode_id": 0},
            {"task_id": 0, "episode_id": 1, "hidden_length": 1},
            {"task_id": 0, "episode_id": 2},
        ],
    )

    report = build_collection_completeness_report(
        reward,
        hidden,
        target_episodes=4,
        num_tasks=1,
        task_ids=[0],
    )

    assert report["complete"] is False
    assert report["complete_episode_ids"] == {"0": [0, 2]}
    assert report["missing_episode_ids"] == {"0": [1, 3]}


def test_collection_completeness_cli_json_returns_nonzero_for_missing(
    tmp_path, capsys
) -> None:
    from dreamervla.diagnostics.check_collection_completeness import main

    reward = tmp_path / "reward"
    hidden = tmp_path / "hidden"
    _write_reward_hidden_pair(
        reward / "ray_shard_000.hdf5",
        hidden / "ray_shard_000.hdf5",
        [{"task_id": 0, "episode_id": 0}],
    )

    code = main(
        [
            "--reward-dir",
            str(reward),
            "--hidden-dir",
            str(hidden),
            "--target-episodes",
            "2",
            "--num-tasks",
            "1",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["missing_episode_ids"] == {"0": [1]}
