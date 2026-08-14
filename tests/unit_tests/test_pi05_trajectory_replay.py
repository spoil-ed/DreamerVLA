from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class _FakePrefixEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode_raw_observation_prefix_batch(self, observations):
        self.calls += 1
        values = [float(item["observation/state"][0]) for item in observations]
        return torch.tensor(values, dtype=torch.float32)[:, None, None].expand(-1, 2, 4)


def _step(index: int) -> dict:
    zeros = np.zeros
    return {
        "actions": np.full(7, index, dtype=np.float64),
        "rewards": np.float32(0),
        "sparse_rewards": np.uint8(0),
        "dones": np.uint8(index == 5),
        "robot_states": zeros(9),
        "states": zeros(8),
        "obs": {
            "agentview_rgb": np.full((2, 2, 3), index, dtype=np.uint8),
            "eye_in_hand_rgb": np.full((2, 2, 3), index + 1, dtype=np.uint8),
            "ee_pos": np.asarray([index, 0, 0], dtype=np.float64),
            "ee_ori": zeros(3),
            "ee_states": zeros(6),
            "gripper_states": zeros(2),
            "joint_states": zeros(7),
        },
    }


def _write_rgb_trajectory(root: Path, episode_id: int, task_id: int) -> None:
    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter

    with RolloutDumpWriter(
        root,
        root / "unused-latent",
        f"traj_{episode_id}.hdf5",
        write_hidden_sidecar=False,
    ) as writer:
        writer.write_demo(
            0,
            [_step(index) for index in range(6)],
            task_id=task_id,
            episode_id=episode_id,
            task_description=f"task {task_id}",
            episode_success=False,
        )


def test_pi05_rgb_replay_uses_every_failure_window_and_rank_shards(tmp_path: Path) -> None:
    from dreamervla.runtime.pi05_trajectory_replay import Pi05TrajectoryReplay

    reward = tmp_path / "reward"
    for episode_id in range(3):
        _write_rgb_trajectory(reward, episode_id, episode_id % 2)

    encoder = _FakePrefixEncoder()
    replay = Pi05TrajectoryReplay(
        data_dir=reward,
        sequence_length=3,
        encoder=encoder,
        encode_batch_size=2,
        rank=0,
        world_size=2,
        seed=3,
        task_ids=(0, 1),
        rotate_images_180=False,
    )

    assert replay.raw_window_count == 12
    assert replay.sampleable_window_count() == 16
    assert replay.num_transitions == 18
    seen: set[tuple[int, int]] = set()
    for _ in range(4):
        batch = replay.sample(2, include_images=False)
        assert batch["obs_embedding"].shape == (2, 3, 2, 4)
        seen.update(
            zip(
                batch["episode_ids"].tolist(),
                batch["start_indices"].tolist(),
                strict=True,
            )
        )
    assert len(seen) == 8
    for episode_id in {item[0] for item in seen}:
        assert {start for ep, start in seen if ep == episode_id} == {0, 1, 2, 3}
    assert encoder.calls == 6  # two local trajectories, 3 microbatches each
