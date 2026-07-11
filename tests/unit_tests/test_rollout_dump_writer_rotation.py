"""RotatingRolloutDumpWriter: slice collected demos into N-sized shards.

Mirrors the Ray-side RolloutDumpWorker rotation but as a drop-in
RolloutDumpWriter (same write_demo/close/context-manager interface), so the
no-Ray collector can honour ``collect.demos_per_shard`` too.
"""

from __future__ import annotations

import h5py
import numpy as np

from dreamervla.dataset.rollout_dump_writer import RotatingRolloutDumpWriter


def _steps(T: int) -> list[dict]:
    out = []
    for t in range(T):
        out.append(
            {
                "actions": np.full(7, t, np.float64),
                "rewards": np.float32(0.0),
                "sparse_rewards": np.uint8(0),
                "dones": np.uint8(1 if t == T - 1 else 0),
                "robot_states": np.zeros(9, np.float64),
                "states": np.zeros(5, np.float64),
                "obs": {
                    "agentview_rgb": np.zeros((4, 4, 3), np.uint8),
                    "eye_in_hand_rgb": np.zeros((4, 4, 3), np.uint8),
                    "ee_pos": np.zeros(3, np.float64),
                    "ee_ori": np.zeros(3, np.float64),
                    "ee_states": np.zeros(6, np.float64),
                    "gripper_states": np.zeros(2, np.float64),
                    "joint_states": np.zeros(7, np.float64),
                },
                "obs_embedding": np.broadcast_to(
                    np.asarray(t, dtype=np.float16), (256, 4096)
                ),
            }
        )
    return out


def _write(
    reward,
    hidden,
    preprocess_config,
    *,
    demos_per_shard,
    start_index=0,
    n=5,
    prefix="r0_shard",
):
    with RotatingRolloutDumpWriter(
        reward, hidden, shard_prefix=prefix, demos_per_shard=demos_per_shard,
        start_index=start_index,
    ) as w:
        for i in range(n):
            w.write_demo(
                index=i,  # ignored by the wrapper; it owns shard-local numbering
                steps=_steps(3),
                preprocess_config=preprocess_config,
                data_attrs={"task_suite_name": "libero_goal", "env_name": "x"},
                task_id=i % 2,
                episode_id=i,
            )


def test_rotates_every_n_demos_and_restarts_demo_index(
    tmp_path, input_token_preprocess_config
):
    reward, hidden = tmp_path / "reward", tmp_path / "hidden"
    _write(
        reward,
        hidden,
        input_token_preprocess_config,
        demos_per_shard=2,
        n=5,
    )

    shards = sorted(p.name for p in reward.glob("*.hdf5"))
    assert shards == ["r0_shard_000.hdf5", "r0_shard_001.hdf5", "r0_shard_002.hdf5"]

    # 5 demos / 2 per shard -> 2, 2, 1; demo_<i> restarts at 0 in each shard.
    per_shard = []
    for name in shards:
        with h5py.File(str(reward / name), "r") as f:
            keys = sorted(f["data"].keys())
            per_shard.append(keys)
            assert int(f["data"].attrs["num_demos"]) == len(keys)
    assert per_shard == [["demo_0", "demo_1"], ["demo_0", "demo_1"], ["demo_0"]]


def test_every_shard_is_independently_readable_with_metadata(
    tmp_path, input_token_preprocess_config
):
    reward, hidden = tmp_path / "reward", tmp_path / "hidden"
    _write(
        reward,
        hidden,
        input_token_preprocess_config,
        demos_per_shard=2,
        n=5,
    )

    # preprocess_config.json written once into the hidden dir.
    assert (hidden / "preprocess_config.json").is_file()

    for name in sorted(p.name for p in reward.glob("*.hdf5")):
        with h5py.File(str(reward / name), "r") as f:
            # data-group env meta re-emitted on each shard so it reads standalone.
            assert f["data"].attrs["task_suite_name"] == "libero_goal"
            for key in f["data"]:
                assert "task_id" in f["data"][key].attrs
        # hidden sidecar shares the filename and holds obs_embedding per demo.
        with h5py.File(str(hidden / name), "r") as hf:
            for key in hf["data"]:
                assert hf["data"][key]["obs_embedding"].shape[1:] == (256, 4096)


def test_start_index_offsets_shard_names_for_resume(
    tmp_path, input_token_preprocess_config
):
    reward, hidden = tmp_path / "reward", tmp_path / "hidden"
    _write(
        reward,
        hidden,
        input_token_preprocess_config,
        demos_per_shard=2,
        start_index=3,
        n=3,
    )

    # start_index=3 -> shards 003, 004 (resume appends past existing 000..002).
    assert sorted(p.name for p in reward.glob("*.hdf5")) == [
        "r0_shard_003.hdf5",
        "r0_shard_004.hdf5",
    ]


def test_seed_replay_reads_rotated_shards(tmp_path, input_token_preprocess_config):
    """The warmup loader globs *.hdf5, so sliced shards seed exactly like one shard."""
    from dreamervla.runners.offline_seed import seed_replay_from_offline
    from dreamervla.runners.online_replay import OnlineReplay

    reward, hidden = tmp_path / "reward", tmp_path / "hidden"
    _write(
        reward,
        hidden,
        input_token_preprocess_config,
        demos_per_shard=2,
        n=5,
    )

    replay = OnlineReplay(capacity=1000, sequence_length=2, task_ids=(0, 1), rank=0)
    n = seed_replay_from_offline(replay, data_dir=reward, hidden_dir=hidden, default_task_id=0)
    assert n == 5  # all demos across the three shards
