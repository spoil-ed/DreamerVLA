import h5py
import numpy as np
import pytest
import torch

from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter
from dreamervla.runners.offline_seed import seed_replay_from_offline
from dreamervla.runners.online_replay import OnlineReplay

_HIDDEN_TOKEN_CONFIG = {
    "action_head_type": "oft_discrete_token",
    "obs_hidden_source": "hidden_token",
    "hidden_key": "obs_embedding",
    "token_count": 256,
    "token_dim": 4096,
    "hidden_dim": 1_048_576,
    "obs_embedding_shape": [256, 4096],
    "hidden_storage_format": "tokenized",
    "num_images_in_input": 1,
    "patches_per_image": 256,
    "history": 1,
    "include_state": False,
    "sidecar_schema_version": 1,
    "required_demo_datasets": ["obs_embedding"],
}


def _demo_steps(T, success, *, include_lang=False):
    steps = []
    for t in range(T):
        ee_pos = np.full(3, 1.0 + t, np.float64)
        ee_ori = np.full(3, 10.0 + t, np.float64)
        gripper = np.full(2, 100.0 + t, np.float64)
        steps.append({
            "actions": np.full(7, t, np.float64),
            "rewards": np.float32(0.0),
            "sparse_rewards": np.uint8(1 if (success and t == T - 1) else 0),
            "dones": np.uint8(1 if t == T - 1 else 0),
            "robot_states": np.zeros(9, np.float64),
            "states": np.zeros(5, np.float64),
            "obs": {
                "agentview_rgb": np.zeros((256, 256, 3), np.uint8),
                "eye_in_hand_rgb": np.zeros((256, 256, 3), np.uint8),
                "ee_pos": ee_pos, "ee_ori": ee_ori,
                "ee_states": np.zeros(6, np.float64), "gripper_states": gripper,
                "joint_states": np.zeros(7, np.float64),
            },
            "obs_embedding": np.broadcast_to(
                np.asarray(t, dtype=np.float16), (256, 4096)
            ),
        })
        if include_lang:
            steps[-1]["lang_emb"] = np.arange(12, dtype=np.float32) + 0.5
    return steps


def _write_fixture(reward_dir, hidden_dir):
    with RolloutDumpWriter(reward_dir, hidden_dir, "r0_shard.hdf5") as w:
        w.write_demo(
            index=0,
            steps=_demo_steps(6, success=True),
            preprocess_config=_HIDDEN_TOKEN_CONFIG,
            task_id=2,
            episode_id=0,
        )
        w.write_demo(
            index=1,
            steps=_demo_steps(6, success=False),
            preprocess_config=_HIDDEN_TOKEN_CONFIG,
            task_id=5,
            episode_id=0,
        )


def test_seed_replay_reads_all_demos_with_task_id(tmp_path):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    _write_fixture(rdir, hdir)
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(2, 5), rank=0)
    n = seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)
    assert n == 2                                    # two episodes added
    assert replay.num_transitions == 12             # 6 + 6
    assert {record["source"] for record in replay.episodes} == {"coldstart"}
    batch = replay.sample(2)
    assert set(int(t) for t in batch["task_ids"].tolist()) <= {2, 5}
    assert set(batch["replay_source_ids"].tolist()) == {0}
    assert batch["obs_embedding"].shape[-2:] == (256, 4096)
    assert replay.episodes[0]["episode"][0]["obs_embedding"].dtype == np.float16
    assert batch["obs_embedding"].dtype == torch.float16


def test_seed_replay_accepts_structurally_complete_official_reference(tmp_path):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    _write_fixture(rdir, hdir)
    with h5py.File(rdir / "r0_shard.hdf5", "a") as handle:
        for demo in handle["data"].values():
            del demo.attrs["complete"]
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(2, 5), rank=0)

    with pytest.raises(ValueError, match="reward demo is not marked complete"):
        seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)
    assert replay.num_transitions == 0

    n = seed_replay_from_offline(
        replay,
        data_dir=rdir,
        hidden_dir=hdir,
        require_reference_complete=False,
    )

    assert n == 2
    assert replay.num_transitions == 12


def test_seed_replay_threads_proprio_and_language_sidecar(tmp_path):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    with RolloutDumpWriter(rdir, hdir, "r0_shard.hdf5") as w:
        w.write_demo(
            index=0,
            steps=_demo_steps(6, success=True, include_lang=True),
            preprocess_config=_HIDDEN_TOKEN_CONFIG,
            task_id=2,
            episode_id=0,
        )
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(2,), rank=0)

    n = seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)

    assert n == 1
    first = replay.episodes[0]["episode"][0]
    np.testing.assert_allclose(
        first["proprio"],
        np.array([1, 1, 1, 10, 10, 10, 100, 100], dtype=np.float32),
    )
    np.testing.assert_allclose(first["lang_emb"], np.arange(12, dtype=np.float32) + 0.5)
    batch = replay.sample(1, include_images=False)
    assert batch["proprio"].shape == (1, 4, 8)
    assert batch["proprio"].dtype == torch.float32
    assert batch["lang_emb"].shape == (1, 12)
    assert batch["lang_emb"].dtype == torch.float32


def test_seed_replay_marks_success_only_at_terminal_step(tmp_path):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    _write_fixture(rdir, hdir)
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(2, 5), rank=0)

    seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)

    success_record = next(record for record in replay.episodes if record["task_id"] == 2)
    assert success_record["success"] is True
    assert success_record["finish_step"] == 6
    assert [bool(step["success"]) for step in success_record["episode"]] == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_seed_replay_caps_episodes_per_task(tmp_path):
    # The online-replay seed caps per-task episodes so the bounded buffer gets coverage
    # without overflowing. 3 demos for task 2, 1 for task 5; cap=2 -> 2 + 1 = 3 added.
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    with RolloutDumpWriter(rdir, hdir, "r0_shard.hdf5") as w:
        w.write_demo(index=0, steps=_demo_steps(6, success=True), preprocess_config=_HIDDEN_TOKEN_CONFIG, task_id=2, episode_id=0)
        w.write_demo(index=1, steps=_demo_steps(6, success=False), preprocess_config=_HIDDEN_TOKEN_CONFIG, task_id=2, episode_id=1)
        w.write_demo(index=2, steps=_demo_steps(6, success=True), preprocess_config=_HIDDEN_TOKEN_CONFIG, task_id=2, episode_id=2)
        w.write_demo(index=3, steps=_demo_steps(6, success=False), preprocess_config=_HIDDEN_TOKEN_CONFIG, task_id=5, episode_id=0)
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(2, 5), rank=0)
    n = seed_replay_from_offline(
        replay, data_dir=rdir, hidden_dir=hdir, max_episodes_per_task=2
    )
    assert n == 3  # task 2 capped at 2, task 5 has 1


def test_seeded_replay_is_training_ready(tmp_path):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    with RolloutDumpWriter(rdir, hdir, "r0_shard.hdf5") as w:
        index = 0
        for task_id in range(10):
            w.write_demo(
                index=index,
                steps=_demo_steps(4, success=(task_id % 2 == 0)),
                preprocess_config=_HIDDEN_TOKEN_CONFIG,
                task_id=task_id,
                episode_id=0,
            )
            index += 1
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=tuple(range(10)), rank=0)

    n = seed_replay_from_offline(
        replay,
        data_dir=rdir,
        hidden_dir=hdir,
        max_episodes_per_task=1,
    )

    assert n == 10
    assert replay.ready_for_training(
        min_transitions=40,
        task_ids=tuple(range(10)),
        min_episodes_per_task=1,
    ) is True


def test_seed_replay_task_id_fallback(tmp_path):
    # Demo without task_id attr -> use provided default.
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    with RolloutDumpWriter(rdir, hdir, "r0_shard.hdf5") as w:
        w.write_demo(index=0, steps=_demo_steps(6, success=True), preprocess_config=_HIDDEN_TOKEN_CONFIG)   # no task_id
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(0,), rank=0)
    n = seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir, default_task_id=0)
    assert n == 1
    batch = replay.sample(1)
    assert int(batch["task_ids"][0]) == 0


def test_seed_replay_missing_task_id_raises(tmp_path):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    with RolloutDumpWriter(rdir, hdir, "r0_shard.hdf5") as w:
        w.write_demo(index=0, steps=_demo_steps(6, success=True), preprocess_config=_HIDDEN_TOKEN_CONFIG)   # no task_id
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(0,), rank=0)
    with pytest.raises(ValueError, match="task_id"):
        seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)  # no default


def test_seed_replay_preflight_rejects_demo_set_mismatch_without_partial_add(tmp_path):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    _write_fixture(rdir, hdir)
    with h5py.File(hdir / "r0_shard.hdf5", "a") as handle:
        del handle["data/demo_1"]
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(2, 5), rank=0)

    with pytest.raises(ValueError, match="demo set mismatch"):
        seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)

    assert replay.num_transitions == 0
    assert replay.episodes == []


def test_seed_replay_preflight_rejects_length_mismatch_without_partial_add(tmp_path):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    _write_fixture(rdir, hdir)
    with h5py.File(hdir / "r0_shard.hdf5", "a") as handle:
        demo = handle["data/demo_1"]
        del demo["obs_embedding"]
        demo.create_dataset("obs_embedding", shape=(5, 256, 4096), dtype="float16")
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(2, 5), rank=0)

    with pytest.raises(ValueError, match="length mismatch"):
        seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)

    assert replay.num_transitions == 0
    assert replay.episodes == []


def test_seed_replay_preflight_rejects_incomplete_sidecar_demo(tmp_path):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    _write_fixture(rdir, hdir)
    with h5py.File(hdir / "r0_shard.hdf5", "a") as handle:
        handle["data/demo_1"].attrs["complete"] = False
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(2, 5), rank=0)

    with pytest.raises(ValueError, match="complete"):
        seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)

    assert replay.num_transitions == 0
    assert replay.episodes == []


def test_seed_replay_preflight_rejects_missing_reward_field_without_partial_add(
    tmp_path,
):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    _write_fixture(rdir, hdir)
    with h5py.File(rdir / "r0_shard.hdf5", "a") as handle:
        del handle["data/demo_1/sparse_rewards"]
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(2, 5), rank=0)

    with pytest.raises(ValueError, match="sparse_rewards"):
        seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)

    assert replay.num_transitions == 0
    assert replay.episodes == []


def test_seed_replay_preflight_rejects_reward_obs_length_mismatch_without_partial_add(
    tmp_path,
):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    _write_fixture(rdir, hdir)
    with h5py.File(rdir / "r0_shard.hdf5", "a") as handle:
        obs = handle["data/demo_1/obs"]
        del obs["ee_pos"]
        obs.create_dataset("ee_pos", shape=(5, 3), dtype="float32")
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(2, 5), rank=0)

    with pytest.raises(ValueError, match="ee_pos.*length mismatch"):
        seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)

    assert replay.num_transitions == 0
    assert replay.episodes == []


def test_seed_replay_preflight_rejects_incomplete_reward_demo(tmp_path):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    _write_fixture(rdir, hdir)
    with h5py.File(rdir / "r0_shard.hdf5", "a") as handle:
        handle["data/demo_1"].attrs["complete"] = False
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(2, 5), rank=0)

    with pytest.raises(ValueError, match="reward demo.*complete"):
        seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)

    assert replay.num_transitions == 0
    assert replay.episodes == []
