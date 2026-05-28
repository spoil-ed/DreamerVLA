from __future__ import annotations

import h5py
import numpy as np

from scripts.preprocess_remaining_steps_reward import (
    SCHEME_NAME,
    _copy_file_with_remaining_rewards,
    remaining_steps_reward,
)


class _Args:
    success_threshold = 0.5
    failure_value = 0.0
    min_value = 0.0
    max_value = 1.0
    compression = None


def test_remaining_steps_reward_success_episode_is_monotonic() -> None:
    rewards = np.array([0, 0, 0, 0, 1], dtype=np.float32)

    shaped, info = remaining_steps_reward(rewards)

    assert info["success"] is True
    assert info["success_index"] == 4
    assert shaped.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_remaining_steps_reward_failure_episode_is_low_constant() -> None:
    rewards = np.zeros(5, dtype=np.float32)

    shaped, info = remaining_steps_reward(rewards, success=False, failure_value=-1.0)

    assert info["success"] is False
    assert info["success_index"] == -1
    assert shaped.tolist() == [-1.0] * 5


def test_hdf5_rewrite_preserves_sparse_rewards_and_metadata(tmp_path) -> None:
    source = tmp_path / "pick_object_demo.hdf5"
    output = tmp_path / "out" / source.name
    with h5py.File(source, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((4, 7), dtype=np.float32))
        demo.create_dataset("dones", data=np.array([0, 0, 0, 1], dtype=np.uint8))
        demo.create_dataset("rewards", data=np.array([0, 0, 0, 1], dtype=np.uint8))

    record = _copy_file_with_remaining_rewards(
        source,
        output,
        metainfo={},
        args=_Args(),
    )

    assert record["demos"] == 1
    assert record["successes"] == 1
    with h5py.File(output, "r") as handle:
        demo = handle["data"]["demo_0"]
        assert demo["sparse_rewards"][:].tolist() == [0, 0, 0, 1]
        np.testing.assert_allclose(demo["rewards"][:], [0.0, 1 / 3, 2 / 3, 1.0])
        assert demo["rewards"].attrs["scheme"] == SCHEME_NAME
        assert demo.attrs["reward_success"] == np.bool_(True)
