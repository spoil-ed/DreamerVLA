from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np

from dreamervla.preprocess.preprocess_remaining_steps_reward import (
    _copy_file_with_remaining_rewards,
)


def test_remaining_reward_output_is_training_ready(tmp_path) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "out" / source.name
    with h5py.File(source, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset(
            "rewards",
            data=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )

    _copy_file_with_remaining_rewards(
        source,
        output,
        metainfo={},
        args=SimpleNamespace(
            success_threshold=0.5,
            failure_value=0.0,
            min_value=0.0,
            max_value=1.0,
            compression="none",
        ),
    )

    with h5py.File(output, "r") as handle:
        assert bool(handle.attrs["complete"]) is True
        assert bool(handle["data/demo_0"].attrs["complete"]) is True
        assert "sparse_rewards" in handle["data/demo_0"]
