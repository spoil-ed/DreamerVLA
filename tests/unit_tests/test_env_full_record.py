"""Test that DreamerVLAOnlineTrainEnv.full_record() exposes LIBERO HDF5 schema fields.

Skipped gracefully when LIBERO is not installed, matching the pattern in
tests/unit_tests/test_rollout_field_mapping.py.
"""

from __future__ import annotations

import numpy as np
import pytest

_LIBERO_AVAILABLE = False
try:
    import libero  # noqa: F401
    _LIBERO_AVAILABLE = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not _LIBERO_AVAILABLE,
    reason="LIBERO not installed",
)


def test_full_record_exposes_libero_schema_fields():
    from dreamervla.envs.libero.libero_env import (
        DreamerVLAOnlineTrainEnv,
        DreamerVLAOnlineTrainEnvConfig,
    )

    cfg = DreamerVLAOnlineTrainEnvConfig(
        task_suite_name="libero_goal", task_id=0, resolution=256, full_record=True
    )
    env = DreamerVLAOnlineTrainEnv(cfg)
    env.reset()
    rec = env.full_record()

    assert rec["agentview_rgb"].shape == (256, 256, 3) and rec["agentview_rgb"].dtype == np.uint8
    assert rec["eye_in_hand_rgb"].shape == (256, 256, 3) and rec["eye_in_hand_rgb"].dtype == np.uint8
    assert rec["ee_pos"].shape == (3,) and rec["ee_ori"].shape == (3,)
    assert rec["ee_states"].shape == (6,) and rec["gripper_states"].shape == (2,)
    assert rec["joint_states"].shape == (7,) and rec["robot_states"].shape == (9,)
    assert rec["states"].ndim == 1 and rec["init_state"].shape == rec["states"].shape

    # Verify gripper-first layout end-to-end
    assert np.array_equal(rec["robot_states"][0:2], rec["gripper_states"])
    assert np.array_equal(rec["robot_states"][2:5], rec["ee_pos"])

    env.close()
