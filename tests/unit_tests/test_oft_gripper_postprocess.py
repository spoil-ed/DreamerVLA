"""Regression guard for the OpenVLA-OFT LIBERO gripper post-process.

The OFT model gripper output is in [0, 1]; LIBERO needs it mapped to {-1, +1}
via ``g = sign(2g-1) * -1`` (normalize_gripper_action binarize + invert). Without
this every rollout/collector path drives the gripper wrong and task success drops
to ~0. ``process_action`` is the single shared implementation reused by the eval
core and every collector path, so it is the thing to lock down.
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from dreamervla.runtime.libero_vla_eval_action import EmbodiedEvalActionMixin
from dreamervla.runtime.oft_collect import process_action, process_action_batch


class _EvalActionHarness(EmbodiedEvalActionMixin):
    def __init__(self, action_postprocess: str = "none") -> None:
        self.cfg = OmegaConf.create(
            {
                "eval": {
                    "dreamer_unnorm_actions": False,
                    "action_postprocess": action_postprocess,
                }
            }
        )
        self._dreamer_clip_actions = True


@pytest.mark.parametrize(
    "g_in, g_out",
    [
        (1.0, -1.0),   # fully closed model output -> +1 after 2g-1 -> sign +1 -> *-1 = -1
        (0.9, -1.0),
        (0.51, -1.0),
        (0.49, 1.0),
        (0.1, 1.0),
        (0.0, 1.0),    # fully open model output -> -1 after 2g-1 -> sign -1 -> *-1 = +1
    ],
)
def test_gripper_binarize_and_invert(g_in: float, g_out: float) -> None:
    out = process_action([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, g_in])
    assert float(out[-1]) == g_out


def test_non_gripper_dims_are_untouched() -> None:
    raw = [0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.9]
    out = process_action(raw)
    np.testing.assert_allclose(out[:6], np.asarray(raw[:6], dtype=np.float32))


def test_shape_dtype_and_no_mutation_of_input() -> None:
    raw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8], dtype=np.float32)
    raw_copy = raw.copy()
    out = process_action(raw)
    assert out.shape == (7,)
    assert out.dtype == np.float32
    # input must not be mutated in place (callers reuse the raw chunk action)
    np.testing.assert_array_equal(raw, raw_copy)


def test_output_gripper_is_binary() -> None:
    # whatever the input gripper, the LIBERO command must be exactly +/-1 (or 0 at the
    # exact 0.5 boundary where sign(0)=0); never a passthrough of the raw [0,1] value.
    for g in np.linspace(0.0, 1.0, 21):
        out = process_action([0, 0, 0, 0, 0, 0, float(g)])
        assert float(out[-1]) in (-1.0, 0.0, 1.0)


def test_batch_postprocess_matches_per_action_and_does_not_mutate_input() -> None:
    raw = np.array(
        [
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.25], [1, 2, 3, 4, 5, 6, 0.75]],
            [[-1, -2, -3, -4, -5, -6, 0.5], [0, 0, 0, 0, 0, 0, 0.49]],
        ],
        dtype=np.float32,
    )
    raw_copy = raw.copy()

    out = process_action_batch(raw)

    expected = np.stack(
        [process_action(action) for action in raw.reshape(-1, raw.shape[-1])],
        axis=0,
    ).reshape(raw.shape)
    assert out.shape == raw.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected)
    np.testing.assert_array_equal(raw, raw_copy)


def test_eval_action_postprocess_defaults_to_no_gripper_mapping() -> None:
    harness = _EvalActionHarness()

    out = harness._dreamer_policy_raw_to_env_action(
        np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
    )

    assert float(out[-1]) == pytest.approx(0.9)


def test_eval_openvla_oft_action_postprocess_reuses_shared_gripper_mapping() -> None:
    harness = _EvalActionHarness("openvla_oft")
    raw = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)

    out = harness._dreamer_policy_raw_to_env_action(raw)

    np.testing.assert_allclose(out, process_action(raw))
