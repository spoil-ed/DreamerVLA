from __future__ import annotations

import numpy as np

from dreamervla.workers.inference._test_rollout_stub import HIDDEN_DIM
from dreamervla.workers.inference.rollout_inference_worker import RolloutInferenceWorker


def _cfg() -> dict:
    return {
        "device": "cpu",
        "action_dim": 7,
        "decoder": {
            "target": "dreamervla.workers.inference._test_rollout_stub:StubRolloutBundle"
        },
    }


def test_forward_batch_returns_action_and_hidden() -> None:
    w = RolloutInferenceWorker(_cfg(), {}, num_envs=3)
    w.init()
    out = w.forward_batch([{"seed": 10}, {"seed": 20}, {"seed": 30}], [0, 1, 2])
    assert set(out) == {"actions", "obs_embedding"}
    assert out["actions"][0].shape == (7,) and out["actions"][0].dtype == np.float32
    assert out["obs_embedding"][1].shape == (HIDDEN_DIM,)
    assert float(out["actions"][1][0]) == 20.0
    assert float(out["obs_embedding"][2][0]) == 30.0


def test_reset_states_clears_only_named_envs() -> None:
    w = RolloutInferenceWorker(_cfg(), {}, num_envs=2)
    w.init()
    w.forward_batch([{"seed": 1}, {"seed": 2}], [0, 1])
    w.reset_states([1])
    assert w._extractors[0].n == 1
    assert w._extractors[1].n == 0


def test_batched_equals_sequential_for_independent_envs() -> None:
    cfg = _cfg()
    w = RolloutInferenceWorker(cfg, {}, num_envs=2)
    w.init()
    batched = w.forward_batch([{"seed": 5}, {"seed": 9}], [0, 1])

    w2 = RolloutInferenceWorker(cfg, {}, num_envs=2)
    w2.init()
    a = w2.forward_batch([{"seed": 5}], [0])
    b = w2.forward_batch([{"seed": 9}], [1])

    assert float(batched["actions"][0][0]) == float(a["actions"][0][0])
    assert float(batched["obs_embedding"][1][0]) == float(b["obs_embedding"][0][0])


def test_forward_batch_applies_gripper_postprocess() -> None:
    # StubRolloutBundle returns action_chunk[0] = [seed]*7, so the gripper dim is `seed`.
    # The ray path must apply process_action (g -> sign(2g-1)*-1) here, or LIBERO grasping
    # (hence success) fails — the exact bug this guards. Non-gripper dims stay raw.
    w = RolloutInferenceWorker(_cfg(), {}, num_envs=2)
    w.init()
    out = w.forward_batch([{"seed": 20}, {"seed": 0}], [0, 1])
    assert float(out["actions"][0][-1]) == -1.0  # g=20 -> sign(39)*-1 = -1
    assert float(out["actions"][1][-1]) == 1.0   # g=0  -> sign(-1)*-1 = +1
    assert float(out["actions"][0][0]) == 20.0   # non-gripper dim untouched
