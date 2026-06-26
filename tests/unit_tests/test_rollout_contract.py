import numpy as np

from dreamervla.workers.inference.rollout_contract import RolloutBatchOutput


def test_rollout_batch_output_requires_actions_only():
    actions = [np.zeros(7, dtype=np.float32)]
    out = RolloutBatchOutput(actions=actions)

    assert out.actions == actions
    assert out.logprobs is None
    assert out.values is None
    assert out.policy_version is None
    assert out.sidecars == {}


def test_rollout_batch_output_accepts_optional_hidden_sidecar():
    actions = [np.zeros(7, dtype=np.float32)]
    hidden = [np.ones(4, dtype=np.float16)]
    out = RolloutBatchOutput(actions=actions, sidecars={"hidden": hidden})

    assert out.sidecars["hidden"] == hidden


def test_rollout_batch_output_preserves_legacy_dict_shape():
    actions = [np.zeros(7, dtype=np.float32)]
    hidden = [np.ones(4, dtype=np.float16)]
    out = RolloutBatchOutput(
        actions=actions,
        logprobs=[0.0],
        values=[1.0],
        policy_version=3,
        sidecars={"obs_embedding": hidden},
    )

    assert out.to_legacy_dict() == {
        "actions": actions,
        "logprobs": [0.0],
        "values": [1.0],
        "policy_version": 3,
        "obs_embedding": hidden,
    }
