from __future__ import annotations

import numpy as np


def test_oft_rollout_bundle_wires_decoder_and_extractor(monkeypatch) -> None:
    import dreamervla.runners.oft_collect_common as common
    import dreamervla.runners.rollout_hidden_extractor as rhe
    from dreamervla.workers.inference import oft_rollout

    class _FakePolicy:
        pass

    class _FakeDecoder:
        def __init__(self, policy, unnorm_key) -> None:
            self.policy = policy
            self.unnorm_key = unnorm_key

        def predict_batch(self, preps):
            return [
                ([np.ones(7, np.float32)] * 8, np.zeros(229376, np.float16))
                for _ in preps
            ]

    class _FakeExtractor:
        def __init__(self, policy, **kw) -> None:
            self.kw = kw

        def reset(self) -> None:
            return None

        def prepare(self, obs, task):
            return {"ok": True}

    monkeypatch.setattr(common, "load_policy", lambda cfg, gpu: _FakePolicy())
    monkeypatch.setattr(rhe, "OFTBatchedDecoder", _FakeDecoder)
    monkeypatch.setattr(rhe, "OFTRolloutHiddenExtractor", _FakeExtractor)

    bundle = oft_rollout.OFTRolloutBundle(
        policy_cfg={"model_path": "x", "policy_mode": "discrete", "num_images_in_input": 1},
        unnorm_key="libero_goal_no_noops",
        image_keys=["agentview_rgb"],
        history=1,
        device="cpu",
    )
    ex = bundle.make_extractor()
    assert ex.prepare({}, "t") == {"ok": True}
    out = bundle.predict_batch([{"ok": True}])
    assert out[0][1].shape == (229376,)
