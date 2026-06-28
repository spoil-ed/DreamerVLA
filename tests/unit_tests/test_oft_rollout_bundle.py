from __future__ import annotations

import numpy as np


def test_oft_rollout_bundle_wires_decoder_and_extractor(monkeypatch) -> None:
    import dreamervla.runners.oft_collect_common as common
    import dreamervla.runners.rollout_hidden_extractor as rhe
    from dreamervla.workers.inference import oft_rollout

    class _FakePolicy:
        pass

    class _FakeDecoder:
        def __init__(
            self,
            policy,
            unnorm_key,
            obs_hidden_source="action_query",
            image_keys=None,
        ) -> None:
            self.policy = policy
            self.unnorm_key = unnorm_key
            self.obs_hidden_source = obs_hidden_source
            self.image_keys = image_keys

        def predict_batch(self, preps):
            return [
                ([np.ones(7, np.float32)] * 8, np.zeros(8, np.float16))
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
        obs_hidden_source="input_token_embedding",
        device="cpu",
    )
    ex = bundle.make_extractor()
    assert ex.prepare({}, "t") == {"ok": True}
    assert bundle._decoder.obs_hidden_source == "input_token_embedding"
    assert bundle._decoder.image_keys == ["agentview_rgb"]
    assert ex.kw["obs_hidden_source"] == "input_token_embedding"
    out = bundle.predict_batch([{"ok": True}])
    assert out[0][1].shape == (8,)


def test_oft_rollout_bundle_loads_on_requested_gpu_and_moves_policy(monkeypatch) -> None:
    import dreamervla.runners.oft_collect_common as common
    import dreamervla.runners.rollout_hidden_extractor as rhe
    from dreamervla.workers.inference import oft_rollout

    captured = {}

    class _FakePolicy:
        def to(self, device):
            captured["to_device"] = device
            return self

    class _FakeDecoder:
        def __init__(self, *args, **kwargs) -> None:
            return None

    def _load_policy(cfg, gpu):
        captured["gpu"] = gpu
        return _FakePolicy()

    monkeypatch.setattr(common, "load_policy", _load_policy)
    monkeypatch.setattr(rhe, "OFTBatchedDecoder", _FakeDecoder)

    bundle = oft_rollout.OFTRolloutBundle(
        policy_cfg={"model_path": "x", "policy_mode": "discrete", "num_images_in_input": 1},
        unnorm_key="libero_goal_no_noops",
        image_keys=["agentview_rgb"],
        history=1,
        device="cuda:1",
    )
    assert captured["gpu"] == 1
    assert bundle.to("cuda:2") is bundle
    assert captured["to_device"] == "cuda:2"


def test_oft_rollout_bundle_cpu_device_requests_cpu_policy_load(monkeypatch) -> None:
    import dreamervla.runners.oft_collect_common as common
    import dreamervla.runners.rollout_hidden_extractor as rhe
    from dreamervla.workers.inference import oft_rollout

    captured = {}

    class _FakePolicy:
        pass

    class _FakeDecoder:
        def __init__(self, *args, **kwargs) -> None:
            return None

    def _load_policy(cfg, device_ref):
        captured["device_ref"] = device_ref
        return _FakePolicy()

    monkeypatch.setattr(common, "load_policy", _load_policy)
    monkeypatch.setattr(rhe, "OFTBatchedDecoder", _FakeDecoder)

    oft_rollout.OFTRolloutBundle(
        policy_cfg={"model_path": "x", "policy_mode": "discrete", "num_images_in_input": 1},
        unnorm_key="libero_goal_no_noops",
        image_keys=["agentview_rgb"],
        history=1,
        device="cpu",
    )

    assert captured["device_ref"] == "cpu"


def test_oft_rollout_bundle_validates_detected_policy_mode(monkeypatch) -> None:
    import pytest

    import dreamervla.runners.oft_collect_common as common
    from dreamervla.workers.inference import oft_rollout

    class _FakePolicy:
        pass

    def _load_policy(cfg, gpu):
        cfg["_policy_mode"] = "l1"
        cfg["_use_proprio"] = True
        return _FakePolicy()

    monkeypatch.setattr(common, "load_policy", _load_policy)

    with pytest.raises(ValueError, match="Detected OFT head"):
        oft_rollout.OFTRolloutBundle(
            policy_cfg={
                "model_path": "x",
                "policy_mode": "auto",
                "num_images_in_input": 1,
            },
            unnorm_key="libero_goal_no_noops",
            image_keys=["agentview_rgb"],
            history=1,
            expected_action_head_type="oft_discrete_token",
            expected_include_state=False,
            device="cpu",
        )
