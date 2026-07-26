from __future__ import annotations

import numpy as np

import dreamervla.workers.inference.rollout_inference_worker as rollout_worker_module
from dreamervla.workers.inference._test_rollout_stub import HIDDEN_DIM
from dreamervla.workers.inference.rollout_inference_worker import RolloutInferenceWorker


def _cfg() -> dict:
    return {
        "device": "cpu",
        "action_dim": 7,
        "decoder": {"target": "dreamervla.workers.inference._test_rollout_stub:StubRolloutBundle"},
    }


def test_forward_batch_returns_action_and_hidden() -> None:
    w = RolloutInferenceWorker(_cfg(), {}, num_envs=3)
    w.init()
    out = w.forward_batch([{"seed": 10}, {"seed": 20}, {"seed": 30}], [0, 1, 2])
    assert set(out) == {"actions", "obs_embedding"}
    assert out["actions"][0].shape == (7,) and out["actions"][0].dtype == np.float32
    assert out["obs_embedding"][1].shape == (HIDDEN_DIM,)
    assert out["obs_embedding"][1].dtype == np.float16
    assert float(out["actions"][1][0]) == 20.0
    assert float(out["obs_embedding"][2][0]) == 30.0


def test_init_masks_ray_rank_environment_during_bundle_construction(monkeypatch) -> None:
    seen = {}
    original_build = rollout_worker_module._build_from_cfg

    def _recording_build(cfg):
        import os

        seen.update({key: os.environ.get(key) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")})
        return original_build(cfg)

    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "6")
    monkeypatch.setattr(rollout_worker_module, "_build_from_cfg", _recording_build)

    worker = RolloutInferenceWorker(_cfg(), {}, num_envs=1)
    assert worker.rank == 3
    worker.init()

    assert seen == {"RANK": None, "LOCAL_RANK": None, "WORLD_SIZE": None}
    assert rollout_worker_module.os.environ["RANK"] == "3"
    assert rollout_worker_module.os.environ["WORLD_SIZE"] == "6"


class _DecodeResult:
    def __init__(self, action_chunk, hidden_state, lang_emb):
        self.action_chunk = action_chunk
        self.hidden_state = hidden_state
        self.lang_emb = lang_emb

    def __iter__(self):
        yield self.action_chunk
        yield self.hidden_state


class _LangExtractor:
    def prepare(self, obs, task_description):
        return {"seed": int(obs.get("seed", 0))}

    def reset(self):
        return None


class _LangBundle:
    def make_extractor(self):
        return _LangExtractor()

    def predict_batch(self, preps):
        out = []
        for prep in preps:
            seed = int(prep["seed"])
            action_chunk = [np.full((7,), float(seed), dtype=np.float32) for _ in range(8)]
            hidden_state = np.full((HIDDEN_DIM,), float(seed), dtype=np.float16)
            lang_emb = np.full((6,), float(seed + 1), dtype=np.float32)
            out.append(_DecodeResult(action_chunk, hidden_state, lang_emb))
        return out


def test_forward_batch_returns_language_sidecar_when_decoder_provides_it() -> None:
    w = RolloutInferenceWorker(_cfg(), {}, num_envs=2)
    w._bundle = _LangBundle()
    w._extractors = [w._bundle.make_extractor() for _ in range(2)]

    out = w.forward_batch([{"seed": 10}, {"seed": 20}], [0, 1])

    assert len(out["lang_emb"]) == 2
    assert out["lang_emb"][0].shape == (6,)
    assert out["lang_emb"][0].dtype == np.float16
    assert float(out["lang_emb"][1][0]) == 21.0


def test_reset_states_clears_only_named_envs() -> None:
    w = RolloutInferenceWorker(_cfg(), {}, num_envs=2)
    w.init()
    w.forward_batch([{"seed": 1}, {"seed": 2}], [0, 1])
    first_extractor = w._extractors[0]
    second_extractor = w._extractors[1]
    w.reset_states([1])
    assert w._extractors[0] is first_extractor
    assert w._extractors[1] is second_extractor
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
    assert float(out["actions"][1][-1]) == 1.0  # g=0  -> sign(-1)*-1 = +1
    assert float(out["actions"][0][0]) == 20.0  # non-gripper dim untouched


def test_forward_batch_executes_action_chunk_open_loop() -> None:
    cfg = _cfg()
    cfg["action_steps"] = 3
    w = RolloutInferenceWorker(cfg, {}, num_envs=1)
    w.init()

    first = w.forward_batch([{"seed": 0}], [0])
    second = w.forward_batch([{"seed": 10}], [0])
    third = w.forward_batch([{"seed": 20}], [0])

    assert [int(out["actions"][0][0]) for out in (first, second, third)] == [0, 1, 2]


def test_forward_batch_can_disable_hidden_sidecar() -> None:
    cfg = _cfg()
    cfg["emit_hidden_sidecar"] = False
    w = RolloutInferenceWorker(cfg, {}, num_envs=1)
    w.init()

    out = w.forward_batch([{"seed": 10}], [0])

    assert set(out) == {"actions"}
    assert out["actions"][0].shape == (7,)
