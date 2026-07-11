"""Async inference is decoupled from the concrete model.

The async OnlineCotrainRayRunner resolves its inference worker class from config
(inference.worker_target), so OFT can select the model-agnostic RolloutInferenceWorker
(OFTRolloutBundle) instead of the VLA-specific InferenceWorker. The OFT rollout is a
fixed base policy in the async overlap loop, so its pull_weights is a no-op.
"""

from __future__ import annotations

from dreamervla.runners.online_cotrain_ray_runner import _resolve_worker_cls
from dreamervla.workers.inference.rollout_inference_worker import RolloutInferenceWorker


def test_resolve_worker_cls_colon_form():
    cls = _resolve_worker_cls(
        "dreamervla.workers.inference.rollout_inference_worker:RolloutInferenceWorker"
    )
    assert cls is RolloutInferenceWorker


def test_resolve_worker_cls_dotted_form():
    cls = _resolve_worker_cls(
        "dreamervla.workers.inference.rollout_inference_worker.RolloutInferenceWorker"
    )
    assert cls is RolloutInferenceWorker


def test_inference_worker_target_resolves_explicitly():
    cls = _resolve_worker_cls(
        "dreamervla.workers.inference.inference_worker:InferenceWorker"
    )
    assert cls.__name__ == "InferenceWorker"


def test_rollout_inference_worker_pull_weights_is_noop():
    # Fixed OFT base policy in the async overlap loop -> nothing to pull.
    worker = RolloutInferenceWorker.__new__(RolloutInferenceWorker)
    assert worker.pull_weights("store", "policy", 0) is None
