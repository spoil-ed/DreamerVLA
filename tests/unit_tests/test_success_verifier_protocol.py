import torch

from dreamervla.algorithms.verifier import SuccessVerifier


def test_stub_satisfies_verifier_protocol():
    class _Stub:
        def predict_success(self, latent_video, *, threshold, stride=1, min_steps=1, **kwargs):
            b = latent_video.shape[0]
            return {
                "complete": torch.zeros(b, dtype=torch.bool),
                "finish_step": torch.zeros(b, dtype=torch.long),
            }

    assert isinstance(_Stub(), SuccessVerifier)


def test_latent_success_classifier_declares_predict_success():
    # Contract smoke test: the default verifier exposes the method the WMPO loop
    # calls (outcome.py). Assert the attribute exists without constructing the
    # (heavyweight) model so the test stays a fast unit test.
    from dreamervla.models.reward.latent_success_classifier import (
        LatentSuccessClassifier,
    )

    assert hasattr(LatentSuccessClassifier, "predict_success")
