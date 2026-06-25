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
    # Contract smoke test: the default verifier exposes the method the LUMOS loop
    # calls (outcome.py). Assert the attribute exists without constructing the
    # (heavyweight) model so the test stays a fast unit test.
    from dreamervla.models.reward.latent_success_classifier import (
        LatentSuccessClassifier,
    )

    assert hasattr(LatentSuccessClassifier, "predict_success")


def test_latent_success_classifier_predict_success_returns_score():
    from dreamervla.models.reward.latent_success_classifier import (
        LatentSuccessClassifier,
        LatentSuccessClassifierConfig,
    )

    classifier = LatentSuccessClassifier(
        LatentSuccessClassifierConfig(
            latent_dim=1,
            window=2,
            head_type="linear",
        )
    ).eval()
    with torch.no_grad():
        classifier.head.weight.zero_()
        classifier.head.bias.zero_()
        classifier.head.weight[1].fill_(1.0)

    video = torch.tensor([[[0.0], [0.0], [1.0]], [[1.0], [1.0], [0.0]]])
    out = classifier.predict_success(video, threshold=2.0, min_steps=0)

    assert set(out) == {"complete", "finish_step", "score", "score_step"}
    assert out["complete"].tolist() == [False, False]
    assert out["finish_step"].tolist() == [2, 2]
    assert out["score_step"].tolist() == [2, 1]
    assert torch.all((out["score"] > 0.0) & (out["score"] < 1.0))
