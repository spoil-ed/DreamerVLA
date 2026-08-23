"""The classifier builder requires config-selected Hydra construction."""

import pytest
from omegaconf import OmegaConf

from dreamervla.algorithms.critic import LatentSuccessClassifier, build_classifier


def test_build_classifier_rejects_missing_target():
    with pytest.raises(ValueError, match="explicit Hydra _target_"):
        build_classifier({"latent_dim": 8, "window": 4})


def test_build_classifier_honors_hydra_target():
    model = build_classifier(
        {
            "_target_": "dreamervla.algorithms.critic.LatentSuccessClassifier",
            "latent_dim": 8,
        }
    )
    assert isinstance(model, LatentSuccessClassifier)


def test_build_classifier_accepts_omegaconf_config():
    model = build_classifier(
        OmegaConf.create(
            {
                "_target_": "dreamervla.algorithms.critic.LatentSuccessClassifier",
                "latent_dim": 8,
            }
        )
    )
    assert isinstance(model, LatentSuccessClassifier)


def test_build_classifier_rejects_empty_target():
    with pytest.raises(ValueError, match="explicit Hydra _target_"):
        build_classifier({"_target_": None, "latent_dim": 8})
