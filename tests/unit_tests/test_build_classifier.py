"""Hydra-core decoupling: a single _target_-aware classifier builder.

`build_classifier` centralizes success-classifier construction so runners never
hardcode the concrete class. A config with a Hydra `_target_` is instantiated
(swap from config alone); legacy configs / checkpoint blobs without one fall back
to the default LatentSuccessClassifier — byte-identical to the old call sites.
"""

from omegaconf import OmegaConf

from dreamervla.algorithms.critic import LatentSuccessClassifier, build_classifier


def test_build_classifier_defaults_to_latent_success_for_legacy_blob():
    model = build_classifier({"latent_dim": 8, "window": 4})
    assert isinstance(model, LatentSuccessClassifier)
    assert model.cfg.latent_dim == 8
    assert model.cfg.window == 4


def test_build_classifier_honors_hydra_target():
    model = build_classifier(
        {
            "_target_": "dreamervla.algorithms.critic.LatentSuccessClassifier",
            "latent_dim": 8,
        }
    )
    assert isinstance(model, LatentSuccessClassifier)


def test_build_classifier_accepts_omegaconf_config():
    model = build_classifier(OmegaConf.create({"latent_dim": 8}))
    assert isinstance(model, LatentSuccessClassifier)


def test_build_classifier_ignores_stray_target_on_fallback():
    # A None/empty _target_ must not leak into the dataclass kwargs.
    model = build_classifier({"_target_": None, "latent_dim": 8})
    assert isinstance(model, LatentSuccessClassifier)
