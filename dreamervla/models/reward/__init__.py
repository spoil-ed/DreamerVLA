from collections.abc import Mapping
from typing import Any

from dreamervla.models.reward.latent_success_classifier import (  # noqa: F401
    LatentSuccessClassifier,
    LatentSuccessClassifierConfig,
)


def build_classifier(config: Mapping[str, Any]) -> Any:
    """Construct a success classifier from a config mapping, decoupled from the class.

    A config carrying a Hydra ``_target_`` is built via ``hydra.utils.instantiate``
    (swap the classifier from config alone). Configs / checkpoint blobs without a
    ``_target_`` fall back to the default ``LatentSuccessClassifier`` — byte-identical
    to the historical ``LatentSuccessClassifier(LatentSuccessClassifierConfig(**blob))``
    call sites, so existing checkpoints load unchanged.
    """
    from omegaconf import OmegaConf

    plain = (
        OmegaConf.to_container(config, resolve=True)
        if OmegaConf.is_config(config)
        else dict(config)
    )
    if plain.get("_target_"):
        import hydra

        return hydra.utils.instantiate(config)
    plain.pop("_target_", None)
    return LatentSuccessClassifier(LatentSuccessClassifierConfig(**plain))
