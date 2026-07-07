from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dreamervla.algorithms.critic.critic import Critic
from dreamervla.algorithms.critic.latent_success_classifier import (
    LatentSuccessClassifier,
    LatentSuccessClassifierConfig,
)
from dreamervla.algorithms.critic.twohot_critic import (
    ReturnPercentileTracker,
    TwohotCritic,
)


def build_classifier(config: Mapping[str, Any]) -> Any:
    """Construct a success classifier from a config mapping.

    A config carrying a Hydra ``_target_`` is built via ``hydra.utils.instantiate``.
    Legacy configs without a target still construct ``LatentSuccessClassifier``.
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


__all__ = [
    "Critic",
    "LatentSuccessClassifier",
    "LatentSuccessClassifierConfig",
    "ReturnPercentileTracker",
    "TwohotCritic",
    "build_classifier",
]
