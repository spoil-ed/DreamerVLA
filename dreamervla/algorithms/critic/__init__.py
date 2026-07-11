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
    """Construct the explicitly targeted success classifier."""
    from omegaconf import OmegaConf

    plain = (
        OmegaConf.to_container(config, resolve=True)
        if OmegaConf.is_config(config)
        else dict(config)
    )
    if not plain.get("_target_"):
        raise ValueError("classifier config requires an explicit Hydra _target_")
    import hydra

    return hydra.utils.instantiate(config)


__all__ = [
    "Critic",
    "LatentSuccessClassifier",
    "LatentSuccessClassifierConfig",
    "ReturnPercentileTracker",
    "TwohotCritic",
    "build_classifier",
]
