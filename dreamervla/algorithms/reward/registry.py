"""Registry for swappable LUMOS reward models (mirrors the actor-update registry)."""

from __future__ import annotations

from collections.abc import Iterable

from dreamervla.algorithms.reward.protocol import RewardModel

_REWARD_MODELS: dict[str, RewardModel] = {}


def _normalise_name(name: str) -> str:
    normalised = name.strip().lower().replace("-", "_")
    if not normalised:
        raise ValueError("Reward model name must be non-empty.")
    return normalised


def register_reward_model(model: RewardModel, *, aliases: Iterable[str] = ()) -> RewardModel:
    """Register a reward model and aliases."""

    keys = [_normalise_name(model.name), *(_normalise_name(a) for a in aliases)]
    for key in keys:
        existing = _REWARD_MODELS.get(key)
        if existing is not None and existing is not model:
            raise ValueError(f"Reward model `{key}` is already registered to `{existing.name}`.")
    for key in keys:
        _REWARD_MODELS[key] = model
    return model


def get_reward_model(name: str) -> RewardModel:
    """Return a registered reward model by canonical name or alias."""

    key = _normalise_name(name)
    try:
        return _REWARD_MODELS[key]
    except KeyError as exc:
        known = ", ".join(reward_model_names())
        raise ValueError(
            f"Unknown reward model `{name}`. Available reward models: {known}."
        ) from exc


def reward_model_names() -> tuple[str, ...]:
    """Return canonical registered reward-model names."""

    return tuple(sorted({m.name for m in _REWARD_MODELS.values()}))
