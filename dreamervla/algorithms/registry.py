"""Registry for actor-update routes used by DreamerVLA runners."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from dreamervla.algorithms.ppo import (
    dino_lumos_dense_chunk_step,
    dino_lumos_dense_step,
    dino_lumos_step,
)

ActorUpdateStep = Callable[..., dict[str, Any]]
WorldModelArg = Literal["world_model", "chunk_world_model"]


@dataclass(frozen=True)
class ActorUpdateRoute:
    """Executable actor-update route metadata.

    ``step_fn`` implementations intentionally keep their domain-specific
    signatures. The runner uses the metadata here to assemble the appropriate
    kwargs without hard-coding every route name at the call site.
    """

    name: str
    step_fn: ActorUpdateStep
    world_model_arg: WorldModelArg
    requires_classifier: bool = False
    uses_critic: bool = False
    uses_real_relabel: bool = False


_ACTOR_UPDATE_ROUTES: dict[str, ActorUpdateRoute] = {}


def _normalise_name(name: str) -> str:
    normalised = name.strip().lower().replace("-", "_")
    if not normalised:
        raise ValueError("Actor update route name must be non-empty.")
    return normalised


def register_actor_update_route(
    route: ActorUpdateRoute,
    *,
    aliases: Iterable[str] = (),
) -> ActorUpdateRoute:
    """Register an actor-update route and aliases."""

    keys = [_normalise_name(route.name), *(_normalise_name(alias) for alias in aliases)]
    for key in keys:
        existing = _ACTOR_UPDATE_ROUTES.get(key)
        if existing is not None and existing is not route:
            raise ValueError(
                f"Actor update route `{key}` is already registered to `{existing.name}`."
            )
    for key in keys:
        _ACTOR_UPDATE_ROUTES[key] = route
    return route


def get_actor_update_route(name: str) -> ActorUpdateRoute:
    """Return a registered non-Dreamer actor-update route."""

    key = _normalise_name(name)
    try:
        return _ACTOR_UPDATE_ROUTES[key]
    except KeyError as exc:
        known = ", ".join(actor_update_names())
        raise ValueError(
            f"Unknown actor update route `{name}`. Available routes: {known}."
        ) from exc


def actor_update_names() -> tuple[str, ...]:
    """Return canonical registered actor-update names."""

    return tuple(sorted({route.name for route in _ACTOR_UPDATE_ROUTES.values()}))


register_actor_update_route(
    ActorUpdateRoute(
        name="LUMOS",
        step_fn=dino_lumos_step,
        world_model_arg="chunk_world_model",
        requires_classifier=True,
    ),
    aliases=("outcome",),
)

register_actor_update_route(
    ActorUpdateRoute(
        name="LUMOS_DENSE_CHUNK",
        step_fn=dino_lumos_dense_chunk_step,
        world_model_arg="chunk_world_model",
    ),
    aliases=("dense_chunk",),
)

register_actor_update_route(
    ActorUpdateRoute(
        name="LUMOS_DENSE",
        step_fn=dino_lumos_dense_step,
        world_model_arg="world_model",
        uses_critic=True,
        uses_real_relabel=True,
    ),
    aliases=("ppo", "grpo"),
)


__all__ = [
    "ActorUpdateRoute",
    "ActorUpdateStep",
    "actor_update_names",
    "get_actor_update_route",
    "register_actor_update_route",
]
