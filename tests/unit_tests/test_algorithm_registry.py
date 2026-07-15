from __future__ import annotations

import pytest

from dreamervla.algorithms.ppo import (
    dino_lumos_dense_chunk_step,
    dino_lumos_dense_step,
    dino_lumos_step,
)
from dreamervla.algorithms.registry import (
    actor_update_names,
    get_actor_update_route,
)


def test_actor_update_registry_resolves_canonical_routes_and_aliases() -> None:
    outcome = get_actor_update_route("LUMOS")
    assert outcome.step_fn is dino_lumos_step
    assert outcome.world_model_arg == "chunk_world_model"
    assert outcome.requires_classifier is True
    assert get_actor_update_route("outcome") is outcome

    dense_chunk = get_actor_update_route("dense-chunk")
    assert dense_chunk.step_fn is dino_lumos_dense_chunk_step
    assert dense_chunk.world_model_arg == "chunk_world_model"
    assert dense_chunk.requires_classifier is False

    dense = get_actor_update_route("ppo")
    assert dense.step_fn is dino_lumos_dense_step
    assert dense.world_model_arg == "world_model"
    assert dense.uses_critic is True
    assert dense.uses_real_relabel is True


def test_actor_update_registry_reports_available_names() -> None:
    names = actor_update_names()

    assert "LUMOS" in names
    assert "LUMOS_DENSE_CHUNK" in names
    assert "LUMOS_DENSE" in names

    with pytest.raises(ValueError, match="Unknown actor update route"):
        get_actor_update_route("not_a_route")
