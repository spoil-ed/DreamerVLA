from __future__ import annotations

from pathlib import Path

import pytest

from dreamer_vla.algorithms.ppo import (
    dino_wmpo_dense_chunk_step,
    dino_wmpo_dense_step,
    dino_wmpo_outcome_step,
)
from dreamer_vla.algorithms.registry import (
    actor_update_names,
    get_actor_update_route,
)


def test_actor_update_registry_resolves_canonical_routes_and_aliases() -> None:
    outcome = get_actor_update_route("wmpo_outcome")
    assert outcome.step_fn is dino_wmpo_outcome_step
    assert outcome.world_model_arg == "chunk_world_model"
    assert outcome.requires_classifier is True
    assert get_actor_update_route("outcome") is outcome

    dense_chunk = get_actor_update_route("dense-chunk")
    assert dense_chunk.step_fn is dino_wmpo_dense_chunk_step
    assert dense_chunk.world_model_arg == "chunk_world_model"
    assert dense_chunk.requires_classifier is False

    dense = get_actor_update_route("ppo")
    assert dense.step_fn is dino_wmpo_dense_step
    assert dense.world_model_arg == "world_model"
    assert dense.uses_critic is True
    assert dense.uses_real_relabel is True


def test_actor_update_registry_reports_available_names() -> None:
    names = actor_update_names()

    assert "wmpo_outcome" in names
    assert "wmpo_dense_chunk" in names
    assert "wmpo_dense" in names

    with pytest.raises(ValueError, match="Unknown actor update route"):
        get_actor_update_route("not_a_route")


def test_online_runner_uses_actor_update_registry() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source = (
        project_root / "dreamer_vla" / "runners" / "online_dreamervla.py"
    ).read_text(encoding="utf-8")

    assert "get_actor_update_route" in source
    assert "from dreamer_vla.algorithms.ppo import" not in source
