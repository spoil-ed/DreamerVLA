from __future__ import annotations

import os

import pytest

from dreamervla.config_resolvers import register_dreamervla_resolvers


def pytest_configure() -> None:
    os.environ.setdefault("DVLA_DATA_ROOT", "data")
    register_dreamervla_resolvers()


@pytest.fixture()
def hidden_token_preprocess_config() -> dict[str, object]:
    """Canonical sidecar metadata for rollout-writer tests."""

    return {
        "action_head_type": "oft_discrete_token",
        "obs_hidden_source": "hidden_token",
        "hidden_key": "obs_embedding",
        "token_count": 256,
        "token_dim": 4096,
        "hidden_dim": 1_048_576,
        "obs_embedding_shape": [256, 4096],
        "hidden_storage_format": "tokenized",
        "num_images_in_input": 1,
        "patches_per_image": 256,
        "history": 1,
        "include_state": False,
        "sidecar_schema_version": 1,
        "required_demo_datasets": ["obs_embedding"],
    }
