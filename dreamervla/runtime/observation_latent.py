"""Producer-neutral observation-latent metadata used by collection and replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from dreamervla.preprocess.sidecar_schema import (
    DEFAULT_HIDDEN_KEY,
    HIDDEN_TOKEN_STORAGE_FORMAT,
    OBSERVATION_LATENT_SCHEMA_VERSION,
    required_demo_datasets,
    validate_observation_latent_preprocess_config,
)


@dataclass(frozen=True)
class ObservationLatentSpec:
    """Stable geometry and provenance for one VLA observation representation."""

    policy_family: str
    obs_hidden_source: str
    action_head_type: str
    token_count: int
    token_dim: int
    chunk_size: int
    history: int = 1
    include_state: bool = False
    prompt_style: str = ""
    rotate_images_180: bool = False
    prefix_selection: str | None = None
    alignment_source: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ObservationLatentSpec:
        """Construct from a Hydra/plain mapping without retaining config objects."""

        return cls(
            policy_family=str(value["policy_family"]),
            obs_hidden_source=str(value["obs_hidden_source"]),
            action_head_type=str(value["action_head_type"]),
            token_count=int(value["token_count"]),
            token_dim=int(value["token_dim"]),
            chunk_size=int(value["chunk_size"]),
            history=int(value.get("history", value.get("expected_history", 1))),
            include_state=bool(
                value.get("include_state", value.get("expected_include_state", False))
            ),
            prompt_style=str(value.get("prompt_style", value.get("expected_prompt_style", ""))),
            rotate_images_180=bool(
                value.get("rotate_images_180", value.get("expected_rotate_images_180", False))
            ),
            prefix_selection=(
                str(value["prefix_selection"])
                if value.get("prefix_selection") is not None
                else None
            ),
            alignment_source=(
                str(value["alignment_source"])
                if value.get("alignment_source") is not None
                else None
            ),
        )

    @property
    def hidden_dim(self) -> int:
        return int(self.token_count * self.token_dim)

    def preprocess_config(self, **producer_metadata: Any) -> dict[str, Any]:
        """Build the sidecar manifest persisted beside rollout HDF5 shards."""

        config: dict[str, Any] = {
            **asdict(self),
            "hidden_key": DEFAULT_HIDDEN_KEY,
            "hidden_storage_format": HIDDEN_TOKEN_STORAGE_FORMAT,
            "hidden_dim": self.hidden_dim,
            "wm_obs_dim": self.hidden_dim,
            "obs_embedding_shape": [self.token_count, self.token_dim],
            "time_horizon": self.chunk_size,
            "sidecar_schema_version": OBSERVATION_LATENT_SCHEMA_VERSION,
            "required_demo_datasets": required_demo_datasets(),
            **producer_metadata,
        }
        config = {key: value for key, value in config.items() if value is not None}
        validate_observation_latent_preprocess_config(
            config, context="observation latent preprocess_config"
        )
        return config


__all__ = ["ObservationLatentSpec"]
