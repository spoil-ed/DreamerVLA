"""World-model environment backend contracts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WorldModelEnvProtocol(Protocol):
    """Env backend that advances observations through a world model snapshot."""

    wm_version: int
    classifier_version: int

    def reset(
        self,
        *,
        task_id: int = 0,
        episode_id: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the initial observation and reset info."""
        ...

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance one action step and return Gymnasium-style output."""
        ...

    def load_world_model_state(self, state_dict: dict[str, Any], version: int) -> None:
        """Load an inference snapshot for the world model."""
        ...

    def load_classifier_state(self, state_dict: dict[str, Any], version: int) -> None:
        """Load an inference snapshot for the classifier or verifier."""
        ...
