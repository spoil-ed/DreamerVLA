"""Shared rollout inference output contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RolloutBatchOutput:
    """Normalized output from a rollout or policy inference worker."""

    actions: list[Any]
    logprobs: list[Any] | None = None
    values: list[Any] | None = None
    policy_version: int | None = None
    sidecars: dict[str, list[Any]] = field(default_factory=dict)

    def to_compat_dict(self) -> dict[str, Any]:
        """Return the dict shape consumed by the current runner."""
        out: dict[str, Any] = {"actions": self.actions}
        if self.logprobs is not None:
            out["logprobs"] = self.logprobs
        if self.values is not None:
            out["values"] = self.values
        if self.policy_version is not None:
            out["policy_version"] = int(self.policy_version)
        out.update(self.sidecars)
        return out
