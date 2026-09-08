from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeEnd:
    """Gymnasium-style episode end flags for success-vs-timeout rollouts."""

    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return bool(self.terminated or self.truncated)


def resolve_episode_end(*, success: bool, elapsed_steps: int, max_steps: int) -> EpisodeEnd:
    """Return terminal/timeout flags using Dreamer-style episode semantics."""
    terminated = bool(success)
    truncated = bool(not terminated and int(elapsed_steps) >= int(max_steps))
    return EpisodeEnd(terminated=terminated, truncated=truncated)


__all__ = ["EpisodeEnd", "resolve_episode_end"]
