"""Pure scheduling helper for periodic real-eval windows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RealEvalState:
    """Tracks the last counters that triggered a real-eval window."""

    last_eval_episode: int = 0
    last_eval_update: int = 0


def should_run_real_eval(
    *,
    enabled: bool,
    completed_episodes: int,
    learner_updates: int,
    every_episodes: int,
    every_learner_updates: int,
    state: RealEvalState,
) -> bool:
    """Return whether periodic real eval should run for the current counters."""
    if not bool(enabled):
        return False
    if int(every_episodes) > 0:
        episode_delta = int(completed_episodes) - int(state.last_eval_episode)
        if episode_delta >= int(every_episodes):
            return True
    if int(every_learner_updates) > 0:
        update_delta = int(learner_updates) - int(state.last_eval_update)
        if update_delta >= int(every_learner_updates):
            return True
    return False


__all__ = ["RealEvalState", "should_run_real_eval"]
