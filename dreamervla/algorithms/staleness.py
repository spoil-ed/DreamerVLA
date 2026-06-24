"""Off-policy staleness policy for async cotrain (RLinf-style version gating).

A replayed sample was collected under some rollout-policy version. As the learner
advances, old samples become increasingly off-policy. ``staleness_threshold`` bounds
how many policy versions old a sample may be before it is dropped from the RL update.
"""

from __future__ import annotations


def version_age(record_version: int, current_version: int) -> int:
    """How many policy versions old a sample is (clamped at 0)."""
    return max(0, int(current_version) - int(record_version))


def is_stale(record_version: int, current_version: int, threshold: int) -> bool:
    """True if the sample is older than ``threshold`` policy versions.

    ``threshold`` < 0 disables gating (nothing is ever stale).
    """
    if int(threshold) < 0:
        return False
    return version_age(record_version, current_version) > int(threshold)
