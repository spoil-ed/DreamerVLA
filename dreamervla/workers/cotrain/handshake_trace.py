"""Opt-in tracing of the manual-cotrain env<->rollout handshake.

Set ``DVLA_COTRAIN_HANDSHAKE_TRACE=1`` to print a per-step trace of the
``EnvGroup.interact`` / ``RolloutGroup.generate`` loop. It is off by default so
normal runs are not flooded.

Use it to localize an env<->rollout stall such as the
``EnvGroup.interact did not finish ... RolloutGroup.generate is still running or
waiting for StopMsg`` timeout: the last trace line before output stops points at
the blocked side -- e.g. an env that logged ``recv action response WAIT`` with no
matching rollout ``recv action request OK`` for the same ``key`` means the
rollout never received that observation.

Each Ray worker is a separate process, so the env var must be set in the launch
environment; Ray workers inherit it from the driver. Lines go to stdout (Ray
prefixes them with ``(Worker pid=...)``) and are flushed immediately so they are
not lost when a worker later hangs.
"""

from __future__ import annotations

import os
import time

_ENABLED = os.environ.get("DVLA_COTRAIN_HANDSHAKE_TRACE", "0").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
    "off",
)


def trace_enabled() -> bool:
    """Return whether handshake tracing is turned on."""
    return _ENABLED


def trace(msg: str) -> None:
    """Print one handshake-trace line when enabled; a no-op otherwise."""
    if _ENABLED:
        print(f"[handshake t={time.perf_counter():.3f}] {msg}", flush=True)
