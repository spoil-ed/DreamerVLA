"""Process-local import log suppression for DreamerVLA launchers.

Python imports ``sitecustomize`` automatically when the repository root is on
``PYTHONPATH``. Keep this file narrowly scoped to third-party startup notices
that are printed before DreamerVLA logging is initialized.
"""

from __future__ import annotations

import os


def _truthy_env(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no"}


if _truthy_env("DVLA_SUPPRESS_GYM_NOTICE"):
    try:
        import gym_notices.notices as _gym_notices

        _gym_notices.notices.clear()
    except Exception:
        pass
