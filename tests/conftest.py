from __future__ import annotations

import os

from dreamervla.config_resolvers import register_dreamervla_resolvers


def pytest_configure() -> None:
    os.environ.setdefault("DVLA_DATA_ROOT", "data")
    register_dreamervla_resolvers()
