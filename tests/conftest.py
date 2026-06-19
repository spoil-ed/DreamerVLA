from __future__ import annotations

import os


def pytest_configure() -> None:
    os.environ.setdefault("DVLA_DATA_ROOT", "data")
