from __future__ import annotations

from pathlib import Path

import pytest


def test_data_root_falls_back_to_dvla_root_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dreamervla.utils.paths import data_root

    monkeypatch.delenv("DVLA_DATA_ROOT", raising=False)
    monkeypatch.setenv("DVLA_ROOT", "/repo/root")

    assert data_root() == Path("/repo/root/data")


def test_data_root_prefers_dvla_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    from dreamervla.utils.paths import data_path, data_root

    monkeypatch.setenv("DVLA_ROOT", "/repo/root")
    monkeypatch.setenv("DVLA_DATA_ROOT", "/asset/root")

    assert data_root() == Path("/asset/root")
    assert data_path("checkpoints", "model") == Path(
        "/asset/root/checkpoints/model"
    )
