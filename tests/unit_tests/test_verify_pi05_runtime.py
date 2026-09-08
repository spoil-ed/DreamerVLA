from __future__ import annotations

import pytest

from dreamervla.diagnostics.checks.verify_pi05_runtime import (
    CRITICAL_PI05_DISTRIBUTION_VERSIONS,
    verify_pi05_distribution_versions,
)


def test_verify_pi05_runtime_accepts_locked_versions() -> None:
    verify_pi05_distribution_versions(CRITICAL_PI05_DISTRIBUTION_VERSIONS.__getitem__)


def test_verify_pi05_runtime_accepts_cuda_local_torch_versions() -> None:
    installed = dict(CRITICAL_PI05_DISTRIBUTION_VERSIONS)
    installed["torch"] = "2.11.0+cu128"
    installed["torchaudio"] = "2.11.0+cu128"
    installed["torchvision"] = "0.26.0+cu128"

    verify_pi05_distribution_versions(installed.__getitem__)


def test_verify_pi05_runtime_rejects_missing_future_and_bddl_drift() -> None:
    installed = dict(CRITICAL_PI05_DISTRIBUTION_VERSIONS)
    installed.pop("future")
    installed["bddl"] = "1.0.1"

    with pytest.raises(SystemExit, match="future is missing") as error:
        verify_pi05_distribution_versions(installed.__getitem__)

    assert "bddl==1.0.1; expected 3.6.0" in str(error.value)
