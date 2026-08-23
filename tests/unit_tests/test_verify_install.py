from __future__ import annotations

import pytest

from dreamervla.diagnostics import verify_install

EXPECTED_CRITICAL_VERSIONS = {
    "diffusers": "0.33.0",
    "draccus": "0.8.0",
    "jsonlines": "4.0.0",
    "mujoco": "3.8.0",
    "numpy": "1.26.4",
    "peft": "0.11.0",
    "protobuf": "4.25.9",
    "ray": "2.55.1",
    "sentencepiece": "0.1.99",
    "tensorflow": "2.15.0",
    "tensorflow-datasets": "4.9.3",
    "tensorflow-graphics": "2021.12.3",
    "tensorflow-metadata": "1.17.3",
    "timm": "0.9.10",
    "tokenizers": "0.19.1",
    "torch": "2.5.1",
    "torchaudio": "2.5.1",
    "torchvision": "0.20.1",
    "transformers": "4.40.1",
    "wandb": "0.26.1",
}


def test_verify_install_declares_exact_critical_distribution_versions() -> None:
    assert getattr(verify_install, "CRITICAL_DISTRIBUTION_VERSIONS", None) == (
        EXPECTED_CRITICAL_VERSIONS
    )


def test_verify_install_rejects_critical_distribution_drift() -> None:
    verifier = getattr(verify_install, "verify_distribution_versions", None)
    assert verifier is not None

    installed = dict(EXPECTED_CRITICAL_VERSIONS)
    installed["draccus"] = "0.11.5"

    with pytest.raises(SystemExit, match=r"draccus==0\.11\.5; expected 0\.8\.0"):
        verifier(installed.__getitem__)


def test_verify_install_accepts_exact_critical_distribution_versions() -> None:
    verifier = getattr(verify_install, "verify_distribution_versions", None)
    assert verifier is not None
    verifier(EXPECTED_CRITICAL_VERSIONS.__getitem__)


def test_verify_install_accepts_cuda_local_versions_for_pytorch_wheels() -> None:
    verifier = getattr(verify_install, "verify_distribution_versions", None)
    assert verifier is not None
    installed = dict(EXPECTED_CRITICAL_VERSIONS)
    installed.update(
        {
            "torch": "2.5.1+cu124",
            "torchaudio": "2.5.1+cu124",
            "torchvision": "0.20.1+cu124",
        }
    )
    verifier(installed.__getitem__)
