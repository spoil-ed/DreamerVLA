"""X-01: checkpoint payloads carry a format_version; loader stays dual-read."""

from __future__ import annotations

import torch

from dreamervla.constants import CHECKPOINT_FORMAT_VERSION
from dreamervla.utils.hf_checkpoint import (
    is_hf_checkpoint,
    load_runner_payload,
    resolve_hf_checkpoint_dir,
)


def _make_hf_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"")
    return path.resolve()


def test_checkpoint_format_version_is_two() -> None:
    assert CHECKPOINT_FORMAT_VERSION == 2


def test_load_runner_payload_roundtrips_versioned_payload(tmp_path) -> None:
    path = tmp_path / "latest.ckpt"
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "env_step": 42,
        "update_step": 7,
        "cfg": {"a": 1},
        "state_dicts": {"world_model": {"w": torch.zeros(2)}},
    }
    torch.save(payload, path)

    loaded = load_runner_payload(path)
    assert loaded["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert loaded["env_step"] == 42
    assert torch.equal(loaded["state_dicts"]["world_model"]["w"], torch.zeros(2))


def test_load_runner_payload_reads_compat_unversioned_payload(tmp_path) -> None:
    # Pre-X-01 checkpoints have no format_version; the loader must still read them.
    path = tmp_path / "ckpt" / "latest.ckpt"
    path.parent.mkdir(parents=True)
    legacy_payload = {"cfg": {}, "state_dicts": {"policy": {"p": torch.ones(1)}}, "pickles": {}}
    torch.save(legacy_payload, path)

    loaded = load_runner_payload(path)
    assert loaded.get("format_version") is None
    assert torch.equal(loaded["state_dicts"]["policy"]["p"], torch.ones(1))


def test_torch_ckpt_beside_hf_sidecars_not_detected_as_hf(tmp_path) -> None:
    # Regression (GPU resume smoke): with training.checkpoint_format=both, a run's
    # checkpoints/ dir holds the torch resume artifact (latest.ckpt) AND per-module HF
    # sidecars (latest_hf_policy/, latest_hf_world_model/, ...). is_hf_checkpoint on the
    # torch file must stay False so resume() routes it to the torch loader; otherwise the
    # sibling-scan mis-classifies it as HF and runners without HF resume raise.
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    torch_ckpt = ckpt_dir / "latest.ckpt"
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, "state_dicts": {}}, torch_ckpt)
    _make_hf_dir(ckpt_dir / "latest_hf_policy")
    _make_hf_dir(ckpt_dir / "latest_hf_world_model")

    assert is_hf_checkpoint(torch_ckpt) is False


def test_hf_sidecar_dir_and_inner_file_still_detected(tmp_path) -> None:
    # A real HF dir, and a weight file inside it, must still resolve to that dir.
    hf = _make_hf_dir(tmp_path / "checkpoints" / "latest_hf_policy")
    assert is_hf_checkpoint(hf) is True
    assert resolve_hf_checkpoint_dir(hf / "model.safetensors") == hf


def test_run_dir_with_hf_subdir_still_detected(tmp_path) -> None:
    # Directory input that CONTAINS an HF subdir still resolves (nested-level scan).
    run = tmp_path / "run"
    _make_hf_dir(run / "latest_hf")
    assert is_hf_checkpoint(run) is True


def test_load_runner_payload_is_independent_of_overwrite(tmp_path) -> None:
    # Resume loads optimizer/module tensors that training then overwrites in place
    # (checkpoints/latest.ckpt is rewritten every checkpoint_every updates). The loaded
    # payload must be an independent in-memory copy, NOT an mmap view of the file — an
    # mmap-backed tensor reads the rewritten bytes (silent optimizer-state corruption)
    # or faults (SIGBUS) on the next access. Regression for the GPU resume smoke crash.
    path = tmp_path / "latest.ckpt"
    torch.save({"state_dicts": {"opt": {"exp_avg": torch.full((64,), 3.0)}}}, path)
    loaded = load_runner_payload(path)
    retained = loaded["state_dicts"]["opt"]["exp_avg"]
    torch.save({"state_dicts": {"opt": {"exp_avg": torch.zeros(64)}}}, path)  # overwrite

    assert torch.equal(retained, torch.full((64,), 3.0))
