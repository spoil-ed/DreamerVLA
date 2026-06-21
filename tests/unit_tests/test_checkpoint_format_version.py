"""X-01: checkpoint payloads carry a format_version; loader stays dual-read."""

from __future__ import annotations

import torch

from dreamervla.constants import CHECKPOINT_FORMAT_VERSION
from dreamervla.utils.hf_checkpoint import load_runner_payload


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


def test_load_runner_payload_reads_legacy_unversioned_payload(tmp_path) -> None:
    # Pre-X-01 checkpoints have no format_version; the loader must still read them.
    path = tmp_path / "ckpt" / "latest.ckpt"
    path.parent.mkdir(parents=True)
    legacy = {"cfg": {}, "state_dicts": {"policy": {"p": torch.ones(1)}}, "pickles": {}}
    torch.save(legacy, path)

    loaded = load_runner_payload(path)
    assert loaded.get("format_version") is None
    assert torch.equal(loaded["state_dicts"]["policy"]["p"], torch.ones(1))
