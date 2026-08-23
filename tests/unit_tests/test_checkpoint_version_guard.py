"""RLinf-aligned hardening: validate checkpoint format_version on load.

The payload always stores ``format_version`` but nothing ever checked it, so a
newer-format checkpoint loaded by older code would be silently mishandled.
RLinf stores and validates its version on load; CLAUDE.md likewise calls for
early validation of resume checkpoints. The guard only hard-fails the unsafe
direction (checkpoint newer than the code); missing/older versions stay
loadable so the dual-read backward-compat contract is preserved.
"""

import pytest
import torch

from dreamervla.constants import CHECKPOINT_FORMAT_VERSION
from dreamervla.utils.hf_checkpoint import load_runner_payload


def test_load_runner_payload_rejects_future_format_version(tmp_path):
    path = tmp_path / "future.ckpt"
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION + 1, "state_dicts": {}}, path)

    with pytest.raises(ValueError, match="format_version"):
        load_runner_payload(path)


def test_load_runner_payload_accepts_current_version(tmp_path):
    path = tmp_path / "cur.ckpt"
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, "marker": 42}, path)

    assert load_runner_payload(path)["marker"] == 42


def test_load_runner_payload_accepts_compat_payload_without_version(tmp_path):
    # Old / HF-style payloads predate the field; dual-read must still load them.
    path = tmp_path / "legacy_payload.ckpt"
    torch.save({"state_dicts": {"x": 1}}, path)

    assert "state_dicts" in load_runner_payload(path)
