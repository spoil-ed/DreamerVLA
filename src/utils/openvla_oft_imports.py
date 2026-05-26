from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDORED_OPENVLA_OFT_ROOT = PROJECT_ROOT / "src" / "openvla-oft"
WORKSPACE_OPENVLA_OFT_ROOT = PROJECT_ROOT.parent / "openvla-oft"


def ensure_openvla_oft_on_path() -> Path:
    """Put the vendored OpenVLA-OFT checkout first on sys.path."""
    for candidate in (VENDORED_OPENVLA_OFT_ROOT, WORKSPACE_OPENVLA_OFT_ROOT):
        if (candidate / "prismatic").is_dir():
            candidate_str = str(candidate)
            if candidate_str in sys.path:
                sys.path.remove(candidate_str)
            sys.path.insert(0, candidate_str)
            return candidate
    raise FileNotFoundError(
        "Unable to find an OpenVLA-OFT checkout under "
        f"{VENDORED_OPENVLA_OFT_ROOT} or {WORKSPACE_OPENVLA_OFT_ROOT}."
    )


__all__ = ["ensure_openvla_oft_on_path"]
