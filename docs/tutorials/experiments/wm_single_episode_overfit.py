#!/usr/bin/env python3
"""Compatibility wrapper for the WM single-episode overfit probe."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from dreamervla.diagnostics.wm_single_episode_overfit import main

    main()
