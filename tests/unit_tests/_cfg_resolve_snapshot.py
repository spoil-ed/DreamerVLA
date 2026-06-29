"""Resolved-config semantic snapshot harness (behaviour-preserving gate).

Compose a set of configs via the real train.yaml root and dump each as
canonical JSON (sorted keys, interpolations left intact via resolve=False).
Two snapshots are *semantically equal* iff the JSON files are identical, which
is order-independent and ignores YAML formatting — the correct proof that a
config refactor (CFG-05/X-03/CFG-08) did not change the resolved config.

Usage:
    python scripts/_cfg_resolve_snapshot.py <out_dir> [name=override ...]

`name` is a config selection passed straight to hydra overrides, e.g.
    dreamervla=openvla_oft_lumos
    experiment=dreamervla_oft_wm_lumos
The label written is the override value's leaf (after '=').
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = str(
    next(
        p / "configs"
        for p in Path(__file__).resolve().parents
        if (p / "configs").is_dir()
    )
)


def snapshot(out_dir: str, selections: list[str]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        for sel in selections:
            label = sel.split("=", 1)[1].replace("/", "_")
            cfg = compose(config_name="train", overrides=[sel])
            data = OmegaConf.to_container(cfg, resolve=False)
            (out / f"{label}.json").write_text(
                json.dumps(data, indent=2, sort_keys=True, default=str)
            )
            print(f"[snapshot] {sel} -> {out / f'{label}.json'}", flush=True)


if __name__ == "__main__":
    snapshot(sys.argv[1], sys.argv[2:])
