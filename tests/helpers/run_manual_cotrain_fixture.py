"""Run the test-only tiny manual-cotrain fixture in a subprocess."""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.train import run


def main() -> None:
    register_dreamervla_resolvers()
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_manual_cotrain_fixture.py FIXTURE [key=value ...]")
    fixture = Path(sys.argv[1]).expanduser().resolve()
    cfg = OmegaConf.load(fixture)
    if len(sys.argv) > 2:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[2:]))
    run(cfg)


if __name__ == "__main__":
    main()
