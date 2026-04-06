from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.workspace.base_workspace import BaseWorkspace


# Hydra entry
@hydra.main(
    config_path="../configs",
    config_name="debug",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Resolve config
    OmegaConf.resolve(cfg)
    # Workspace class
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    # Run workspace
    workspace.run()


if __name__ == "__main__":
    main()
