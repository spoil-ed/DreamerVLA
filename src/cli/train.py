from __future__ import annotations

import sys
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.workspace.base_workspace import BaseWorkspace


def run(cfg: DictConfig) -> None:
    # Resolve config
    OmegaConf.resolve(cfg)
    # Workspace class
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    # Run workspace
    workspace.run()


main = run


def _parse_hydra_like_args(argv: list[str]) -> tuple[str, list[str]]:
    config_name = "nopretokenize_sft_cotrain_libero_10"
    overrides: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            print(
                "Usage: python -m src.cli.train --config-name CONFIG [overrides]\n\n"
                "Examples:\n"
                "  python -m src.cli.train --config-name pretokenize_vla_libero_10 training.num_epochs=5\n"
                "  python -m src.cli.train --config-name eval_libero_vla eval.task_suite_name=libero_goal"
            )
            raise SystemExit(0)
        if arg in ("--config-name", "-cn"):
            if i + 1 >= len(argv):
                raise SystemExit(f"{arg} requires a value")
            config_name = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
            i += 1
            continue
        overrides.append(arg)
        i += 1
    return config_name, overrides


def main(argv: list[str] | None = None) -> None:
    config_name, overrides = _parse_hydra_like_args(list(sys.argv[1:] if argv is None else argv))
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    run(cfg)


if __name__ == "__main__":
    main()
