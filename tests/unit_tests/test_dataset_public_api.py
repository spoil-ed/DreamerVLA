from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

from hydra.utils import get_class
from omegaconf import OmegaConf


def test_all_nested_dataset_config_references_resolve() -> None:
    """Include factory selectors and nested targets, not just root datasets."""
    root = Path(__file__).resolve().parents[2]

    def check(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, str) and value.startswith("dreamervla.dataset."):
            get_class(value)

    for path in (root / "configs").rglob("*.yaml"):
        check(OmegaConf.to_container(OmegaConf.load(path), resolve=False))


def test_dataset_imports_in_runtime_and_tests_resolve() -> None:
    """Catch stale Python module/symbol references after moving files."""
    root = Path(__file__).resolve().parents[2]
    for folder in ("dreamervla", "tests"):
        for path in (root / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("dreamervla.dataset"):
                            importlib.import_module(alias.name)
                elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "dreamervla.dataset"
                ):
                    module = importlib.import_module(node.module)
                    for alias in node.names:
                        if alias.name != "*" and not hasattr(module, alias.name):
                            importlib.import_module(f"{node.module}.{alias.name}")


def test_importing_dataset_package_does_not_load_optional_model_stacks() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dreamervla.dataset; assert not {'openpi', 'prismatic', 'jax'} & sys.modules.keys()",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
