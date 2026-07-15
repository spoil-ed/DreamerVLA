from __future__ import annotations

from pathlib import Path

import pytest

from dreamervla.utils.run_config import find_run_config, load_run_config


def _write_yaml(path: Path, text: str = "value: 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_find_run_config_returns_existing_explicit_yaml_file(tmp_path: Path) -> None:
    config = _write_yaml(tmp_path / "custom.yaml")

    assert find_run_config(config) == config


def test_find_run_config_walks_up_from_nested_checkpoint(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = _write_yaml(run_root / ".hydra" / "config.yaml")
    checkpoint = run_root / "checkpoints" / "global_step_7" / "manual_cotrain.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert find_run_config(checkpoint) == config


def test_find_run_config_treats_existing_dotted_path_as_directory(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run.v1"
    config = _write_yaml(run_root / ".hydra" / "config.yaml")

    assert find_run_config(run_root) == config


def test_find_run_config_prefers_any_ancestor_canonical_over_nearer_legacy(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    canonical = _write_yaml(run_root / ".hydra" / "config.yaml")
    checkpoint_dir = run_root / "checkpoints" / "global_step_7"
    _write_yaml(checkpoint_dir / "resolved_config.yaml")
    checkpoint = checkpoint_dir / "manual_cotrain.ckpt"
    checkpoint.touch()

    assert find_run_config(checkpoint) == canonical


def test_find_run_config_falls_back_to_legacy_resolved_config(tmp_path: Path) -> None:
    run_root = tmp_path / "legacy-run"
    legacy = _write_yaml(run_root / "resolved_config.yaml")
    checkpoint = run_root / "checkpoints" / "latest.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert find_run_config(checkpoint) == legacy


def test_find_run_config_reports_original_input_when_no_config_exists(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "missing.ckpt"

    with pytest.raises(FileNotFoundError, match=str(checkpoint)):
        find_run_config(checkpoint)


def test_load_run_config_registers_hydra_and_dreamervla_resolvers(
    tmp_path: Path,
) -> None:
    _write_yaml(
        tmp_path / ".hydra" / "config.yaml",
        "product: ${dvla_mul:6,7}\nyear: ${now:%Y}\n",
    )

    loaded = load_run_config(tmp_path)

    assert loaded.product == 42
    assert len(loaded.year) == 4
    assert loaded.year.isdigit()
