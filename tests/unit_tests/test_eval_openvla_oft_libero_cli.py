from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.diagnostics.eval_openvla_oft_libero import (
    parse_suite_name,
    resolve_num_images_for_camera_inputs,
)


def test_parse_suite_name_accepts_libero_10_aliases() -> None:
    assert parse_suite_name("libero_10") == "libero_10"
    assert parse_suite_name("libero10") == "libero_10"
    assert parse_suite_name("libero_long") == "libero_10"


def test_camera_inputs_selects_effective_num_images() -> None:
    assert resolve_num_images_for_camera_inputs("primary", None) == 1
    assert resolve_num_images_for_camera_inputs(None, None) == 1


@pytest.mark.parametrize(
    "camera_inputs,num_images",
    [("primary+wrist", None), (None, 2), ("primary", 2)],
)
def test_camera_inputs_rejects_removed_multiview_route(
    camera_inputs: str | None,
    num_images: int | None,
) -> None:
    with pytest.raises(ValueError, match="mainline"):
        resolve_num_images_for_camera_inputs(camera_inputs, num_images)


def test_openvla_oft_eval_entry_is_hydra_configured() -> None:
    project_root = Path(__file__).resolve().parents[2]
    text = (
        project_root / "dreamervla" / "diagnostics" / "eval_openvla_oft_libero.py"
    ).read_text(encoding="utf-8")
    config_text = (
        project_root / "configs" / "scripts" / "openvla_oft_official_eval.yaml"
    ).read_text(encoding="utf-8")

    assert "argparse" not in text
    assert "ArgumentParser" not in text
    assert "parse_args" not in text
    assert "initialize_config_dir" in text
    assert "ckpt:" in config_text
    assert "suite: libero_goal" in config_text


def test_openvla_oft_eval_script_config_composes() -> None:
    project_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        config_dir=str(project_root / "configs" / "scripts"),
        job_name="test_openvla_oft_eval_config",
        version_base=None,
    ):
        cfg = compose(
            config_name="openvla_oft_official_eval",
            overrides=["task_ids=0-2", "use_proprio=0"],
        )

    data = OmegaConf.to_container(cfg, resolve=True)
    assert data["suite"] == "libero_goal"
    assert data["task_ids"] == "0-2"
    assert data["use_proprio"] == 0
