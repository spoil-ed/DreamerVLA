from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from dreamervla.preprocess.preprocess_oft_action_hidden import (
    _action_head_type_for_mode,
    _input_token_sidecar_dims,
    _resolve_num_images_in_input,
    _project_path,
    resolve_oft_policy_mode,
)


def _make_component_ckpt(tmp_path: Path) -> Path:
    (tmp_path / "action_head--6650_checkpoint.pt").write_bytes(b"")
    return tmp_path


def test_auto_mode_detects_l1_from_action_head_component(tmp_path: Path) -> None:
    assert resolve_oft_policy_mode(_make_component_ckpt(tmp_path), "auto") == "l1"


def test_auto_mode_falls_back_to_discrete_for_merged_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    assert resolve_oft_policy_mode(tmp_path, "auto") == "discrete"


def test_l1_mode_requires_action_head_component(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="action_head"):
        resolve_oft_policy_mode(tmp_path, "l1")


def test_explicit_discrete_mode_wins_even_with_component_present(tmp_path: Path) -> None:
    assert resolve_oft_policy_mode(_make_component_ckpt(tmp_path), "discrete") == "discrete"


def test_invalid_policy_mode_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy mode"):
        resolve_oft_policy_mode(tmp_path, "diffusion")


def test_action_head_type_attr_follows_mode() -> None:
    assert _action_head_type_for_mode("l1") == "oft_l1_regression"
    assert _action_head_type_for_mode("discrete") == "oft_discrete_token"


def test_num_images_defaults_to_history_times_views() -> None:
    args = Namespace(
        num_images_in_input=None,
        history=2,
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
    )
    assert _resolve_num_images_in_input(args) == 4
    args = Namespace(num_images_in_input=None, history=1, image_keys=["agentview_rgb"])
    assert _resolve_num_images_in_input(args) == 1
    args = Namespace(num_images_in_input=2, history=1, image_keys=["agentview_rgb"])
    assert _resolve_num_images_in_input(args) == 2


def test_input_token_sidecar_dims_use_current_frame_patch_tokens() -> None:
    class VisionBackbone:
        def get_num_patches(self) -> int:
            return 256

    class VLA:
        vision_backbone = VisionBackbone()

    token_count, flat_dim = _input_token_sidecar_dims(
        VLA(),
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
        token_dim=4096,
    )

    assert token_count == 512
    assert flat_dim == 512 * 4096


def test_oft_preprocess_uses_wmpo_prismatic_constants() -> None:
    source = _project_path(
        "dreamervla/preprocess/preprocess_oft_action_hidden.py"
    ).read_text(encoding="utf-8")

    assert "openvla_oft.constants" not in source
    assert "prismatic.vla.constants" in source


def test_oft_preprocess_script_checks_env_and_resumes_partial_sidecars() -> None:
    source = _project_path("scripts/preprocess/35_oft_action_hidden.sh").read_text(
        encoding="utf-8"
    )

    assert "_check_openvla_oft_env" in source
    assert "ensure_openvla_oft_on_path" in source
    assert "prismatic.vla.constants" in source
    assert "OVERWRITE_ARGS=(--overwrite)" in source
    assert "resume incomplete action-hidden sidecar" in source
    assert "resume incomplete input-token sidecar" in source
    assert "existing action-hidden sidecar is incomplete; rerun" not in source
    assert "existing input-token sidecar is incomplete; rerun" not in source
