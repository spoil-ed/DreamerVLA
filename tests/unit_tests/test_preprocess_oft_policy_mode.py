from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dreamervla.preprocess.preprocess_oft_action_hidden import (
    _action_head_type_for_mode,
    _input_token_sidecar_dims,
    _load_oft_components,
    _resolve_num_images_in_input,
    _project_path,
    _write_source_sidecars,
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
    assert 'OFT_CHUNK_SIZE="${OFT_CHUNK_SIZE:-1}"' in source
    assert '--chunk-size "${OFT_CHUNK_SIZE}"' in source
    assert "resume incomplete action-hidden sidecar" in source
    assert "resume incomplete input-token sidecar" in source
    assert "existing action-hidden sidecar is incomplete; rerun" not in source
    assert "existing input-token sidecar is incomplete; rerun" not in source


def test_fake_oft_components_write_structural_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "open_the_middle_drawer_demo.hdf5"
    length = 3
    with h5py.File(source, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((length, 7), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset(
            "agentview_rgb",
            data=np.zeros((length, 16, 16, 3), dtype=np.uint8),
        )
        obs.create_dataset(
            "eye_in_hand_rgb",
            data=np.zeros((length, 16, 16, 3), dtype=np.uint8),
        )
        obs.create_dataset("ee_pos", data=np.zeros((length, 3), dtype=np.float32))
        obs.create_dataset("ee_ori", data=np.zeros((length, 3), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.zeros((length, 1), dtype=np.float32))

    args = Namespace(
        fake_oft_components=True,
        fake_num_patches=2,
        num_images_in_input=None,
        history=2,
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
        oft_ckpt=str(tmp_path / "fake_ckpt"),
        center_crop=False,
        unnorm_key="fake",
        include_state=True,
        rotate_images_180=False,
        hidden_key="obs_embedding",
        time_horizon=2,
        action_dim=3,
        token_dim=8,
        output_dtype="float32",
        chunk_size=2,
        prompt_style="vla_policy",
        resolution=256,
        save_action_hidden=True,
        resolved_policy_mode="discrete",
        max_demos_per_file=None,
    )
    components = _load_oft_components(args, torch.device("cpu"))
    out_action = tmp_path / "action" / source.name
    out_input = tmp_path / "input" / source.name
    out_action.parent.mkdir()
    out_input.parent.mkdir()

    stats = _write_source_sidecars(
        source_path=source,
        out_c_path=None,
        out_d_path=None,
        out_action_path=out_action,
        out_input_path=out_input,
        components=components,
        args=args,
        rank=0,
    )

    assert stats == {"demos": 1, "frames": length}
    with h5py.File(out_action, "r") as handle:
        assert bool(handle.attrs["complete"])
        demo = handle["data"]["demo_0"]
        assert demo["obs_embedding"].shape == (length, 2 * 3 * 8)
        assert demo["action_hidden_states"].shape == (length, 2 * 3, 8)
    with h5py.File(out_input, "r") as handle:
        assert bool(handle.attrs["complete"])
        assert handle["data"]["demo_0"]["obs_embedding"].shape == (length, 2 * 2 * 8)
