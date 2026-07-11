from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dreamervla.preprocess.preprocess_oft_input_tokens import (
    _action_head_type_for_mode,
    _input_token_sidecar_dims,
    _load_oft_components,
    _project_path,
    _resolve_num_images_in_input,
    _write_source_input_tokens,
    resolve_oft_policy_mode,
)


def _make_component_ckpt(tmp_path: Path) -> Path:
    (tmp_path / "action_head--6650_checkpoint.pt").write_bytes(b"")
    return tmp_path


def test_auto_mode_rejects_l1_action_head_component(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="L1/action-query checkpoints are closed"):
        resolve_oft_policy_mode(_make_component_ckpt(tmp_path), "auto")


def test_auto_mode_is_not_a_public_compatibility_alias(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="policy_mode='discrete'"):
        resolve_oft_policy_mode(tmp_path, "auto")


def test_explicit_l1_mode_is_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="L1/action-query checkpoints are closed"):
        resolve_oft_policy_mode(tmp_path, "l1")


def test_discrete_mode_rejects_checkpoint_with_l1_component(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="L1/action-query checkpoints are closed"):
        resolve_oft_policy_mode(_make_component_ckpt(tmp_path), "discrete")


def test_invalid_policy_mode_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy_mode"):
        resolve_oft_policy_mode(tmp_path, "diffusion")


def test_action_head_type_attr_follows_mode() -> None:
    with pytest.raises(ValueError, match="discrete"):
        _action_head_type_for_mode("l1")
    assert _action_head_type_for_mode("discrete") == "oft_discrete_token"


def test_num_images_accepts_only_one_image_history_one() -> None:
    args = Namespace(
        num_images_in_input=None,
        history=2,
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
    )
    with pytest.raises(ValueError, match="history=1"):
        _resolve_num_images_in_input(args)
    args = Namespace(num_images_in_input=None, history=1, image_keys=["agentview_rgb"])
    assert _resolve_num_images_in_input(args) == 1
    args = Namespace(num_images_in_input=2, history=1, image_keys=["agentview_rgb"])
    with pytest.raises(ValueError, match="num_images_in_input=1"):
        _resolve_num_images_in_input(args)


def test_input_token_sidecar_dims_accept_only_canonical_patch_tokens() -> None:
    class VisionBackbone:
        def get_num_patches(self) -> int:
            return 256

    class VLA:
        vision_backbone = VisionBackbone()

    token_count, flat_dim = _input_token_sidecar_dims(
        VLA(),
        image_keys=["agentview_rgb"],
        token_dim=4096,
    )

    assert token_count == 256
    assert flat_dim == 256 * 4096

    with pytest.raises(ValueError, match="one agentview image"):
        _input_token_sidecar_dims(
            VLA(),
            image_keys=["agentview_rgb", "eye_in_hand_rgb"],
            token_dim=4096,
        )


def test_oft_preprocess_uses_lumos_prismatic_constants() -> None:
    source = _project_path(
        "dreamervla/preprocess/preprocess_oft_input_tokens.py"
    ).read_text(encoding="utf-8")

    assert "openvla_oft.constants" not in source
    assert "prismatic.vla.constants" in source


def test_oft_preprocess_script_checks_env_and_repairs_partial_sidecars() -> None:
    source = _project_path("scripts/preprocess/35_oft_input_tokens.sh").read_text(
        encoding="utf-8"
    )

    assert "ensure_openvla_oft_on_path" in source
    assert "prismatic.vla.constants" in source
    assert "OVERWRITE_ARGS=(overwrite=true)" in source
    assert "FAKE_ARGS=(fake_oft_components=true)" in source
    assert 'OFT_HISTORY="${OFT_HISTORY:-1}"' in source
    assert 'OFT_IMAGE_KEYS="${OFT_IMAGE_KEYS:-agentview_rgb}"' in source
    assert 'OFT_CHUNK_SIZE="${OFT_CHUNK_SIZE:-1}"' in source
    assert 'chunk_size="${OFT_CHUNK_SIZE}"' in source
    assert "repair incomplete input-token sidecar" in source
    assert "OFT_LATENT_SCHEME" not in source
    assert "out_hidden_token_dir" not in source


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
        fake_num_patches=256,
        num_images_in_input=None,
        history=1,
        image_keys=["agentview_rgb"],
        oft_ckpt=str(tmp_path / "fake_ckpt"),
        center_crop=False,
        unnorm_key="fake",
        include_state=False,
        rotate_images_180=False,
        hidden_key="obs_embedding",
        time_horizon=2,
        action_dim=3,
        token_dim=4096,
        output_dtype="float16",
        chunk_size=2,
        prompt_style="vla_policy",
        resolution=256,
        resolved_policy_mode="discrete",
        max_demos_per_file=None,
    )
    components = _load_oft_components(args, torch.device("cpu"))
    out_input = tmp_path / "input" / source.name
    out_input.parent.mkdir()

    stats = _write_source_input_tokens(
        source_path=source,
        out_input_token_path=out_input,
        components=components,
        args=args,
        rank=0,
    )

    assert stats == {"demos": 1, "frames": length}
    with h5py.File(out_input, "r") as handle:
        assert bool(handle.attrs["complete"])
        assert handle.attrs["obs_hidden_source"] == "input_token_embedding"
        assert handle.attrs["token_count"] == 256
        assert handle.attrs["token_dim"] == 4096
        assert handle.attrs["hidden_storage_format"] == "tokenized"
        demo = handle["data"]["demo_0"]
        assert demo["obs_embedding"].shape == (length, 256, 4096)
        assert demo["lang_emb"].shape == (4096,)
        assert demo["lang_emb"].dtype == np.dtype("float16")
        assert "hidden_token_states" not in demo
