from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from dreamervla.dataset.pixel_hidden_sequence_dataset import PixelHiddenSequenceDataset
from dreamervla.models.world_model.dino_wm import DinoWMWorldModel
from dreamervla.runners.oft_collect_common import make_preprocess_config


def test_input_token_preprocess_config_records_dim_decomposition() -> None:
    cfg = {
        "_policy_mode": "discrete",
        "_use_proprio": False,
        "expected_action_head_type": "oft_discrete_token",
        "expected_obs_hidden_source": "input_token_embedding",
        "expected_prompt_style": "vla_policy",
        "expected_history": 1,
        "expected_rotate_images_180": True,
        "time_horizon": 8,
        "token_dim": 4096,
        "action_dim": 7,
        "num_images_in_input": 1,
        "patches_per_image": 256,
        "token_count": 256,
        "hidden_dim": 1048576,
        "chunk_size": 8,
        "resolution": 256,
        "model_path": "/tmp/oft",
        "unnorm_key": "libero_goal_no_noops",
        "task_suite_name": "libero_goal",
    }

    out = make_preprocess_config(cfg)

    assert out["num_images_in_input"] == 1
    assert out["patches_per_image"] == 256
    assert out["token_count"] == 256
    assert out["token_dim"] == 4096
    assert out["hidden_dim"] == 256 * 4096
    assert out["obs_embedding_shape"] == [256, 4096]
    assert out["hidden_storage_format"] == "tokenized"


def test_input_token_sidecar_accepts_tokenized_storage_with_flat_metadata(tmp_path) -> None:
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(
            {
                "action_head_type": "oft_discrete_token",
                "obs_hidden_source": "input_token_embedding",
                "hidden_key": "obs_embedding",
                "token_count": 3,
                "token_dim": 4,
                "hidden_dim": 12,
                "num_images_in_input": 1,
                "patches_per_image": 3,
                "hidden_storage_format": "tokenized",
            }
        ),
        encoding="utf-8",
    )
    with h5py.File(tmp_path / "shard.hdf5", "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("obs_embedding", data=np.zeros((2, 3, 4), dtype=np.float16))

    dataset = PixelHiddenSequenceDataset.__new__(PixelHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path
    dataset.load_actor_sequence = False

    dataset._validate_hidden_sidecar(
        expected_model_path=None,
        expected_encoder_state_ckpt=None,
        expected_time_horizon=None,
        expected_action_head_type="oft_discrete_token",
        expected_obs_hidden_source="input_token_embedding",
        require_preprocess_config=True,
    )


def test_input_token_sidecar_rejects_flat_storage_even_when_flat_dim_matches(tmp_path) -> None:
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(
            {
                "action_head_type": "oft_discrete_token",
                "obs_hidden_source": "input_token_embedding",
                "hidden_key": "obs_embedding",
                "token_count": 3,
                "token_dim": 4,
                "hidden_dim": 12,
                "num_images_in_input": 1,
                "patches_per_image": 3,
                "hidden_storage_format": "tokenized",
            }
        ),
        encoding="utf-8",
    )
    with h5py.File(tmp_path / "shard.hdf5", "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("obs_embedding", data=np.zeros((2, 12), dtype=np.float16))

    dataset = PixelHiddenSequenceDataset.__new__(PixelHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path
    dataset.load_actor_sequence = False

    try:
        dataset._validate_hidden_sidecar(
            expected_model_path=None,
            expected_encoder_state_ckpt=None,
            expected_time_horizon=None,
            expected_action_head_type="oft_discrete_token",
            expected_obs_hidden_source="input_token_embedding",
            require_preprocess_config=True,
        )
    except ValueError as exc:
        assert "input-token obs_embedding shape mismatch" in str(exc)
    else:
        raise AssertionError("flat input-token sidecar storage must be rejected")


def test_input_token_sidecar_rejects_token_count_decomposition_mismatch(tmp_path) -> None:
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(
            {
                "action_head_type": "oft_discrete_token",
                "obs_hidden_source": "input_token_embedding",
                "hidden_key": "obs_embedding",
                "token_count": 4,
                "token_dim": 4,
                "hidden_dim": 16,
                "num_images_in_input": 1,
                "patches_per_image": 3,
                "hidden_storage_format": "tokenized",
            }
        ),
        encoding="utf-8",
    )
    with h5py.File(tmp_path / "shard.hdf5", "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("obs_embedding", data=np.zeros((2, 4, 4), dtype=np.float16))

    dataset = PixelHiddenSequenceDataset.__new__(PixelHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path
    dataset.load_actor_sequence = False

    try:
        dataset._validate_hidden_sidecar(
            expected_model_path=None,
            expected_encoder_state_ckpt=None,
            expected_time_horizon=None,
            expected_action_head_type="oft_discrete_token",
            expected_obs_hidden_source="input_token_embedding",
            require_preprocess_config=True,
        )
    except ValueError as exc:
        assert "token_count decomposition mismatch" in str(exc)
    else:
        raise AssertionError("bad token_count decomposition must be rejected")


def test_dino_wm_rollout_returns_tokenized_obs_embedding() -> None:
    model = DinoWMWorldModel(
        obs_dim=8,
        action_dim=3,
        token_count=2,
        token_dim=4,
        time_horizon=1,
        model_dim=16,
        depth=1,
        heads=4,
        mlp_dim=32,
        max_seq_len=8,
        num_hist=1,
    )
    obs_tokens = torch.randn(2, 1, 2, 4)
    actions = torch.randn(2, 3, 3)

    out = model.rollout(obs_embedding=obs_tokens, actions=actions)

    assert out["obs_embedding"].shape == (2, 3, 2, 4)
    assert out["obs_tokens"].shape == (2, 3, 2, 4)


def test_dino_wm_source_uses_role_based_wm_wording() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "dreamervla"
        / "models"
        / "world_model"
        / "dino_wm.py"
    ).read_text(encoding="utf-8")
    assert "DINO-WM" not in source
