from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest
import torch

from dreamervla.algorithms.critic.latent_success_classifier import (
    LatentSuccessClassifier,
    LatentSuccessClassifierConfig,
)
from dreamervla.dataset.pixel_hidden_sequence_dataset import PixelHiddenSequenceDataset
from dreamervla.models.embodiment.world_model.wm import WorldModel
from dreamervla.preprocess.sidecar_schema import (
    SIDECAR_SCHEMA_VERSION,
    validate_hidden_token_preprocess_config,
    validate_hidden_token_sidecar_dir,
)
from dreamervla.runners.oft_collect_common import make_preprocess_config
from dreamervla.runners.rollout_hidden_extractor import (
    hidden_token_from_projected,
)


def test_hidden_token_preprocess_config_records_dim_decomposition() -> None:
    cfg = {
        "_policy_mode": "discrete",
        "_use_proprio": False,
        "expected_action_head_type": "oft_discrete_token",
        "expected_obs_hidden_source": "hidden_token",
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

    assert out["obs_hidden_source"] == "hidden_token"
    assert out["num_images_in_input"] == 1
    assert out["patches_per_image"] == 256
    assert out["token_count"] == 256
    assert out["token_dim"] == 4096
    assert out["hidden_dim"] == 256 * 4096
    assert out["obs_embedding_shape"] == [256, 4096]
    assert out["hidden_storage_format"] == "tokenized"


def test_preprocess_config_rejects_legacy_projected_token_source() -> None:
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
        "hidden_dim": 1_048_576,
        "chunk_size": 8,
        "resolution": 256,
        "model_path": "/tmp/oft",
        "unnorm_key": "libero_goal_no_noops",
        "task_suite_name": "libero_goal",
    }

    with pytest.raises(ValueError, match="hidden_token"):
        make_preprocess_config(cfg)


def test_hidden_token_preserves_canonical_projected_tokens() -> None:
    projected = torch.zeros(2, 256, 4096)
    projected[:, -1, -1] = 3

    actual = hidden_token_from_projected(
        projected,
        image_keys=["agentview_rgb"],
        patches_per_image=256,
    )

    assert actual.shape == (2, 256, 4096)
    torch.testing.assert_close(actual, projected)


def test_hidden_token_rejects_insufficient_projected_tokens() -> None:
    with pytest.raises(ValueError, match=r"\[B,256,4096\]"):
        hidden_token_from_projected(
            torch.zeros(2, 255, 4096),
            image_keys=["agentview_rgb"],
            patches_per_image=256,
        )


def _write_sidecar_fixture(tmp_path: Path, *, flat: bool = False, token_count: int = 256) -> None:
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(
            {
                "action_head_type": "oft_discrete_token",
                "obs_hidden_source": "hidden_token",
                "hidden_key": "obs_embedding",
                "token_count": token_count,
                "token_dim": 4096,
                "hidden_dim": token_count * 4096,
                "obs_embedding_shape": [token_count, 4096],
                "num_images_in_input": 1,
                "patches_per_image": token_count,
                "hidden_storage_format": "tokenized",
                "history": 1,
                "include_state": False,
                "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
                "required_demo_datasets": ["obs_embedding"],
            }
        ),
        encoding="utf-8",
    )
    with h5py.File(tmp_path / "shard.hdf5", "w") as handle:
        handle.attrs["complete"] = True
        demo = handle.create_group("data/demo_0")
        shape = (2, token_count * 4096) if flat else (2, token_count, 4096)
        demo.create_dataset("obs_embedding", shape=shape, dtype="float16")


def test_hidden_token_sidecar_accepts_only_canonical_tokenized_storage(tmp_path) -> None:
    _write_sidecar_fixture(tmp_path)

    dataset = PixelHiddenSequenceDataset.__new__(PixelHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path

    dataset._validate_hidden_sidecar(
        expected_model_path=None,
        expected_encoder_state_ckpt=None,
        expected_time_horizon=None,
        expected_action_head_type="oft_discrete_token",
        expected_obs_hidden_source="hidden_token",
        require_preprocess_config=True,
    )


def test_hidden_token_sidecar_rejects_flat_storage_even_when_flat_dim_matches(tmp_path) -> None:
    _write_sidecar_fixture(tmp_path, flat=True)

    dataset = PixelHiddenSequenceDataset.__new__(PixelHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path

    try:
        dataset._validate_hidden_sidecar(
            expected_model_path=None,
            expected_encoder_state_ckpt=None,
            expected_time_horizon=None,
            expected_action_head_type="oft_discrete_token",
            expected_obs_hidden_source="hidden_token",
            require_preprocess_config=True,
        )
    except ValueError as exc:
        assert "must be tokenized [T,N,D]" in str(exc)
    else:
        raise AssertionError("flat hidden-token sidecar storage must be rejected")


def test_hidden_token_sidecar_accepts_metadata_defined_token_geometry(tmp_path) -> None:
    _write_sidecar_fixture(tmp_path, token_count=255)

    dataset = PixelHiddenSequenceDataset.__new__(PixelHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path

    config = dataset._validate_hidden_sidecar(
        expected_model_path=None,
        expected_encoder_state_ckpt=None,
        expected_time_horizon=None,
        expected_action_head_type="oft_discrete_token",
        expected_obs_hidden_source="hidden_token",
        require_preprocess_config=True,
    )

    assert config["token_count"] == 255
    assert config["obs_embedding_shape"] == [255, 4096]


def test_hidden_token_sidecar_rejects_dataset_metadata_geometry_mismatch(tmp_path) -> None:
    _write_sidecar_fixture(tmp_path, token_count=255)
    with h5py.File(tmp_path / "shard.hdf5", "a") as handle:
        del handle["data/demo_0/obs_embedding"]
        handle["data/demo_0"].create_dataset(
            "obs_embedding", shape=(2, 256, 4096), dtype="float16"
        )

    with pytest.raises(ValueError, match=r"expected trailing shape \(255, 4096\)"):
        validate_hidden_token_sidecar_dir(tmp_path)


def test_sidecar_identity_does_not_alias_ckpts_and_checkpoints_paths() -> None:
    assert not PixelHiddenSequenceDataset._same_path(
        "/tmp/data/ckpts/model",
        "/tmp/data/checkpoints/model",
    )


def test_wm_rollout_returns_tokenized_obs_embedding() -> None:
    model = WorldModel(
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


def test_world_model_constructor_rejects_removed_56x1024_interface() -> None:
    with pytest.raises(ValueError, match="removed 56x1024"):
        WorldModel(
            obs_dim=56 * 1024,
            token_count=56,
            token_dim=1024,
            model_dim=8,
            depth=1,
            heads=2,
            mlp_dim=16,
        )


def test_classifier_constructor_rejects_removed_56x1024_interface() -> None:
    with pytest.raises(ValueError, match="removed 56x1024"):
        LatentSuccessClassifier(
            LatentSuccessClassifierConfig(
                latent_dim=1024,
                token_count=56,
                token_dim=1024,
                window=2,
                hidden_dim=8,
                num_layers=1,
                num_heads=2,
                head_type="spatial_tf",
            )
        )


@pytest.mark.parametrize("head_type", ["transformer", "linear", "mlp2"])
def test_classifier_constructor_rejects_flat_57344_alias_regardless_of_token_dim(
    head_type: str,
) -> None:
    with pytest.raises(ValueError, match="removed 56x1024"):
        LatentSuccessClassifier(
            LatentSuccessClassifierConfig(
                latent_dim=56 * 1024,
                token_count=None,
                token_dim=4096,
                window=2,
                hidden_dim=8,
                head_type=head_type,
            )
        )


def test_classifier_does_not_infer_removed_57344_flat_width() -> None:
    with pytest.raises(ValueError, match="latent_dim must be explicit"):
        LatentSuccessClassifier(
            LatentSuccessClassifierConfig(
                latent_dim=None,
                token_count=None,
                token_dim=1024,
                time_horizon=8,
                action_dim=7,
                window=1,
                hidden_dim=8,
                head_type="linear",
            )
        )


def _canonical_sidecar_metadata() -> dict[str, object]:
    return {
        "action_head_type": "oft_discrete_token",
        "obs_hidden_source": "hidden_token",
        "hidden_key": "obs_embedding",
        "token_count": 256,
        "token_dim": 4096,
        "hidden_dim": 1_048_576,
        "obs_embedding_shape": [256, 4096],
        "hidden_storage_format": "tokenized",
        "num_images_in_input": 1,
        "patches_per_image": 256,
        "history": 1,
        "include_state": False,
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "required_demo_datasets": ["obs_embedding"],
    }


@pytest.mark.parametrize("field", ["sidecar_schema_version", "required_demo_datasets"])
def test_sidecar_schema_requires_explicit_contract_declarations(field: str) -> None:
    metadata = _canonical_sidecar_metadata()
    metadata.pop(field)

    with pytest.raises(ValueError, match=field):
        validate_hidden_token_preprocess_config(metadata, context="test sidecar")


@pytest.mark.parametrize("field", ["save_action_hidden", "action_hidden_key"])
def test_sidecar_schema_rejects_removed_action_hidden_fields(field: str) -> None:
    metadata = _canonical_sidecar_metadata()
    metadata[field] = True if field.startswith("save_") else "action_hidden_states"

    with pytest.raises(ValueError, match="removed sidecar fields"):
        validate_hidden_token_preprocess_config(metadata, context="test sidecar")


def test_sidecar_hdf5_rejects_removed_action_hidden_attribute(tmp_path) -> None:
    _write_sidecar_fixture(tmp_path)
    with h5py.File(tmp_path / "shard.hdf5", "a") as handle:
        handle.attrs["save_action_hidden"] = False

    with pytest.raises(ValueError, match="removed sidecar attributes"):
        validate_hidden_token_sidecar_dir(tmp_path)


def test_sidecar_hdf5_rejects_extra_action_hidden_dataset(tmp_path) -> None:
    _write_sidecar_fixture(tmp_path)
    with h5py.File(tmp_path / "shard.hdf5", "a") as handle:
        handle["data/demo_0"].create_dataset(
            "action_hidden_states",
            shape=(2, 56, 4096),
            dtype="float16",
        )

    with pytest.raises(ValueError, match="unexpected datasets"):
        validate_hidden_token_sidecar_dir(tmp_path)


def _write_known_legacy_sidecar(
    tmp_path: Path,
    *,
    token_count: int = 256,
    save_action_hidden: bool = False,
) -> None:
    _write_sidecar_fixture(tmp_path, token_count=token_count)
    metadata = _canonical_sidecar_metadata()
    metadata["obs_hidden_source"] = "input_token_embedding"
    for field in (
        "token_count",
        "hidden_dim",
        "patches_per_image",
        "obs_embedding_shape",
    ):
        metadata.pop(field)
    metadata.update(
        {
            "save_action_hidden": save_action_hidden,
            "action_hidden_key": "action_hidden_states",
            "actor_sequence_keys": {
                "hidden": "actor_hidden_states",
                "input_ids": "actor_input_ids",
                "attention_mask": "actor_attention_mask",
                "seq_lens": "actor_seq_lens",
            },
        }
    )
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    with h5py.File(tmp_path / "shard.hdf5", "a") as handle:
        handle.attrs.update(
            {
                "obs_hidden_source": "input_token_embedding",
                "hidden_key": "obs_embedding",
                "hidden_dim": token_count * 4096,
                "token_count": token_count,
                "token_dim": 4096,
                "hidden_storage_format": "tokenized",
                "history": 1,
                "include_state": False,
                "action_head_type": "oft_discrete_token",
                "save_action_hidden": False,
                "save_actor_sequence": False,
                "action_hidden_sequence_dim": 0,
                "action_hidden_dim": 0,
                "action_trigger_token_id": -1,
                "actor_hidden_dim": 0,
                "actor_sequence_dim": 0,
            }
        )


def _write_reference_shard(
    reference_dir: Path,
    *,
    marked_complete: bool,
) -> None:
    reference_dir.mkdir(parents=True)
    with h5py.File(reference_dir / "shard.hdf5", "w") as handle:
        if marked_complete:
            handle.attrs["complete"] = True
        demo = handle.create_group("data/demo_0")
        for key in (
            "rewards",
            "dones",
            "robot_states",
            "states",
            "sparse_rewards",
        ):
            demo.create_dataset(key, shape=(2, 1), dtype="float32")
        demo.create_dataset("actions", shape=(2, 7), dtype="float32")
        obs = demo.create_group("obs")
        for key in (
            "agentview_rgb",
            "eye_in_hand_rgb",
            "ee_pos",
            "ee_ori",
            "ee_states",
            "gripper_states",
            "joint_states",
        ):
            obs.create_dataset(key, shape=(2, 1), dtype="float32")


def test_sidecar_dir_accepts_known_legacy_manifest_when_hdf5_is_canonical(
    tmp_path: Path,
) -> None:
    """Old preprocessing kept shape facts in HDF5 attrs, not the JSON manifest."""

    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()
    reference_dir = tmp_path / "reward"
    _write_known_legacy_sidecar(hidden_dir)
    _write_reference_shard(reference_dir, marked_complete=False)

    normalized = validate_hidden_token_sidecar_dir(
        hidden_dir,
        reference_dir=reference_dir,
        require_reference_complete=False,
        require_sparse_rewards=True,
    )

    assert normalized["obs_hidden_source"] == "hidden_token"
    assert normalized["token_count"] == 256
    assert normalized["token_dim"] == 4096
    assert normalized["obs_embedding_shape"] == [256, 4096]


def test_legacy_manifest_cannot_enable_removed_action_payload(tmp_path: Path) -> None:
    _write_known_legacy_sidecar(tmp_path, save_action_hidden=True)

    with pytest.raises(ValueError, match="enables removed action/actor"):
        validate_hidden_token_sidecar_dir(tmp_path)


def test_legacy_manifest_does_not_reopen_56_token_sidecars(tmp_path: Path) -> None:
    _write_known_legacy_sidecar(tmp_path, token_count=56)

    with pytest.raises(ValueError, match="not a safe legacy projected-token sidecar"):
        validate_hidden_token_sidecar_dir(tmp_path)


def test_world_model_rejects_flat_canonical_observation() -> None:
    model = WorldModel(
        obs_dim=256 * 4096,
        token_count=256,
        token_dim=4096,
        model_dim=8,
        depth=1,
        heads=2,
        mlp_dim=16,
    )

    with pytest.raises(ValueError, match="flat observations are closed"):
        model.obs_to_tokens(torch.zeros(1, 256 * 4096))


def test_classifier_rejects_flat_canonical_observation() -> None:
    classifier = LatentSuccessClassifier(
        LatentSuccessClassifierConfig(
            latent_dim=4096,
            token_count=256,
            token_dim=4096,
            window=2,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            head_type="spatial_tf",
        )
    )

    with pytest.raises(ValueError, match="flat observation inputs are not supported"):
        classifier(torch.zeros(1, 2, 256 * 4096))


def test_wm_source_uses_role_based_wm_wording() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "dreamervla"
        / "models"
        / "embodiment"
        / "world_model"
        / "wm.py"
    ).read_text(encoding="utf-8")
    assert ("DINO" + "-WM") not in source
    assert ("dino" + "_wm") not in source.lower()
    assert ("dino" + "wm") not in source.lower()
