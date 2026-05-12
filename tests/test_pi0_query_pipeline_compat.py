from __future__ import annotations

import json

import pytest
import torch

from src.dataloader.libero_pixel_rynn_hidden_sequence_dataset import (
    LIBEROPixelRynnHiddenSequenceDataset,
)
from src.models.vla_actor import VLAActionHeadActor


def test_rynn_hidden_sidecar_validates_action_head_type(tmp_path) -> None:
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(
            {
                "model_path": "/tmp/model",
                "encoder_state_ckpt": "/tmp/encoder.ckpt",
                "time_horizon": 5,
                "action_head_type": "legacy",
                "save_actor_sequence": True,
            }
        ),
        encoding="utf-8",
    )
    dataset = LIBEROPixelRynnHiddenSequenceDataset.__new__(LIBEROPixelRynnHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path
    dataset.load_actor_sequence = False

    with pytest.raises(ValueError, match="action_head_type mismatch"):
        dataset._validate_hidden_sidecar(
            expected_model_path=None,
            expected_encoder_state_ckpt=None,
            expected_time_horizon=5,
            expected_action_head_type="pi0_query",
            require_preprocess_config=True,
        )


def test_pi0_query_vla_actor_uses_one_query_per_action_step() -> None:
    actor = VLAActionHeadActor(
        hidden_dim=16,
        action_dim=3,
        time_horizon=4,
        vla_hidden_size=16,
        hidden_size_factor=0.25,
        num_encoder_layers=1,
        adapter_type="identity",
        action_head_type="pi0_query",
    )

    chunk = actor({"mode": "sample", "hidden": torch.randn(2, 16), "deterministic": True})[2][
        "action_chunk"
    ]

    assert actor.action_token_embeddings.weight.shape == (1, 4 * 16)
    assert chunk.shape == (2, 4, 3)
