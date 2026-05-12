from __future__ import annotations

import torch

from scripts.preprocess_rynn_pixel_hidden import _prepare_actor_sequence_arrays
from src.dataloader.libero_pixel_rynn_hidden_sequence_dataset import LIBEROPixelRynnHiddenSequenceDataset
from src.models.world_model.dreamerv3_torch import (
    CompactTokenSequenceAutoencoder,
    DreamerV3LatentState,
    DreamerV3PixelRynnBackboneWorldModel,
)


def test_prepare_actor_sequence_arrays_appends_action_trigger_and_pads() -> None:
    hidden_states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    attention_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    input_ids = [
        [11, 12, 13],
        [21, 22],
    ]

    arrays = _prepare_actor_sequence_arrays(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        input_ids=input_ids,
        target_token_id=10004,
    )

    assert arrays["actor_hidden_states"].shape == (2, 4, 3)
    assert arrays["actor_attention_mask"].shape == (2, 5)
    assert arrays["actor_input_ids"].shape == (2, 5)
    assert arrays["actor_seq_lens"].tolist() == [3, 2]

    assert arrays["actor_input_ids"][0].tolist() == [11, 12, 13, 10004, 0]
    assert arrays["actor_input_ids"][1].tolist() == [21, 22, 10004, 0, 0]
    assert arrays["actor_attention_mask"][0].tolist() == [True, True, True, True, False]
    assert arrays["actor_attention_mask"][1].tolist() == [True, True, True, False, False]


def test_pad_or_truncate_actor_sequence_arrays() -> None:
    array = torch.arange(2 * 3).numpy().reshape(2, 3)

    padded = LIBEROPixelRynnHiddenSequenceDataset._pad_or_truncate_array(array, 5, axis=1)
    truncated = LIBEROPixelRynnHiddenSequenceDataset._pad_or_truncate_array(array, 2, axis=1)

    assert padded.shape == (2, 5)
    assert padded[:, :3].tolist() == array.tolist()
    assert padded[:, 3:].tolist() == [[0, 0], [0, 0]]
    assert truncated.shape == (2, 2)
    assert truncated.tolist() == array[:, :2].tolist()


def test_rynn_world_model_decodes_full_hidden_sequence() -> None:
    model = DreamerV3PixelRynnBackboneWorldModel(
        obs_dim=8,
        action_dim=2,
        image_channels=2,
        image_size=64,
        deter=16,
        hidden=8,
        stoch=2,
        classes=4,
        blocks=1,
        depth=4,
        actor_sequence_length=5,
        sequence_decoder_query_dim=6,
        sequence_decoder_units=12,
        hidden_decoder_units=12,
    )
    latent = DreamerV3LatentState(
        deter=torch.zeros(3, 16),
        stoch=torch.zeros(3, 2, 4),
        logits=None,
    )

    decoded = model({"mode": "actor_input_sequence", "latent": latent})

    assert decoded.shape == (3, 5, 8)


def test_compact_token_sequence_autoencoder_preserves_batch_time_and_masks() -> None:
    model = CompactTokenSequenceAutoencoder(
        in_dim=8,
        latent_tokens=3,
        latent_dim=6,
        target_tokens=4,
        num_heads=2,
    )
    hidden = torch.randn(2, 5, 7, 8)
    mask = torch.tensor(
        [
            [[True, True, True, True, True, False, False]] * 5,
            [[True, True, True, False, False, False, False]] * 5,
        ]
    )

    out = model(hidden, mask)

    assert out["latent"].shape == (2, 5, 3, 6)
    assert out["reconstruction"].shape == (2, 5, 4, 8)
    assert out["target"].shape == (2, 5, 4, 8)
    assert out["target_mask"].shape == (2, 5, 4)
    assert out["target_mask"][0, 0].tolist() == [True, True, True, True]
    assert out["target_mask"][1, 0].tolist() == [False, True, True, True]
