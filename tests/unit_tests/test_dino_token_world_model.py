from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest
import torch

from dreamervla.models.embodiment.world_model.dino_token import (
    DinoTokenEmbedding,
    DinoTokenViTPredictor,
    DinoTokenWorldModel,
    generate_dino_causal_mask,
)


def _tiny_model() -> DinoTokenWorldModel:
    return DinoTokenWorldModel(
        token_count=2,
        token_dim=4,
        action_dim=2,
        proprio_dim=3,
        action_emb_dim=2,
        proprio_emb_dim=2,
        num_action_repeat=1,
        num_proprio_repeat=1,
        num_hist=3,
        num_pred=1,
        depth=1,
        heads=2,
        dim_head=2,
        mlp_dim=8,
        dropout=0.0,
        emb_dropout=0.0,
    )


def test_causal_mask_is_lower_triangular_by_frame_not_patch() -> None:
    mask = generate_dino_causal_mask(num_patches=2, num_frames=3)

    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    assert mask.shape == (1, 1, 6, 6)
    assert torch.equal(mask[0, 0], expected)


def test_token_embedding_is_the_dino_kernel_one_conv1d() -> None:
    embedding = DinoTokenEmbedding(in_chans=2, emb_dim=3)
    with torch.no_grad():
        embedding.patch_embed.weight.copy_(
            torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]], [[2.0], [-1.0]]])
        )
        embedding.patch_embed.bias.zero_()
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    actual = embedding(values)

    expected = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 4.0, 2.0]]])
    assert torch.equal(actual, expected)


def test_openvla_tokens_are_layer_normalized_at_the_dino_boundary() -> None:
    model = _tiny_model()
    tokens = torch.arange(1, 1 + 2 * 2 * 4, dtype=torch.float32).reshape(1, 2, 2, 4)
    proprio = torch.zeros(1, 2, 3)

    encoded = model.encode_obs({"visual": tokens, "proprio": proprio})["visual"]

    assert torch.allclose(encoded.mean(dim=-1), torch.zeros(1, 2, 2), atol=1.0e-6)
    assert torch.allclose(
        encoded.var(dim=-1, unbiased=False),
        torch.ones(1, 2, 2),
        atol=1.0e-5,
    )


def test_predictor_uses_unscaled_standard_normal_position_initialization() -> None:
    torch.manual_seed(17)
    expected = torch.randn(1, 6, 8)
    torch.manual_seed(17)

    predictor = DinoTokenViTPredictor(
        num_patches=2,
        num_frames=3,
        dim=8,
        depth=1,
        heads=2,
        mlp_dim=16,
        dim_head=4,
        dropout=0.0,
        emb_dropout=0.0,
    )

    assert torch.equal(predictor.pos_embedding.detach(), expected)


def test_forward_matches_dino_shifted_target_and_excludes_action_from_loss() -> None:
    model = _tiny_model()
    tokens = torch.arange(1, 1 + 4 * 2 * 4, dtype=torch.float32).reshape(1, 4, 2, 4)
    proprio = torch.arange(1, 1 + 4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    previous_actions = torch.full((1, 4, 2), -100.0)
    current_actions = torch.arange(1, 1 + 4 * 2, dtype=torch.float32).reshape(1, 4, 2)
    full_z = model.encode(
        {"visual": tokens, "proprio": proprio},
        current_actions,
    )
    captured: dict[str, torch.Tensor] = {}

    def identity_predict(self, z: torch.Tensor) -> torch.Tensor:
        captured["z_src"] = z.detach().clone()
        return z

    model.predict = types.MethodType(identity_predict, model)
    losses = model(
        {
            "obs_embedding": tokens,
            "proprio": proprio,
            "actions": previous_actions,
            "current_actions": current_actions,
            "lang_emb": torch.randn(1, 4096),
        }
    )

    z_src = full_z[:, :3]
    z_target = full_z[:, 1:]
    non_action = model.model_dim - model.action_condition_dim
    expected_loss = (z_src[..., :non_action] - z_target[..., :non_action]).square().mean()
    expected_visual_mse = (
        z_src[..., : model.token_dim] - z_target[..., : model.token_dim]
    ).square().mean()

    assert torch.equal(captured["z_src"], z_src)
    assert torch.allclose(losses["_loss"], expected_loss)
    assert torch.allclose(losses["z_loss"], expected_loss)
    assert torch.allclose(losses["hidden_mse"], expected_visual_mse)
    assert losses["teacher_forced_steps"].item() == 3


def test_rollout_reuses_predicted_tokens_and_overwrites_future_actions() -> None:
    model = _tiny_model()
    with torch.no_grad():
        model.action_encoder.patch_embed.weight.copy_(torch.eye(2).unsqueeze(-1))
        model.action_encoder.patch_embed.bias.zero_()
    tokens = torch.zeros(1, 3, 2, 4)
    proprio = torch.zeros(1, 3, 3)
    actions = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]]
    )
    calls: list[torch.Tensor] = []

    def increment_predict(self, z: torch.Tensor) -> torch.Tensor:
        calls.append(z.detach().clone())
        return z + 1.0

    model.predict = types.MethodType(increment_predict, model)

    observations, latent = model.rollout(
        {"visual": tokens, "proprio": proprio},
        actions,
    )

    assert latent.shape[:3] == (1, 6, 2)
    assert observations["visual"].shape == (1, 6, 2, 4)
    assert len(calls) == 3
    assert torch.equal(calls[1][:, -1, :, :4], torch.ones(1, 2, 4))
    action_slice = latent[..., -model.action_condition_dim :]
    assert torch.equal(action_slice[:, 3, 0], actions[:, 3])
    assert torch.equal(action_slice[:, 4, 0], actions[:, 4])
    assert torch.equal(action_slice[:, 3, 0], action_slice[:, 3, 1])


def test_forward_requires_exact_dino_training_window() -> None:
    model = _tiny_model()
    batch = {
        "obs_embedding": torch.zeros(1, 5, 2, 4),
        "proprio": torch.zeros(1, 5, 3),
        "current_actions": torch.zeros(1, 5, 2),
    }

    try:
        model(batch)
    except ValueError as exc:
        assert "num_hist + num_pred" in str(exc)
    else:
        raise AssertionError("DINO shifted training must reject non-four-frame windows")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="upstream DINO mask is CUDA-only")
def test_predictor_is_numerically_identical_to_local_upstream_dino_wm() -> None:
    source = Path(
        "/mnt/data/spoil/workspace/Related_Work/worldmodel/dino_wm/models/vit.py"
    )
    if not source.is_file():
        pytest.skip("local DINO-WM reference checkout is unavailable")
    spec = importlib.util.spec_from_file_location("_dreamervla_dino_reference_vit", source)
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    torch.manual_seed(123)
    upstream = reference.ViTPredictor(
        num_patches=2,
        num_frames=3,
        dim=8,
        depth=2,
        heads=2,
        mlp_dim=16,
        pool="mean",
        dim_head=4,
        dropout=0.0,
        emb_dropout=0.0,
    ).cuda()
    local = DinoTokenViTPredictor(
        num_patches=2,
        num_frames=3,
        dim=8,
        depth=2,
        heads=2,
        mlp_dim=16,
        pool="mean",
        dim_head=4,
        dropout=0.0,
        emb_dropout=0.0,
    ).cuda()
    local.load_state_dict(upstream.state_dict(), strict=True)
    values = torch.randn(3, 6, 8, device="cuda")

    with torch.no_grad():
        expected = upstream(values)
        actual = local(values)

    assert torch.equal(actual, expected)
