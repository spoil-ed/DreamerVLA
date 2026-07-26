from __future__ import annotations

import torch
import torch.nn as nn

from dreamervla.algorithms.critic.latent_success_classifier import LatentSuccessClassifier


def _spatial(**kw):
    base = dict(
        latent_dim=4,
        token_dim=4,
        token_count=2,
        token_pool="mean",
        head_type="spatial_tf",
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
    )
    base.update(kw)
    return LatentSuccessClassifier(**base)


def test_spatial_tf_has_vision_input_norm_by_default():
    model = _spatial()
    assert isinstance(model.vision_norm, nn.LayerNorm)
    assert model.vision_norm.normalized_shape == (4,)


def test_spatial_tf_vision_norm_makes_large_scale_input_finite():
    model = _spatial().eval()
    big = torch.randn(2, 2, 2, 4) * 1e4
    logits = model(big)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_spatial_tf_vision_norm_can_be_disabled():
    model = _spatial(vision_input_norm=False)
    assert isinstance(model.vision_norm, nn.Identity)


def test_spatial_tf_concats_proprio_and_language_on_each_vision_token():
    model = _spatial(
        latent_dim=7,
        proprio_dim=2,
        proprio_emb_dim=1,
        lang_dim=3,
        lang_emb_dim=2,
    ).eval()
    vision = torch.randn(2, 2, 2, 4)
    proprio = torch.randn(2, 2, 2)
    language = torch.randn(2, 3)

    conditioned = model._spatial_input_tokens(
        vision,
        proprio=proprio,
        lang_emb=language,
    )

    assert conditioned.shape == (2, 2, 2, 7)
    assert model.vision_norm.normalized_shape == (7,)
    assert model.vision_proj.in_features == 7
    assert torch.allclose(conditioned[..., :4], vision)
    assert torch.allclose(conditioned[:, :, 0, 4:], conditioned[:, :, 1, 4:])


def test_spatial_tf_accepts_wm_tokens_with_embedded_proprio():
    model = _spatial(
        latent_dim=7,
        proprio_dim=2,
        proprio_emb_dim=1,
        lang_dim=3,
        lang_emb_dim=2,
    ).eval()
    wm_observation = torch.randn(2, 2, 2, 5)

    conditioned = model._spatial_input_tokens(
        wm_observation,
        proprio=None,
        lang_emb=torch.randn(2, 3),
    )

    assert conditioned.shape == (2, 2, 2, 7)
    assert torch.allclose(conditioned[..., :5], wm_observation)
