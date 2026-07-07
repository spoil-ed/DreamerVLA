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
