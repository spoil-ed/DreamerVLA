from __future__ import annotations

import types

import numpy as np
import torch

from dreamervla.diagnostics.eval_dino_token_wm import (
    deterministic_window_starts,
    one_step_token_predictions,
    token_prediction_metrics,
)
from dreamervla.models.embodiment.world_model import DinoTokenWorldModel


def _tiny_model() -> DinoTokenWorldModel:
    return DinoTokenWorldModel(
        token_count=2,
        token_dim=4,
        action_dim=2,
        proprio_dim=3,
        action_emb_dim=2,
        proprio_emb_dim=2,
        num_hist=3,
        num_pred=1,
        depth=1,
        heads=2,
        dim_head=2,
        mlp_dim=8,
        dropout=0.0,
    )


def test_one_step_diagnostic_compares_same_targets_with_persistence() -> None:
    model = _tiny_model()
    tokens = torch.arange(1, 1 + 4 * 2 * 4, dtype=torch.float32).reshape(1, 4, 2, 4)
    proprio = torch.zeros(1, 4, 3)
    actions = torch.zeros(1, 4, 2)

    def identity_predict(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    model.predict = types.MethodType(identity_predict, model)

    predicted, target, persistence = one_step_token_predictions(
        model,
        tokens=tokens,
        proprio=proprio,
        actions=actions,
    )

    normalized = model.token_norm(tokens)
    assert torch.equal(predicted, normalized[:, :3])
    assert torch.equal(persistence, normalized[:, :3])
    assert torch.equal(target, normalized[:, 1:])


def test_token_prediction_metrics_flatten_each_predicted_frame() -> None:
    target = torch.tensor([[[[1.0, 0.0]], [[0.0, 2.0]]]])
    predicted = target.clone()
    predicted[:, 1] = 0.0

    metrics = token_prediction_metrics(predicted, target)

    assert np.allclose(metrics["cos"], [1.0, 0.0])
    assert np.allclose(metrics["mse"], [0.0, 2.0])


def test_deterministic_window_starts_cover_the_demo() -> None:
    starts = deterministic_window_starts(length=20, window=4, max_windows=5)

    assert starts == [0, 4, 8, 12, 16]
