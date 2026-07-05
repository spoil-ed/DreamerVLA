from __future__ import annotations

import math

import torch


def test_classifier_depth_ablation_profiles_params_on_cpu() -> None:
    from dreamervla.diagnostics.classifier_depth_ablation import profile_one

    nparam, mem_mb, fwd_ms = profile_one(
        num_layers=1,
        hidden_dim=16,
        token_count=2,
        token_dim=4,
        window=2,
        batch=1,
        device=torch.device("cpu"),
    )

    assert nparam > 0
    assert math.isnan(mem_mb)
    assert math.isnan(fwd_ms)
