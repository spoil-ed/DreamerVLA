"""PERF-Q1: EMAHelper.step must fuse the EMA blend with torch._foreach_* and stay
numerically identical to the per-parameter mul_/add_ reference loop.

Plan: docs/plans/2026-06-23-perf-q1-ema-foreach.md
"""

from __future__ import annotations

import torch
from torch import nn

from dreamervla.utils.ema import EMAHelper


class _MixedModule(nn.Module):
    """Several trainable params with mixed shapes and dtypes (exercises bucketing)."""

    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.a = nn.Parameter(torch.randn(4, 3, generator=g))  # fp32
        self.b = nn.Parameter(torch.randn(5, generator=g))  # fp32
        self.c = nn.Parameter(torch.randn(2, 2, generator=g).double())  # fp64


def _reference_shadow(model: nn.Module, decay: float, n_steps: int) -> dict[str, torch.Tensor]:
    """EMA shadows via the explicit per-parameter mul_/add_ loop (today's math)."""
    shadow = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    for step in range(n_steps):
        _mutate(model, step)
        for name, p in model.named_parameters():
            if name in shadow:
                shadow[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    return shadow


def _mutate(model: nn.Module, step: int) -> None:
    """Perturb params between updates so successive EMA steps differ."""
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            p.add_(torch.full_like(p, 0.01 * (step + 1) * (i + 1)))


def test_ema_step_matches_reference_loop_and_uses_foreach(monkeypatch) -> None:
    decay = 0.97
    n_steps = 5

    # Two models seeded identically so reference and subject see the same param path.
    ref_model = _MixedModule()
    sub_model = _MixedModule()
    for rp, sp in zip(ref_model.parameters(), sub_model.parameters(), strict=True):
        sp.data.copy_(rp.data)

    ref_shadow = _reference_shadow(ref_model, decay, n_steps)

    # Spy on the fused multi-tensor kernels: RED if step() never calls them.
    calls = {"mul": 0, "add": 0}
    real_mul = torch._foreach_mul_
    real_add = torch._foreach_add_

    def spy_mul(*args, **kwargs):
        calls["mul"] += 1
        return real_mul(*args, **kwargs)

    def spy_add(*args, **kwargs):
        calls["add"] += 1
        return real_add(*args, **kwargs)

    monkeypatch.setattr(torch, "_foreach_mul_", spy_mul)
    monkeypatch.setattr(torch, "_foreach_add_", spy_add)

    helper = EMAHelper(sub_model, decay=decay, update_after_step=0)
    for step in range(n_steps):
        _mutate(sub_model, step)
        helper.step(sub_model)

    # The blend must be fused via torch._foreach_*.
    assert calls["mul"] > 0, "EMA blend did not call torch._foreach_mul_"
    assert calls["add"] > 0, "EMA blend did not call torch._foreach_add_"

    # Numerical equivalence gate: identical two ops, same order, same dtype per tensor
    # => bit-identical result. atol=0.
    assert set(helper.shadow) == set(ref_shadow)
    for name, sub_t in helper.shadow.items():
        ref_t = ref_shadow[name]
        assert sub_t.dtype == ref_t.dtype
        assert torch.equal(sub_t, ref_t), f"shadow[{name}] diverged from reference loop"


def test_ema_warmup_copy_branch_unchanged() -> None:
    """During warmup the shadow is a straight copy of the live params (no blend)."""
    model = _MixedModule()
    helper = EMAHelper(model, decay=0.99, update_after_step=3)
    _mutate(model, 0)
    helper.step(model)  # optimization_step=1 <= 3 => copy branch
    for name, p in model.named_parameters():
        if name in helper.shadow:
            assert torch.equal(helper.shadow[name], p.detach())
