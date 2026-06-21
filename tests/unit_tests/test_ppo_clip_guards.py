"""A2/A3 numerical-stability guards on the shared PPO primitives.

Locks the opt-in `clip_log_ratio` (log-ratio clamp before exp) and `clip_ratio_c`
(dual-clip) behaviour, and pins that both are no-ops when left unset so enabling
them in a config is the only thing that changes numerics.
"""

from __future__ import annotations

import torch

from dreamervla.algorithms.ppo.grpo import _ppo_clip_term, _ppo_ratio


def test_ppo_ratio_default_off_is_plain_exp() -> None:
    lp = torch.tensor([2.0, -3.0])
    olp = torch.tensor([0.0, 0.5])
    assert torch.allclose(_ppo_ratio(lp, olp), torch.exp(lp - olp))


def test_ppo_ratio_clamps_log_ratio_before_exp() -> None:
    # summed-trajectory log-ratios can blow up; clamp bounds exp() to [e^-c, e^c].
    lp = torch.tensor([20.0, -20.0, 3.0])
    olp = torch.zeros(3)
    out = _ppo_ratio(lp, olp, clip_log_ratio=10.0)
    assert torch.allclose(out, torch.exp(torch.tensor([10.0, -10.0, 3.0])))


def test_ppo_clip_term_default_off_matches_single_clip() -> None:
    ratio = torch.tensor([5.0])
    adv = torch.tensor([-1.0])
    # ratio_clipped = clamp(5, 0.8, 1.28) = 1.28; max(1*5, 1*1.28) = 5
    assert torch.allclose(_ppo_clip_term(ratio, adv, 0.2, 0.28), torch.tensor([5.0]))


def test_ppo_clip_term_dual_clip_caps_negative_adv() -> None:
    ratio = torch.tensor([5.0])
    adv = torch.tensor([-1.0])
    # unbounded surrogate would be 5; dual-clip caps it at clip_ratio_c * |adv| = 3.
    out = _ppo_clip_term(ratio, adv, 0.2, 0.28, clip_ratio_c=3.0)
    assert torch.allclose(out, torch.tensor([3.0]))


def test_ppo_clip_term_dual_clip_is_noop_below_cap() -> None:
    ratio = torch.tensor([1.1])
    adv = torch.tensor([-1.0])
    base = _ppo_clip_term(ratio, adv, 0.2, 0.28)
    dual = _ppo_clip_term(ratio, adv, 0.2, 0.28, clip_ratio_c=3.0)
    # surrogate 1.1 < cap 3.0 -> dual-clip leaves it untouched.
    assert torch.allclose(base, dual)
