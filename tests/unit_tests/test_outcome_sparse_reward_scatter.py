"""PERF-Q5: vectorizing the sparse outcome-reward build must be bit-for-bit equal.

``_build_reward_tensor`` places a sparse 1.0 at ``finish_step`` for every complete
rollout and zero elsewhere. The audit replaces its Python loop + per-element
``.item()`` with a single ``scatter_``. This is a pure-numerics equivalence gate:
the vectorized output must equal the original loop's output with atol=0
(``torch.equal``) across boundary / clamp / incomplete-row cases.
"""

import pytest
import torch

from dreamervla.algorithms.ppo.outcome import _build_reward_tensor


def _loop_oracle(*, batch, max_steps, finish_step, complete):
    """The ORIGINAL Python-loop reward build, kept as an independent oracle."""
    reward = torch.zeros((batch, max_steps), dtype=torch.float32)
    if max_steps <= 0:
        return reward
    finish = finish_step.detach().cpu().long().clamp(min=0, max=max_steps - 1)
    comp = complete.detach().cpu().bool()
    for i in range(batch):
        if comp[i].item():
            reward[i, finish[i].item()] = 1.0
    return reward


def test_scatter_matches_loop_representative():
    batch, max_steps = 6, 5
    # Row 0: complete at boundary 0. Row 1: incomplete (finish nonzero -> all zero).
    # Row 2: complete mid. Row 3: complete at last column. Row 4: complete, finish
    # OUT OF RANGE -> must clamp to max_steps-1. Row 5: incomplete at boundary.
    finish_step = torch.tensor([0, 3, 2, 4, 99, 4])
    complete = torch.tensor([True, False, True, True, True, False])
    out = _build_reward_tensor(
        batch=batch,
        max_steps=max_steps,
        chunk_size=2,
        finish_step=finish_step,
        complete=complete,
    )
    ref = _loop_oracle(
        batch=batch,
        max_steps=max_steps,
        finish_step=finish_step,
        complete=complete,
    )
    assert torch.equal(out, ref), (out, ref)
    assert out.dtype == torch.float32 and out.shape == (batch, max_steps)


def test_scatter_matches_loop_negative_finish_clamps():
    # A negative finish index must clamp to 0 (matches the loop's clamp(min=0)).
    batch, max_steps = 3, 4
    finish_step = torch.tensor([-7, 1, 2])
    complete = torch.tensor([True, True, False])
    out = _build_reward_tensor(
        batch=batch,
        max_steps=max_steps,
        chunk_size=1,
        finish_step=finish_step,
        complete=complete,
    )
    ref = _loop_oracle(
        batch=batch,
        max_steps=max_steps,
        finish_step=finish_step,
        complete=complete,
    )
    assert torch.equal(out, ref), (out, ref)


def test_scatter_all_incomplete_is_zero():
    batch, max_steps = 4, 3
    finish_step = torch.tensor([0, 1, 2, 0])
    complete = torch.zeros(batch, dtype=torch.bool)
    out = _build_reward_tensor(
        batch=batch,
        max_steps=max_steps,
        chunk_size=1,
        finish_step=finish_step,
        complete=complete,
    )
    assert torch.equal(out, torch.zeros((batch, max_steps), dtype=torch.float32))


def test_scatter_rejects_zero_max_steps():
    with pytest.raises(ValueError, match="max_steps"):
        _build_reward_tensor(
            batch=3,
            max_steps=0,
            chunk_size=1,
            finish_step=torch.tensor([0, 0, 0]),
            complete=torch.tensor([True, True, True]),
        )
