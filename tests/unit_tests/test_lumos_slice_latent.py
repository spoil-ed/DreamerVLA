"""MEM-RL-01: latent slicing for group-aligned micro-batching of the LUMOS update.

`_slice_latent` is the batch-dim companion to `_repeat_latent`; it lets the outcome
step process the effective batch in slices without materializing the whole
imagination on GPU.
"""

import torch

from dreamervla.algorithms.ppo.grpo import _slice_latent


def test_slice_latent_tensor_slices_batch_dim():
    x = torch.arange(20).reshape(10, 2)
    assert torch.equal(_slice_latent(x, 2, 6), x[2:6])


def test_slice_latent_nested_dict():
    d = {"a": torch.arange(10), "b": {"c": torch.arange(30).reshape(10, 3)}}
    out = _slice_latent(d, 4, 8)
    assert torch.equal(out["a"], d["a"][4:8])
    assert torch.equal(out["b"]["c"], d["b"]["c"][4:8])


def test_slice_latent_then_concat_roundtrips():
    x = torch.randn(8, 5)
    halves = torch.cat([_slice_latent(x, 0, 4), _slice_latent(x, 4, 8)], dim=0)
    assert torch.equal(halves, x)
