"""PERF-W1 (Q8): reduce_mean_dict must issue a single batched all_reduce.

The distributed collective cannot be exercised without spawning ranks, so these
tests pin the two things that ARE unit-testable:
  1. world_size == 1 path returns the input means unchanged (as floats).
  2. the batched stacking-then-mean logic issues exactly ONE all_reduce and is
     numerically identical to a reference per-key reduction.
"""

from __future__ import annotations

import torch

from dreamervla.runtime.distributed import NopretokenizeSFTDistributedHelper


def _make_helper(world_size: int) -> NopretokenizeSFTDistributedHelper:
    return NopretokenizeSFTDistributedHelper(
        rank=0,
        local_rank=0,
        world_size=world_size,
        strategy="ddp",
        fsdp_mixed_precision="bf16",
        enable_activation_checkpointing=False,
    )


def _reference_reduce_mean_dict(
    helper: NopretokenizeSFTDistributedHelper, metrics: dict[str, float]
) -> dict[str, float]:
    """Reference = the EXISTING per-key path (`reduce_mean` per key).

    Equivalence is "numerically identical to the current implementation", so the
    reference must round-trip each value through the same float32 tensor that
    `reduce_mean` uses (raw Python doubles would NOT match the current code).
    """
    return {key: helper.reduce_mean(value) for key, value in metrics.items()}


def test_world_size_one_returns_input_means_as_floats():
    helper = _make_helper(world_size=1)
    metrics = {"loss": 1.5, "acc": 0, "lr": 3e-4}

    out = helper.reduce_mean_dict(metrics)

    assert out == _reference_reduce_mean_dict(helper, metrics)
    assert all(isinstance(v, float) for v in out.values())
    assert list(out.keys()) == list(metrics.keys())


def test_empty_dict_returns_empty():
    helper = _make_helper(world_size=1)
    assert helper.reduce_mean_dict({}) == {}
    assert helper.reduce_min_max_dict({}) == {}


def test_world_size_one_returns_rank_extrema_as_floats():
    helper = _make_helper(world_size=1)

    out = helper.reduce_min_max_dict({"step": 7, "norm": torch.tensor(1.25)})

    assert out == {
        "step_rank_min": 7.0,
        "step_rank_max": 7.0,
        "norm_rank_min": 1.25,
        "norm_rank_max": 1.25,
    }


def test_rank_extrema_issue_two_batched_collectives(monkeypatch):
    helper = _make_helper(world_size=4)
    monkeypatch.setattr(helper, "_reduce_device", lambda: torch.device("cpu"))
    calls = []

    def _fake_all_reduce(tensor, op=None):  # noqa: ANN001
        calls.append(op)

    monkeypatch.setattr(
        "dreamervla.runtime.distributed.dist.all_reduce", _fake_all_reduce
    )

    helper.reduce_min_max_dict({"a": 1.0, "b": 2.0, "c": 3.0})

    assert calls == [torch.distributed.ReduceOp.MIN, torch.distributed.ReduceOp.MAX]


def test_issues_exactly_one_all_reduce_for_multi_key_dict(monkeypatch):
    """RED-driver: the old per-key implementation calls all_reduce once PER key.

    The batched implementation must call it exactly once regardless of key count.
    """
    helper = _make_helper(world_size=4)
    monkeypatch.setattr(helper, "_reduce_device", lambda: torch.device("cpu"))

    calls = {"n": 0}

    def _fake_all_reduce(tensor, op=None):  # noqa: ANN001
        calls["n"] += 1
        # identity SUM (single rank) so the value path stays meaningful
        return None

    monkeypatch.setattr(
        "dreamervla.runtime.distributed.dist.all_reduce", _fake_all_reduce
    )

    metrics = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    helper.reduce_mean_dict(metrics)

    assert calls["n"] == 1


def test_batched_matches_reference_per_key_reduction(monkeypatch):
    """Stub all_reduce as identity SUM; with world_size==1 divisor is 1.0, so the
    batched result must equal the reference per-key reduction exactly."""
    helper = _make_helper(world_size=1)
    monkeypatch.setattr(helper, "_reduce_device", lambda: torch.device("cpu"))

    metrics = {"loss": 1.5, "acc": 0.25, "lr": 3e-4, "step": 7}

    out = helper.reduce_mean_dict(metrics)
    ref = _reference_reduce_mean_dict(helper, metrics)

    assert out == ref
    assert all(isinstance(v, float) for v in out.values())


def test_distributed_divisor_and_keys_preserved(monkeypatch):
    """With a forced world_size and identity-SUM stub, batched output divides by
    world_size (matching the per-key path's `/ world_size`) and preserves keys."""
    world_size = 4
    helper = _make_helper(world_size=world_size)
    monkeypatch.setattr(helper, "_reduce_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        "dreamervla.runtime.distributed.dist.all_reduce",
        lambda tensor, op=None: None,  # identity SUM (single rank)
    )

    metrics = {"a": 8.0, "b": 4.0, "c": 0.0}
    out = helper.reduce_mean_dict(metrics)

    expected = {key: float(value) / world_size for key, value in metrics.items()}
    assert list(out.keys()) == list(metrics.keys())
    for key in metrics:
        assert out[key] == expected[key]
