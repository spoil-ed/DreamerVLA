"""Unit tests for the bucketed weight-sync partition logic (ray-free)."""

from __future__ import annotations

import torch

from dreamervla.hybrid_engines.weight_syncer.bucket import bucket_state_dict


def _bucket_bytes(bucket: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in bucket.values())


def test_bucket_state_dict_preserves_every_key_in_order() -> None:
    state_dict = {
        "a": torch.zeros(10, dtype=torch.float32),  # 40 bytes
        "b": torch.zeros(10, dtype=torch.float32),  # 40 bytes
        "c": torch.zeros(10, dtype=torch.float32),  # 40 bytes
    }

    buckets = bucket_state_dict(state_dict, bucket_bytes=80)

    flat = [key for bucket in buckets for key in bucket]
    assert flat == ["a", "b", "c"]
    assert all(bucket for bucket in buckets)  # no empty buckets


def test_bucket_state_dict_respects_byte_budget() -> None:
    state_dict = {f"p{i}": torch.zeros(10, dtype=torch.float32) for i in range(3)}  # 40 bytes each

    buckets = bucket_state_dict(state_dict, bucket_bytes=80)

    assert len(buckets) >= 2
    for bucket in buckets:
        assert len(bucket) == 1 or _bucket_bytes(bucket) <= 80


def test_bucket_state_dict_oversized_tensor_gets_its_own_bucket() -> None:
    state_dict = {"big": torch.zeros(100, dtype=torch.float32)}  # 400 bytes > budget

    buckets = bucket_state_dict(state_dict, bucket_bytes=80)

    assert len(buckets) == 1
    assert list(buckets[0]) == ["big"]
