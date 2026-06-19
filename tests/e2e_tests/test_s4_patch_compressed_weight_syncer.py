from __future__ import annotations

import uuid

import ray
import torch

from dreamervla.scheduler.cluster import Cluster


def test_patch_weight_syncer_applies_single_version_delta() -> None:
    from dreamervla.hybrid_engines.weight_syncer.patch import PatchWeightSyncer

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        syncer = PatchWeightSyncer(store_name=f"patch-weight-store-{uuid.uuid4().hex}")
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)

        with torch.no_grad():
            source.weight.fill_(1.0)
            source.bias.zero_()
        syncer.push("policy", source.state_dict(), version=1)
        assert syncer.pull("policy", target, local_version=0) == 1

        with torch.no_grad():
            source.bias.fill_(2.0)
        syncer.push("policy", source.state_dict(), version=2)
        meta_version, meta = ray.get(syncer._store.get.remote("policy::meta"))
        if isinstance(meta, ray.ObjectRef):
            meta = ray.get(meta)
        assert meta_version == 2
        assert meta["patch_keys"] == ["bias"]

        assert syncer.pull("policy", target, local_version=1) == 2
        assert torch.allclose(target.weight, torch.ones_like(target.weight))
        assert torch.allclose(target.bias, torch.full_like(target.bias, 2.0))
    finally:
        cluster.shutdown()


def test_compressed_weight_syncer_stores_transport_dtype_and_round_trips() -> None:
    from dreamervla.hybrid_engines.weight_syncer.compression import CompressedWeightSyncer

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        syncer = CompressedWeightSyncer(
            store_name=f"compressed-weight-store-{uuid.uuid4().hex}",
            transport_dtype="fp16",
        )
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        with torch.no_grad():
            source.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
            source.bias.copy_(torch.tensor([1.0, 2.0]))
            target.weight.zero_()
            target.bias.zero_()

        syncer.push("policy", source.state_dict(), version=3)
        version, packed = ray.get(syncer._store.get.remote("policy"))
        if isinstance(packed, ray.ObjectRef):
            packed = ray.get(packed)
        assert version == 3
        assert packed["weight"].tensor.dtype is torch.float16

        assert syncer.pull("policy", target, local_version=0) == 3
        assert torch.allclose(target.weight, source.weight)
        assert torch.allclose(target.bias, source.bias)
    finally:
        cluster.shutdown()
