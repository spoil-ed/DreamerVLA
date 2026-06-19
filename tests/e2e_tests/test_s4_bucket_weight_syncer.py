from __future__ import annotations

import ray
import torch

from dreamervla.scheduler.cluster import Cluster


def test_bucket_weight_syncer_round_trips_into_model_and_versions() -> None:
    from dreamervla.hybrid_engines.weight_syncer.bucket import BucketWeightSyncer

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        source = torch.nn.Linear(8, 8)
        target = torch.nn.Linear(8, 8)
        with torch.no_grad():
            source.weight.fill_(2.0)
            source.bias.fill_(1.0)
            target.weight.zero_()
            target.bias.zero_()

        # bucket_bytes small enough to force more than one bucket
        syncer = BucketWeightSyncer(
            store_name="test_bucket_weight_store",
            bucket_bytes=64,
        )
        syncer.push("policy", source.state_dict(), version=1)

        # weight (256B) and bias (32B) should land in separate buckets
        meta_version, meta = ray.get(syncer._store.get.remote("policy::meta"))
        if isinstance(meta, ray.ObjectRef):
            meta = ray.get(meta)
        assert meta_version == 1
        assert int(meta["num_buckets"].item()) >= 2

        assert syncer.pull("policy", target, local_version=0) == 1
        assert torch.allclose(target.weight, source.weight)
        assert torch.allclose(target.bias, source.bias)

        # monotonic: pulling at an equal/newer local version is a no-op
        assert syncer.pull("policy", target, local_version=1) is None
    finally:
        cluster.shutdown()


def test_bucket_weight_syncer_matches_object_store_syncer() -> None:
    from dreamervla.hybrid_engines.weight_syncer.bucket import BucketWeightSyncer
    from dreamervla.hybrid_engines.weight_syncer.objectstore import (
        ObjectStoreWeightSyncer,
    )

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        source = torch.nn.Linear(4, 6)
        with torch.no_grad():
            source.weight.normal_()
            source.bias.normal_()

        bucket_target = torch.nn.Linear(4, 6)
        store_target = torch.nn.Linear(4, 6)

        BucketWeightSyncer(store_name="parity_bucket_store", bucket_bytes=32).push(
            "policy", source.state_dict(), version=5
        )
        ObjectStoreWeightSyncer(store_name="parity_store").push(
            "policy", source.state_dict(), version=5
        )

        BucketWeightSyncer(store_name="parity_bucket_store", bucket_bytes=32).pull(
            "policy", bucket_target, local_version=0
        )
        ObjectStoreWeightSyncer(store_name="parity_store").pull(
            "policy", store_target, local_version=0
        )

        assert torch.allclose(bucket_target.weight, store_target.weight)
        assert torch.allclose(bucket_target.bias, store_target.bias)
    finally:
        cluster.shutdown()
