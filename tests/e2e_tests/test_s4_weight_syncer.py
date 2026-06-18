from __future__ import annotations

import ray
import torch

from dreamervla.scheduler.cluster import Cluster


def test_object_store_weight_syncer_push_pull_versions() -> None:
    try:
        from dreamervla.hybrid_engines.weight_syncer.objectstore import (
            ObjectStoreWeightSyncer,
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("ObjectStoreWeightSyncer module should exist") from exc

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        source = torch.nn.Linear(2, 2)
        target = torch.nn.Linear(2, 2)
        with torch.no_grad():
            source.weight.fill_(2.0)
            source.bias.fill_(1.0)
            target.weight.zero_()
            target.bias.zero_()

        syncer = ObjectStoreWeightSyncer(store_name="test_weight_store")
        syncer.push("policy", source.state_dict(), version=1)

        version, state_ref = ray.get(syncer.store.get.remote("policy"))
        assert version == 1
        assert isinstance(state_ref, ray.ObjectRef)

        assert syncer.pull("policy", target, local_version=0) == 1
        assert torch.allclose(target.weight, source.weight)
        assert torch.allclose(target.bias, source.bias)
        assert syncer.pull("policy", target, local_version=1) is None
    finally:
        cluster.shutdown()
