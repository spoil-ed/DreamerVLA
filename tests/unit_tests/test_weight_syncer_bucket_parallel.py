"""Q11: bucketed weight sync must batch per-bucket ``ray.get`` into one call.

These tests stub the module-level ``ray`` in ``bucket.py`` with a fake object
store + actor, so they run ray-free. They assert two things:

(a) FINAL EFFECT is unchanged — the same per-bucket ``set`` keys/values land in
    the store on ``push``, and ``pull`` reconstructs the identical merged
    state_dict and version.
(b) ``ray.get`` is invoked ONCE for the bucket batch rather than once-per-bucket
    (the call-count assertion is the RED driver for the parallel-get refactor).
"""

from __future__ import annotations

from typing import Any

import torch

from dreamervla.hybrid_engines.weight_syncer import bucket as bucket_mod
from dreamervla.hybrid_engines.weight_syncer.bucket import BucketWeightSyncer


class _FakeObjectRef:
    """Stand-in for ``ray.ObjectRef`` wrapping an in-process value."""

    def __init__(self, value: Any) -> None:
        self.value = value


class _FakeRemoteCall:
    """Deferred actor method call; resolved when passed to ``ray.get``."""

    def __init__(self, fn: Any, args: tuple[Any, ...]) -> None:
        self.fn = fn
        self.args = args

    def run(self) -> Any:
        return self.fn(*self.args)


class _FakeMethod:
    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def remote(self, *args: Any) -> _FakeRemoteCall:
        return _FakeRemoteCall(self._fn, args)


class _FakeStore:
    """Mirrors ``_WeightStore``: version-gated ``set`` + ``get``."""

    def __init__(self) -> None:
        self.items: dict[str, tuple[int, Any]] = {}

    def _set(self, key: str, version: int, state_dict: Any) -> None:
        current = self.items.get(str(key))
        if current is None or int(version) >= int(current[0]):
            # mimic ray.put: wrap stored value in an ObjectRef
            self.items[str(key)] = (int(version), _FakeObjectRef(state_dict))

    def _get(self, key: str) -> tuple[int, Any] | None:
        return self.items.get(str(key))

    def __getattr__(self, name: str) -> _FakeMethod:
        if name == "set":
            return _FakeMethod(self._set)
        if name == "get":
            return _FakeMethod(self._get)
        raise AttributeError(name)


class _FakeRay:
    """Minimal ``ray`` shim recording every ``ray.get`` invocation."""

    ObjectRef = _FakeObjectRef

    def __init__(self) -> None:
        self.get_calls = 0
        self.batched_get_calls = 0  # ray.get(list/tuple) — the batched form

    def get(self, refs: Any) -> Any:
        self.get_calls += 1
        if isinstance(refs, (list, tuple)):
            self.batched_get_calls += 1
            return [self._resolve_one(r) for r in refs]
        return self._resolve_one(refs)

    @staticmethod
    def _resolve_one(ref: Any) -> Any:
        if isinstance(ref, _FakeRemoteCall):
            return ref.run()
        if isinstance(ref, _FakeObjectRef):
            return ref.value
        return ref


def _install_fake_ray(monkeypatch) -> _FakeRay:
    fake = _FakeRay()
    monkeypatch.setattr(bucket_mod, "ray", fake)
    return fake


def _make_syncer(store: _FakeStore) -> BucketWeightSyncer:
    syncer = BucketWeightSyncer.__new__(BucketWeightSyncer)
    syncer.store_name = "fake"
    syncer.bucket_bytes = 80  # 40 bytes/tensor -> multiple buckets
    syncer._store = store
    return syncer


def _multi_bucket_state() -> dict[str, torch.Tensor]:
    # 40 bytes each, budget 80 -> 2 keys/bucket, > 2 buckets total.
    return {f"p{i}": torch.full((10,), float(i), dtype=torch.float32) for i in range(5)}


def test_push_batches_bucket_set_into_one_ray_get(monkeypatch) -> None:
    fake = _install_fake_ray(monkeypatch)
    store = _FakeStore()
    syncer = _make_syncer(store)
    state = _multi_bucket_state()

    syncer.push("policy", state, version=3)

    # FINAL EFFECT: meta + per-bucket sets present, every key round-tripped.
    meta_v, meta_ref = store.items["policy::meta"]
    assert meta_v == 3
    num_buckets = int(meta_ref.value["num_buckets"].item())
    assert num_buckets >= 3  # multiple buckets actually exercised

    merged: dict[str, torch.Tensor] = {}
    for i in range(num_buckets):
        v, ref = store.items[f"policy::b{i}"]
        assert v == 3
        merged.update(ref.value)
    assert set(merged) == set(state)
    for k in state:
        assert torch.equal(merged[k], state[k])

    # BATCHED: the per-bucket ``set`` refs are awaited in a SINGLE
    # ``ray.get([...])`` call, not one round-trip per bucket. The meta ``set``
    # stays a separate scalar get submitted after the bucket batch.
    assert fake.batched_get_calls == 1
    # total gets == 1 batched bucket get + 1 scalar meta get.
    assert fake.get_calls == 2


def test_pull_batches_bucket_get_into_one_ray_get(monkeypatch) -> None:
    fake = _install_fake_ray(monkeypatch)
    store = _FakeStore()
    push_syncer = _make_syncer(store)
    state = _multi_bucket_state()
    push_syncer.push("policy", state, version=7)

    num_buckets = int(store.items["policy::meta"][1].value["num_buckets"].item())
    assert num_buckets >= 3

    # Fresh ray shim so pull's get-count is isolated from push.
    fake = _install_fake_ray(monkeypatch)
    pull_syncer = _make_syncer(store)
    model = torch.nn.Module()
    for name, value in state.items():
        model.register_buffer(name, torch.zeros_like(value))

    returned = pull_syncer.pull("policy", model, local_version=0)

    # FINAL EFFECT: identical merged state loaded, correct version returned.
    assert returned == 7
    loaded = dict(model.named_buffers())
    assert set(loaded) == set(state)
    for k in state:
        assert torch.equal(loaded[k], state[k])

    # BATCHED: the per-bucket ``get`` refs are awaited in a SINGLE
    # ``ray.get([...])`` call, not one round-trip per bucket.
    assert fake.batched_get_calls == 1
    # total gets == meta get.remote (1) + meta _resolve (1) + batched bucket
    # get (1) + per-bucket _resolve unwraps (out of scope for Q11).
    assert fake.get_calls == 3 + num_buckets
