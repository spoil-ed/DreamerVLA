"""Regression: ReplayClient.sample must forward the Phase 4 staleness gate to
the underlying replay, but only when it is set.

The cotrain Ray learner reaches the replay through ReplayClient (a thin facade
over an in-process OnlineReplay or a Ray ReplayWorker actor). When Phase 4 added
``staleness_threshold`` to OnlineReplay/ReplayWorker, the facade was not updated,
so a real run died with::

    TypeError: ReplayClient.sample() got an unexpected keyword argument 'staleness_threshold'

The default path must stay a bare 1-arg call so minimal replay backends (e.g.
test doubles) that don't know about staleness still work.
"""

from __future__ import annotations

from dreamervla.workers.actor.learner_worker import ReplayClient
from dreamervla.workers.replay.replay_worker import ReplayWorker


class _SpyReplay:
    """Records how ``sample`` was invoked. No ``.remote`` → called directly."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def sample(self, batch_size, **kwargs):
        self.calls.append((batch_size, dict(kwargs)))
        return {"batch_size": batch_size, **kwargs}


class _MinimalReplay:
    """A backend that does NOT accept staleness_threshold (like a test double)."""

    def sample(self, batch_size):
        return {"batch_size": batch_size}


class _ClassifierSpyReplay:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict]] = []

    def sample_classifier_windows(self, batch_size, **kwargs):
        self.calls.append((int(batch_size), dict(kwargs)))
        return {"batch_size": int(batch_size)}


def test_default_path_sends_no_staleness_kwarg():
    spy = _SpyReplay()
    ReplayClient(spy).sample(7)
    assert spy.calls == [(7, {})]


def test_minimal_backend_works_on_default_path():
    # Must not raise: default path forwards no staleness kwarg.
    out = ReplayClient(_MinimalReplay()).sample(3)
    assert out == {"batch_size": 3}


def test_forwards_staleness_threshold_when_set():
    spy = _SpyReplay()
    ReplayClient(spy).sample(5, staleness_threshold=2)
    assert spy.calls == [(5, {"staleness_threshold": 2})]


def test_none_threshold_is_treated_as_default():
    spy = _SpyReplay()
    ReplayClient(spy).sample(4, staleness_threshold=None)
    assert spy.calls == [(4, {})]


def test_forwards_include_images_when_explicitly_disabled():
    spy = _SpyReplay()
    ReplayClient(spy).sample(6, include_images=False)
    assert spy.calls == [(6, {"include_images": False})]


def test_classifier_options_reach_local_replay_through_client():
    spy = _ClassifierSpyReplay()

    ReplayClient(spy).sample_classifier_windows(
        8,
        window=4,
        chunk_size=2,
        chunk_pool="last",
        early_neg_stride=3,
        sampling_protocol="wmpo",
        balance_batches=True,
    )

    assert spy.calls == [
        (
            8,
            {
                "window": 4,
                "chunk_size": 2,
                "chunk_pool": "last",
                "early_neg_stride": 3,
                "sampling_protocol": "wmpo",
                "balance_batches": True,
            },
        )
    ]


def test_classifier_options_reach_replay_through_ray_worker_facade():
    spy = _ClassifierSpyReplay()
    worker = ReplayWorker.__new__(ReplayWorker)
    worker.replay = spy

    worker.sample_classifier_windows(
        6,
        window=5,
        chunk_size=2,
        chunk_pool="mean",
        early_neg_stride=7,
        sampling_protocol="wmpo",
        balance_batches=True,
    )

    assert spy.calls == [
        (
            6,
            {
                "window": 5,
                "chunk_size": 2,
                "chunk_pool": "mean",
                "early_neg_stride": 7,
                "sampling_protocol": "wmpo",
                "balance_batches": True,
            },
        )
    ]
