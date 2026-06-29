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
