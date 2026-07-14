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

import dreamervla.workers.replay.replay_worker as replay_worker_module
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


class _InitialConditionSpyReplay:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict]] = []

    def sample_initial_conditions(self, batch_size, **kwargs):
        self.calls.append((int(batch_size), dict(kwargs)))
        return {"task_ids": [0] * int(batch_size)}


class _SeedSpyReplay:
    episodes = [object(), object()]
    num_transitions = 72

    @staticmethod
    def task_stats(task_ids=None):
        return {
            str(task_id): {"episodes": 1, "transitions": 36}
            for task_id in (task_ids or ())
        }


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


def test_initial_condition_options_reach_replay_through_ray_worker_facade():
    spy = _InitialConditionSpyReplay()
    worker = ReplayWorker.__new__(ReplayWorker)
    worker.replay = spy

    worker.sample_initial_conditions(
        16,
        task_ids=tuple(range(10)),
        keys=("obs_embedding", "lang_emb", "proprio"),
    )

    assert spy.calls == [
        (
            16,
            {
                "task_ids": tuple(range(10)),
                "keys": ("obs_embedding", "lang_emb", "proprio"),
            },
        )
    ]


def test_failure_selector_reaches_replay_through_ray_worker_facade():
    spy = _InitialConditionSpyReplay()
    worker = ReplayWorker.__new__(ReplayWorker)
    worker.replay = spy

    worker.sample_initial_conditions(
        8,
        keys=("obs_embedding",),
        selector="failed_episode_start",
    )

    assert spy.calls == [
        (
            8,
            {
                "task_ids": None,
                "keys": ("obs_embedding",),
                "selector": "failed_episode_start",
            },
        )
    ]


def test_ray_replay_worker_seeds_official_data_through_shared_loader(monkeypatch):
    calls: list[dict] = []

    def _seed(replay, **kwargs):
        calls.append({"replay": replay, **kwargs})
        return 2

    monkeypatch.setattr(replay_worker_module, "seed_replay_from_offline", _seed)
    replay = _SeedSpyReplay()
    worker = ReplayWorker.__new__(ReplayWorker)
    worker.replay = replay

    metrics = worker.seed_from_offline(
        {
            "data_dir": "/official/reward",
            "hidden_dir": "/official/hidden",
            "task_id": None,
            "infer_task_id_from_shard": True,
            "task_ids": [0, 1],
            "max_episodes_per_task": None,
            "require_reference_complete": False,
        }
    )

    assert calls == [
        {
            "replay": replay,
            "data_dir": "/official/reward",
            "hidden_dir": "/official/hidden",
            "default_task_id": None,
            "infer_task_id_from_shard": True,
            "max_episodes_per_task": None,
            "require_reference_complete": False,
        }
    ]
    assert metrics["replay_buffer/seeded_episodes"] == 2.0
    assert metrics["replay_buffer/seeded_transitions"] == 72.0
    assert metrics["replay_buffer/seeded_task_count"] == 2.0
