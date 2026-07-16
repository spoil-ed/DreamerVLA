from __future__ import annotations

from typing import Any

import pytest
from omegaconf import OmegaConf

import dreamervla.train as train
from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.success_classifier_training_runner import (
    SuccessClassifierTrainingRunner,
)


def _run_with_dummy_runner(monkeypatch: pytest.MonkeyPatch, runner_cls: type) -> None:
    monkeypatch.setattr(train, "validate_cfg", lambda cfg: cfg)
    monkeypatch.setattr(train.hydra.utils, "get_class", lambda _target: runner_cls)
    train.run(OmegaConf.create({"_target_": "dummy.Runner", "training": {}}))


def test_train_run_cleans_up_when_setup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class DummyRunner:
        def __init__(self, _cfg: Any) -> None:
            events.append("init")

        def setup(self) -> None:
            events.append("setup")
            raise RuntimeError("setup failed")

        def execute(self) -> None:
            events.append("execute")

        def teardown(self) -> None:
            events.append("teardown")

        def teardown_after_setup_failure(self) -> None:
            events.append("setup_failure_cleanup")

    with pytest.raises(RuntimeError, match="setup failed"):
        _run_with_dummy_runner(monkeypatch, DummyRunner)

    assert events == ["init", "setup", "setup_failure_cleanup"]


def test_train_run_preserves_setup_error_when_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class DummyRunner:
        def __init__(self, _cfg: Any) -> None:
            return None

        def setup(self) -> None:
            raise RuntimeError("primary setup failure")

        def execute(self) -> None:
            raise AssertionError("execute must not run")

        def teardown(self) -> None:
            raise AssertionError("normal teardown must not run")

        def teardown_after_setup_failure(self) -> None:
            events.append("setup_failure_cleanup")
            raise ValueError("secondary cleanup failure")

    with pytest.raises(RuntimeError, match="primary setup failure"):
        _run_with_dummy_runner(monkeypatch, DummyRunner)

    assert events == ["setup_failure_cleanup"]


class _CleanupTrackingDistributed:
    def __init__(self) -> None:
        self.barrier_calls = 0
        self.cleanup_calls = 0

    def barrier(self) -> None:
        self.barrier_calls += 1

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class _FailingLocalTeardownRunner(BaseRunner):
    def teardown(self) -> None:
        raise RuntimeError("local teardown failed")

    def run(self) -> None:
        return None


def test_base_runner_setup_failure_cleanup_always_cleans_distributed_group(
    tmp_path,
) -> None:
    runner = _FailingLocalTeardownRunner(
        OmegaConf.create({"training": {"out_dir": str(tmp_path)}})
    )
    distributed = _CleanupTrackingDistributed()
    runner.distributed = distributed

    with pytest.raises(RuntimeError, match="local teardown failed"):
        runner.teardown_after_setup_failure()

    assert distributed.cleanup_calls == 1


def test_classifier_setup_failure_cleanup_skips_distributed_barrier() -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner._console_state = None
    runner._metric_logger = None
    distributed = _CleanupTrackingDistributed()
    runner.distributed = distributed

    runner.teardown_after_setup_failure()

    assert distributed.barrier_calls == 0
    assert distributed.cleanup_calls == 1
