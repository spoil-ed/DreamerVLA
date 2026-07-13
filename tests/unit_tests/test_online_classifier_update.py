from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import torch

from dreamervla.runtime.classifier_update import online_classifier_update_step


def test_classifier_update_has_role_based_module() -> None:
    assert importlib.util.find_spec("dreamervla.runtime.classifier_update") is not None


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(window=2, chunk_size=1, chunk_pool="last")
        self.head = torch.nn.Linear(2, 2)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.head(windows.reshape(windows.shape[0], -1))


class _TinyBCEClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(window=2, chunk_size=1, chunk_pool="last")
        self.head = torch.nn.Linear(2, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.head(windows.reshape(windows.shape[0], -1))


class _Replay:
    def __init__(self, labels: list[int]) -> None:
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.last_kwargs = {}

    def sample_classifier_windows(self, batch_size: int, **kwargs):
        self.last_kwargs = dict(kwargs)
        batch_size = int(batch_size)
        labels = self.labels[:batch_size]
        return {
            "windows": torch.arange(batch_size * 2, dtype=torch.float32).reshape(
                batch_size, 2, 1
            ),
            "labels": labels,
        }


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params) -> None:
        super().__init__(params, lr=1.0e-2)
        self.step_calls = 0

    def step(self, closure=None):  # type: ignore[override]
        self.step_calls += 1
        return super().step(closure=closure)


def test_online_classifier_update_trains_single_class_batch_for_ddp_lockstep() -> None:
    classifier = _TinyClassifier()
    optimizer = _CountingSGD(classifier.parameters())

    metrics = online_classifier_update_step(
        classifier=classifier,
        optimizer=optimizer,
        replay=_Replay([0, 0]),
        device=torch.device("cpu"),
        batch_size=2,
        early_neg_stride=8,
        grad_clip=1.0,
    )

    assert optimizer.step_calls == 1
    assert metrics["skipped_single_class_batch"] == 0.0
    assert metrics["updated"] == 1.0
    assert metrics["pos_frac"] == 0.0


def test_online_classifier_update_keeps_mixed_batch_training() -> None:
    classifier = _TinyClassifier()
    optimizer = _CountingSGD(classifier.parameters())

    metrics = online_classifier_update_step(
        classifier=classifier,
        optimizer=optimizer,
        replay=_Replay([0, 1]),
        device=torch.device("cpu"),
        batch_size=2,
        early_neg_stride=8,
        grad_clip=1.0,
    )

    assert optimizer.step_calls == 1
    assert metrics["skipped_single_class_batch"] == 0.0
    assert metrics["pos_frac"] == 0.5


def test_online_classifier_update_supports_wmpo_bce_single_logit() -> None:
    classifier = _TinyBCEClassifier()
    optimizer = _CountingSGD(classifier.parameters())
    replay = _Replay([0, 1])

    metrics = online_classifier_update_step(
        classifier=classifier,
        optimizer=optimizer,
        replay=replay,
        device=torch.device("cpu"),
        batch_size=2,
        early_neg_stride=8,
        grad_clip=1.0,
        loss_type="bce",
        sampling_protocol="wmpo",
        balance_batches=True,
    )

    assert optimizer.step_calls == 1
    assert replay.last_kwargs["sampling_protocol"] == "wmpo"
    assert replay.last_kwargs["balance_batches"] is True
    assert metrics["updated"] == 1.0
    assert metrics["pos_frac"] == 0.5
    assert 0.0 <= metrics["prob_mean"] <= 1.0
