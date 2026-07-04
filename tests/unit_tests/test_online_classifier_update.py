from __future__ import annotations

from types import SimpleNamespace

import torch

from dreamervla.runners.online_dreamervla import online_classifier_update_step


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(window=2, chunk_size=1, chunk_pool="last")
        self.head = torch.nn.Linear(2, 2)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.head(windows.reshape(windows.shape[0], -1))


class _Replay:
    def __init__(self, labels: list[int]) -> None:
        self.labels = torch.tensor(labels, dtype=torch.long)

    def sample_classifier_windows(self, batch_size: int, **kwargs):
        del kwargs
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
