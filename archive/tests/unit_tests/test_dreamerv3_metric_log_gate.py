"""PERF-Q7: offline DreamerV3 runners must materialize per-step metric scalars
to host ONLY on logging-boundary steps, not every step.

These tests drive the REAL ``run()`` loop on CPU. Each runner is built via
``__new__`` (bypassing the CUDA-defaulting ``__init__``); only the attributes the
loop reads are set, and the heavy collaborators (dataset/model build, loader,
resume, checkpoint, viz, console/metric logging) are stubbed so a handful of CPU
steps execute. The metric tensor in the model output is a ``Tensor`` subclass
whose ``.cpu()`` increments a counter, so we can assert the host sync fires only
on the steps where ``row`` is actually logged.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from dreamervla.runners.dreamerv3_pixel_runner import DreamerV3PixelRunner
from dreamervla.runners.dreamerv3_token_runner import DreamerV3TokenRunner

# log_every=2 over 6 steps (global_step 0..5) → log on steps 0,2,4 == 3 logs.
NUM_STEPS = 6
LOG_EVERY = 2
EXPECTED_LOG_STEPS = [s for s in range(NUM_STEPS) if s % LOG_EVERY == 0]


class _SyncCountingTensor(torch.Tensor):
    """A tensor that records every ``.cpu()`` (device->host sync) call.

    The metric values returned by the fake model are wrapped in this subclass so
    the test can count how many times the runner pulls them to host across the
    training loop. The shared counter lives on the class.
    """

    cpu_calls: int = 0

    @classmethod
    def make(cls, value: float) -> _SyncCountingTensor:
        return torch.tensor(value, dtype=torch.float32).as_subclass(cls)

    def cpu(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        type(self).cpu_calls += 1
        return torch.Tensor.cpu(self, *args, **kwargs)


class _FakeWorldModel(nn.Module):
    """Minimal model: one parameter so ``clip_grad_norm_`` sees a real grad.

    ``forward`` returns ``_loss`` (a differentiable scalar wired to the param so
    ``backward`` populates ``param.grad``) and a single metric whose value is a
    ``_SyncCountingTensor`` keyed by the current step (so logged values are
    distinguishable per step).
    """

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        step = int(batch["step"])
        loss = (self.w + 1.0) ** 2  # differentiable, grad == 2*(w+1) == 2.0
        metric = _SyncCountingTensor.make(float(step) + 0.5)
        return {"_loss": loss, "metric": metric}


def _make_cfg() -> Any:
    return OmegaConf.create(
        {
            "dataset": {},
            "world_model": {},
            "training": {
                "seed": 0,
                "num_epochs": 1,
                "log_every": LOG_EVERY,
                "save_every": 0,
                "save_final": False,
            },
            # lr=0 keeps the single param at 0 across steps, so the grad (and
            # thus grad_norm) is the same on every step -> logged value is a
            # clean per-step reference.
            "optim": {"lr": 0.0, "warmup": 0, "grad_clip": 100.0},
            "console": {"log_every": 1, "progress_every_s": 0.0},
        }
    )


def _build_runner(runner_cls, *, log_path, fake_loader):
    """Construct ``runner_cls`` via ``__new__`` and wire the loop's collaborators
    to CPU-only fakes so the genuine ``run()`` executes a few steps."""
    runner = runner_cls.__new__(runner_cls)
    runner.cfg = _make_cfg()
    runner.device = torch.device("cpu")
    runner.rank = 0
    runner.local_rank = 0
    runner.world_size = 1
    runner.use_ddp = False
    runner.global_step = 0
    runner.epoch = 0
    runner.out_dir = log_path.parent
    runner.log_path = log_path
    runner.ckpt_dir = log_path.parent / "ckpt"

    logged_rows: list[dict[str, Any]] = []

    model = _FakeWorldModel()

    # hydra.utils.instantiate is called for dataset then world_model. Hand back
    # a tiny dataset stub first, then the fake model.
    class _FakeDataset:
        data_spec = "fake"

        def __len__(self) -> int:
            return len(fake_loader)

    instantiate_returns = iter([_FakeDataset(), model])

    # Pixel runner's _make_loader returns (loader, sampler); token's returns the
    # bare loader.
    if runner_cls is DreamerV3PixelRunner:
        runner._make_loader = lambda *a, **k: (fake_loader, None)  # type: ignore[assignment]
    else:
        runner._make_loader = lambda *a, **k: fake_loader  # type: ignore[assignment]
    runner._setup_auxiliary_modules = lambda *a, **k: None  # type: ignore[attr-defined]
    runner._maybe_build_viz = lambda *a, **k: None  # type: ignore[attr-defined]
    runner._maybe_resume = lambda *a, **k: False  # type: ignore[assignment]
    runner._save_ckpt = lambda *a, **k: None  # type: ignore[assignment]
    runner._maybe_save_viz = lambda *a, **k: None  # type: ignore[assignment]
    runner._reduce_metrics = lambda metrics: metrics  # identity, single-process
    runner.console_banner = lambda *a, **k: None  # type: ignore[assignment]
    runner.console_progress = lambda *a, **k: None  # type: ignore[assignment]
    runner.console_metrics = lambda *a, **k: None  # type: ignore[assignment]
    runner.log_metrics = lambda row, **k: logged_rows.append(dict(row))  # type: ignore[assignment]
    runner._barrier = lambda *a, **k: None  # type: ignore[attr-defined]

    return runner, instantiate_returns, logged_rows


def _run_loop(runner_cls, monkeypatch, tmp_path):
    log_path = tmp_path / "logs.json.txt"
    # `step` keeps a fake batch identifiable; the loop runs len(loader) steps.
    fake_loader = [{"step": s} for s in range(NUM_STEPS)]
    runner, instantiate_returns, logged_rows = _build_runner(
        runner_cls, log_path=log_path, fake_loader=fake_loader
    )

    import dreamervla.runners.dreamerv3_pixel_runner as pixel_mod
    import dreamervla.runners.dreamerv3_token_runner as token_mod

    target = pixel_mod if runner_cls is DreamerV3PixelRunner else token_mod
    monkeypatch.setattr(
        target.hydra.utils, "instantiate", lambda *a, **k: next(instantiate_returns)
    )

    _SyncCountingTensor.cpu_calls = 0
    runner.run()
    return logged_rows


@pytest.mark.parametrize("runner_cls", [DreamerV3PixelRunner, DreamerV3TokenRunner])
def test_metric_host_sync_only_on_log_steps(runner_cls, monkeypatch, tmp_path):
    logged_rows = _run_loop(runner_cls, monkeypatch, tmp_path)

    # The metric host sync (`.cpu()`) must fire exactly once per LOGGED step, not
    # once per training step. On the un-gated (base) code it fires every step.
    assert _SyncCountingTensor.cpu_calls == len(EXPECTED_LOG_STEPS)

    # And the loop must actually have logged on the expected boundary steps.
    assert [row["global_step"] for row in logged_rows] == EXPECTED_LOG_STEPS


@pytest.mark.parametrize("runner_cls", [DreamerV3PixelRunner, DreamerV3TokenRunner])
def test_logged_metric_values_unchanged(runner_cls, monkeypatch, tmp_path):
    logged_rows = _run_loop(runner_cls, monkeypatch, tmp_path)

    # Logged values are byte-identical to the eager reference: the metric for a
    # step is `step + 0.5`, grad_norm is the L2 norm of the single grad (== 2.0).
    by_step = {row["global_step"]: row for row in logged_rows}
    for step in EXPECTED_LOG_STEPS:
        row = by_step[step]
        assert row["metric"] == pytest.approx(step + 0.5)
        assert row["grad_norm"] == pytest.approx(2.0)
