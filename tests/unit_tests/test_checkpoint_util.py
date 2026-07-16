from __future__ import annotations

from pathlib import Path

import pytest

from dreamervla.utils.checkpoint_util import (
    TopKCheckpointManager,
    format_metric_checkpoint_name,
)


@pytest.mark.parametrize(
    "metric_name",
    ["loss", "eval/accuracy", "../loss", "a\\b"],
)
def test_metric_checkpoint_name_never_escapes_directory(metric_name: str) -> None:
    name = format_metric_checkpoint_name(
        epoch=3,
        metric_name=metric_name,
        metric_value=0.25,
    )

    assert Path(name).name == name
    assert name.startswith("epoch=0003-")
    assert name.endswith("=0.250000.ckpt")


def test_metric_checkpoint_name_expands_beyond_four_epoch_digits() -> None:
    assert (
        format_metric_checkpoint_name(
            epoch=12345,
            metric_name="accuracy",
            metric_value=1.0,
        )
        == "epoch=12345-accuracy=1.000000.ckpt"
    )


@pytest.mark.parametrize("epoch", [-1, -100])
def test_metric_checkpoint_name_rejects_negative_epoch(epoch: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        format_metric_checkpoint_name(
            epoch=epoch,
            metric_name="loss",
            metric_value=0.25,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_metric_checkpoint_name_rejects_nonfinite_value(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        format_metric_checkpoint_name(
            epoch=1,
            metric_name="loss",
            metric_value=value,
        )


def test_topk_manager_uses_flat_metric_names_and_preserves_latest(tmp_path: Path) -> None:
    latest = tmp_path / "latest.ckpt"
    latest.write_bytes(b"latest")
    manager = TopKCheckpointManager(
        save_dir=tmp_path,
        monitor_key="eval/loss",
        metric_name="loss",
        mode="min",
        k=2,
    )

    first = manager.get_ckpt_path({"epoch": 1, "eval/loss": 0.5})
    assert first == str(tmp_path / "epoch=0001-loss=0.500000.ckpt")
    Path(first).write_bytes(b"first")
    second = manager.get_ckpt_path({"epoch": 2, "eval/loss": 0.4})
    assert second == str(tmp_path / "epoch=0002-loss=0.400000.ckpt")
    Path(second).write_bytes(b"second")

    assert manager.get_ckpt_path({"epoch": 3, "eval/loss": 0.6}) is None
    third = manager.get_ckpt_path({"epoch": 4, "eval/loss": 0.3})

    assert third == str(tmp_path / "epoch=0004-loss=0.300000.ckpt")
    assert not Path(first).exists()
    assert Path(second).is_file()
    assert latest.read_bytes() == b"latest"


def test_topk_manager_restores_and_prunes_matching_flat_checkpoints(tmp_path: Path) -> None:
    latest = tmp_path / "latest.ckpt"
    latest.write_bytes(b"latest")
    best = tmp_path / "epoch=0001-loss=0.200000.ckpt"
    middle = tmp_path / "epoch=0002-loss=0.300000.ckpt"
    worst = tmp_path / "epoch=0003-loss=0.400000.ckpt"
    unrelated = tmp_path / "epoch=0004-accuracy=0.900000.ckpt"
    for path in (best, middle, worst, unrelated):
        path.write_bytes(path.name.encode())

    manager = TopKCheckpointManager(
        save_dir=tmp_path,
        monitor_key="eval/loss",
        metric_name="loss",
        mode="min",
        k=2,
    )

    assert manager.path_value_map == {str(best): 0.2, str(middle): 0.3}
    assert not worst.exists()
    assert unrelated.is_file()
    assert latest.read_bytes() == b"latest"
