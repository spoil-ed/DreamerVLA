from __future__ import annotations

import json
from pathlib import Path

import pytest

from dreamervla.launchers.manual_cotrain_async import build_launch


def _write_manual_ckpt(tmp_path: Path, *, step: int) -> Path:
    ckpt_dir = tmp_path / f"manual_cotrain_step_{step}"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "manual_cotrain.ckpt"
    ckpt.write_bytes(b"ckpt")
    manifest = {
        "global_step": step,
        "components": {"policy": {"path": "manual_cotrain.ckpt"}},
    }
    (ckpt_dir / "manual_cotrain_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return ckpt


def test_resume_launch_only_requires_gpus_resume_and_ckpt(tmp_path, monkeypatch):
    ckpt = _write_manual_ckpt(tmp_path, step=5)
    monkeypatch.setenv("DVLA_DATA_ROOT", str(tmp_path / "data"))

    launch = build_launch(
        [
            "resume=true",
            f"ckpt={ckpt}",
            "gpus=1,2,3,4,5",
        ]
    )

    assert launch.resume is True
    assert launch.visible_gpus == ("1", "2", "3", "4", "5")
    assert launch.env["CUDA_VISIBLE_DEVICES"] == "1,2,3,4,5"
    assert launch.resume_step == 5
    assert launch.target_step is not None and launch.target_step > 5
    assert "manual_cotrain.ngpu=5" in launch.command
    assert "+cluster.num_gpus=5" in launch.command
    assert "manual_cotrain.envs_per_worker=8" in launch.command
    assert "manual_cotrain.rollout_epoch=1" in launch.command
    assert "manual_cotrain.max_steps_per_rollout_epoch=64" in launch.command
    assert f"+manual_cotrain.resume_ckpt={ckpt}" in launch.command
    assert f"+actor.init_ckpt.path={ckpt}" in launch.command
    assert f"+learner.init_ckpt.path={ckpt}" in launch.command


def test_fresh_launch_runs_pipeline_with_internal_manual_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("DVLA_DATA_ROOT", str(tmp_path / "data"))

    launch = build_launch(["resume=false", "gpus=1,2,3,4,5"])

    assert launch.resume is False
    assert launch.visible_gpus == ("1", "2", "3", "4", "5")
    assert launch.env["CUDA_VISIBLE_DEVICES"] == "1,2,3,4,5"
    assert "dreamervla.launchers.coldstart_warmup_cotrain" in launch.command
    assert "mode=ray" in launch.command
    assert "task=goal" in launch.command
    assert "ngpu=5" in launch.command
    assert "cotrain_engine=async" in launch.command
    assert "render_backend=osmesa" in launch.command
    assert "manual_cotrain.envs_per_worker=8" in launch.command
    assert "manual_cotrain.max_steps_per_rollout_epoch=64" in launch.command


def test_resume_requires_checkpoint_path():
    with pytest.raises(ValueError, match="resume=true requires ckpt"):
        build_launch(["resume=true", "gpus=1,2,3,4,5"])
