from __future__ import annotations

from pathlib import Path

import pytest

import dreamervla.launchers.frozen_model_cotrain_ray as launcher
from dreamervla.launchers.frozen_model_cotrain_ray import build_launch


def test_frozen_ray_launcher_builds_one_command_for_eight_gpus(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    run_root = tmp_path / "run"
    wm.touch()
    classifier.touch()

    launch = build_launch(
        [
            str(wm),
            str(classifier),
            "--run-root",
            str(run_root),
            "manual_cotrain.global_steps=12",
        ]
    )

    assert launch.visible_gpus == tuple(str(gpu) for gpu in range(8))
    assert launch.env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "experiment=dreamervla_frozen_models_rl_ray" in launch.command
    assert f"init.world_model_state_ckpt={wm.resolve()}" in launch.command
    assert f"init.classifier_state_ckpt={classifier.resolve()}" in launch.command
    assert f"training.out_dir={run_root.resolve()}" in launch.command
    assert "manual_cotrain.ngpu=8" in launch.command
    assert "cluster.num_gpus=8" in launch.command
    assert launch.command[-1] == "manual_cotrain.global_steps=12"
    assert launch.resume is False


def test_frozen_ray_launcher_resume_is_one_command_with_policy_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,4,5,6,7,8,9")
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    run_root = tmp_path / "run"
    resume = run_root / "checkpoints" / "manual_cotrain_step_500" / "manual_cotrain.ckpt"
    wm.touch()
    classifier.touch()
    resume.parent.mkdir(parents=True)
    resume.touch()

    launch = build_launch(
        [
            str(wm),
            str(classifier),
            "--resume-ckpt",
            str(resume),
        ]
    )

    assert launch.resume is True
    assert launch.out_dir == run_root.resolve()
    assert f"manual_cotrain.resume_ckpt={resume.resolve()}" in launch.command
    assert "training.resume=true" in launch.command


def test_frozen_ray_launcher_resume_infers_checkpoint_run_even_if_run_root_env_is_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("RUN_ROOT", str(tmp_path / "stale-stage-root"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    run_root = tmp_path / "frozen-run"
    resume = run_root / "checkpoints" / "manual_cotrain_step_500" / "manual_cotrain.ckpt"
    wm.touch()
    classifier.touch()
    resume.parent.mkdir(parents=True)
    resume.touch()

    launch = build_launch(
        [str(wm), str(classifier), "--resume-ckpt", str(resume)]
    )

    assert launch.out_dir == run_root.resolve()


def test_frozen_ray_launcher_rejects_non_eight_gpu_visibility(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    classifier.touch()

    with pytest.raises(ValueError, match="exactly 8"):
        build_launch([str(wm), str(classifier)])


def test_frozen_ray_launcher_rejects_duplicate_visible_gpu_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,6")
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    classifier.touch()

    with pytest.raises(ValueError, match="distinct"):
        build_launch([str(wm), str(classifier)])


def test_frozen_ray_launcher_resolves_completed_stage_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm_run = tmp_path / "wm-run"
    classifier_run = tmp_path / "classifier-run"
    wm_run.mkdir()
    classifier_run.mkdir()
    selected_wm = tmp_path / "selected-wm.ckpt"
    selected_classifier = tmp_path / "selected-classifier.ckpt"
    selected_wm.touch()
    selected_classifier.touch()
    monkeypatch.setattr(
        launcher,
        "select_world_model_checkpoint",
        lambda path: selected_wm,
    )
    monkeypatch.setattr(
        launcher,
        "select_classifier_checkpoint",
        lambda path: selected_classifier,
    )

    launch = build_launch(
        [str(wm_run), str(classifier_run), "--run-root", str(tmp_path / "out")]
    )

    assert f"init.world_model_state_ckpt={selected_wm}" in launch.command
    assert f"init.classifier_state_ckpt={selected_classifier}" in launch.command
