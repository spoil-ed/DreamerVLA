from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra.core.override_parser.overrides_parser import OverridesParser

import dreamervla.launchers.frozen_model_cotrain_ray as launcher
from dreamervla.launchers.frozen_model_cotrain_ray import build_launch


def _save_classifier_checkpoint(path: Path, *, threshold: float = 0.45) -> None:
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "threshold": threshold,
            "config": {"classifier": {"hidden_dim": 1}},
        },
        path,
    )


def test_frozen_ray_launcher_builds_one_command_for_eight_gpus(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    run_root = tmp_path / "run"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(run_root))

    launch = build_launch(
        [
            "manual_cotrain.global_steps=12",
        ]
    )

    assert launch.visible_gpus == tuple(str(gpu) for gpu in range(8))
    assert launch.env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "experiment=dreamervla_frozen_models_rl_ray" in launch.command
    assert (
        f"init.world_model_state_ckpt={json.dumps(str(wm.resolve()))}"
        in launch.command
    )
    assert (
        f"init.classifier_state_ckpt={json.dumps(str(classifier.resolve()))}"
        in launch.command
    )
    assert f"training.out_dir={json.dumps(str(run_root.resolve()))}" in launch.command
    assert "manual_cotrain.ngpu=8" in launch.command
    assert "cluster.num_gpus=8" in launch.command
    assert "algorithm.lumos.classifier_threshold=0.45" in launch.command
    assert launch.command[-1] == "manual_cotrain.global_steps=12"
    assert launch.resume is False


def test_frozen_ray_launcher_quotes_hydra_checkpoint_paths_containing_equals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm_step=00004000-loss=0.097758.ckpt"
    classifier = tmp_path / "best_window_f10.9711_th0.45.ckpt"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(tmp_path / "run=quoted"))

    launch = build_launch([])

    parsed = OverridesParser.create().parse_overrides(overrides=launch.command[3:])
    values = {override.get_key_element(): override.value() for override in parsed}

    assert values["init.world_model_state_ckpt"] == str(wm.resolve())
    assert values["training.out_dir"] == str((tmp_path / "run=quoted").resolve())


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
    _save_classifier_checkpoint(classifier)
    resume.parent.mkdir(parents=True)
    resume.touch()
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RESUME_CKPT", str(resume))

    launch = build_launch([])

    assert launch.resume is True
    assert launch.out_dir == run_root.resolve()
    assert (
        f"manual_cotrain.resume_ckpt={json.dumps(str(resume.resolve()))}"
        in launch.command
    )
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
    _save_classifier_checkpoint(classifier)
    resume.parent.mkdir(parents=True)
    resume.touch()
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RESUME_CKPT", str(resume))

    launch = build_launch([])

    assert launch.out_dir == run_root.resolve()


def test_frozen_ray_launcher_rejects_non_eight_gpu_visibility(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))

    with pytest.raises(ValueError, match="exactly 8"):
        build_launch([])


def test_frozen_ray_launcher_rejects_duplicate_visible_gpu_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,6")
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))

    with pytest.raises(ValueError, match="distinct"):
        build_launch([])


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
        "select_available_world_model_checkpoint",
        lambda path: selected_wm,
    )
    monkeypatch.setattr(
        launcher,
        "select_available_classifier_checkpoint",
        lambda path: selected_classifier,
    )
    monkeypatch.setattr(
        launcher,
        "resolve_available_classifier_threshold",
        lambda path, default=0.5: 0.45,
    )
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm_run))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier_run))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(tmp_path / "out"))

    launch = build_launch([])

    assert (
        f"init.world_model_state_ckpt={json.dumps(str(selected_wm))}"
        in launch.command
    )
    assert (
        f"init.classifier_state_ckpt={json.dumps(str(selected_classifier))}"
        in launch.command
    )


def test_frozen_ray_launcher_loads_classifier_final_with_best_sibling_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm.ckpt"
    wm.touch()
    checkpoints = tmp_path / "classifier-run" / "checkpoints"
    checkpoints.mkdir(parents=True)
    final = checkpoints / "final.ckpt"
    best = checkpoints / "best_window_f10.9711_th0.45.ckpt"
    torch.save(
        {
            "cfg": {"classifier": {"hidden_dim": 1}},
            "state_dicts": {"model": {"weight": torch.ones(1)}},
            "pickles": {},
        },
        final,
    )
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "threshold": 0.45,
            "f1": 0.9711,
            "step": 500,
            "config": {"classifier": {"hidden_dim": 1}},
        },
        best,
    )
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(final))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(tmp_path / "out"))

    launch = build_launch([])

    assert (
        f"init.classifier_state_ckpt={json.dumps(str(final.resolve()))}"
        in launch.command
    )
    assert "algorithm.lumos.classifier_threshold=0.45" in launch.command


def test_frozen_ray_launcher_rejects_positional_component_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("WORLD_MODEL_CKPT", raising=False)
    monkeypatch.delenv("CLASSIFIER_CKPT", raising=False)

    with pytest.raises(ValueError, match="key=value"):
        build_launch([str(tmp_path / "wm.ckpt"), str(tmp_path / "classifier.ckpt")])


def test_frozen_ray_launcher_requires_explicit_checkpoint_assignments(
    monkeypatch,
) -> None:
    monkeypatch.delenv("WORLD_MODEL_CKPT", raising=False)
    monkeypatch.delenv("CLASSIFIER_CKPT", raising=False)

    with pytest.raises(ValueError, match="WORLD_MODEL_CKPT=/path"):
        build_launch([])
