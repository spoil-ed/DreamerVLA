from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dreamervla.launchers.cotrain import build_launch


def test_cotrain_launcher_requires_pretrained_wm_and_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORLD_MODEL_CKPT", raising=False)
    monkeypatch.delenv("CLASSIFIER_CKPT", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(
        ValueError,
        match="WORLD_MODEL_CKPT and CLASSIFIER_CKPT",
    ):
        build_launch([])


def test_cotrain_launcher_accepts_atomic_warm_start_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    classifier.touch()
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))

    launch = build_launch([])

    assert (
        f"init.world_model_state_ckpt={json.dumps(str(wm.resolve()))}"
        in launch.command
    )
    assert (
        f"init.classifier_state_ckpt={json.dumps(str(classifier.resolve()))}"
        in launch.command
    )
    assert launch.cfg.manual_cotrain.global_steps == 20_000
    assert launch.cfg.manual_cotrain.eval_interval_global_steps == 10
    assert launch.cfg.manual_cotrain.eval_initial_global_step is True
    assert launch.env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert launch.env["LIBERO_CONFIG_PATH"] == str(
        (Path(launch.env["DVLA_DATA_ROOT"]) / ".libero").resolve()
    )


def test_cotrain_launcher_matches_classifier_head_to_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    torch.save(
        {
            "classifier": {
                "head.weight": torch.zeros(2, 1024),
                "head.bias": torch.zeros(2),
            }
        },
        classifier,
    )
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))

    launch = build_launch([])

    assert launch.cfg.ray_components.classifier.kwargs.output_dim == 2
    assert launch.cfg.learner.train_cfg.classifier_loss_type == "ce"
    assert "ray_components.classifier.kwargs.output_dim=2" in launch.command
    assert "learner.train_cfg.classifier_loss_type=ce" in launch.command


def test_cotrain_launcher_reads_global_steps_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    classifier.touch()
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("WMCLS_COTRAIN_GLOBAL_STEPS", "10")

    launch = build_launch([])

    assert "manual_cotrain.global_steps=10" in launch.command
    assert launch.cfg.manual_cotrain.global_steps == 10


def test_cotrain_launcher_accepts_huggingface_component_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wm = tmp_path / "wm_hf"
    classifier = tmp_path / "classifier_hf"
    wm.mkdir()
    classifier.mkdir()
    (wm / "config.json").write_text("{}", encoding="utf-8")
    (classifier / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))

    launch = build_launch([])

    assert f"init.world_model_state_ckpt={json.dumps(str(wm.resolve()))}" in launch.command
    assert (
        f"init.classifier_state_ckpt={json.dumps(str(classifier.resolve()))}"
        in launch.command
    )


def test_cotrain_launcher_rejects_partial_warm_start_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wm = tmp_path / "wm.ckpt"
    wm.touch()
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.delenv("CLASSIFIER_CKPT", raising=False)

    with pytest.raises(
        ValueError,
        match="both WORLD_MODEL_CKPT and CLASSIFIER_CKPT",
    ):
        build_launch([])


def test_cotrain_launcher_reads_gpu_count_from_hydra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    classifier.touch()
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")

    launch = build_launch(
        [
            "manual_cotrain.ngpu=2",
            "cluster.num_gpus=2",
        ]
    )

    assert launch.cfg.manual_cotrain.ngpu == 2
    assert launch.env["CUDA_VISIBLE_DEVICES"] == "2,5"
