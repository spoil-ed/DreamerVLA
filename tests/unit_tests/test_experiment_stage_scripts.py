from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import h5py
from omegaconf import OmegaConf

from dreamervla.diagnostics import experiment_stage_checks


def test_cotrain_experiment_directory_contains_train_and_eval() -> None:
    root = Path(__file__).resolve().parents[2]
    experiments = root / "scripts" / "experiments"
    cotrain = experiments / "cotrain"

    assert sorted(path.name for path in experiments.iterdir()) == [
        "classifier_training",
        "cotrain",
        "single_trajectory_overfit",
        "world_model_training",
    ]
    assert sorted(path.name for path in cotrain.iterdir()) == ["eval.sh", "train.sh"]
    for folder in (
        "classifier_training",
        "single_trajectory_overfit",
        "world_model_training",
    ):
        assert (experiments / folder / "train.sh").is_file()
        assert (experiments / folder / "eval.sh").is_file()


def test_cotrain_train_script_uses_train_only_recipe_without_pinned_warm_states() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/experiments/cotrain/train.sh").read_text(
        encoding="utf-8"
    )

    assert "dreamervla.launchers.cotrain" in text
    assert "experiment=dreamervla_wmcls_cotrain_ray" not in text
    assert "manual_cotrain.global_steps" not in text
    assert "/inspire/" not in text
    assert "20260712" not in text


def test_experiment_scripts_defer_defaults_and_messages_to_python() -> None:
    root = Path(__file__).resolve().parents[2]
    scripts = sorted((root / "scripts/experiments").rglob("*.sh"))

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "${" not in text or "${BASH_SOURCE[0]}" in text
        assert ":-" not in text
        assert "echo " not in text


def test_world_model_training_entrypoint_defers_model_choice_to_hydra() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (
        root / "scripts" / "experiments" / "world_model_training" / "train.sh"
    ).read_text(encoding="utf-8")

    assert "--config-name world_model_training" in script
    assert "wm_dino_token_official" not in script
    assert "wm_official_upper_bound" not in script
    script_config_dir = root / "configs" / "scripts" / "world_model_training"
    assert not any(script_config_dir.glob("*.yaml"))
    assert (root / "configs" / "experiment" / "dino-wm.yaml").is_file()
    assert (root / "configs" / "experiment" / "dreamer-wm.yaml").is_file()
    assert (root / "configs" / "worldmodel" / "dino-wm.yaml").is_file()
    assert (root / "configs" / "worldmodel" / "dreamer-wm.yaml").is_file()


def test_world_model_training_config_switch_selects_expected_recipe() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "experiments" / "world_model_training" / "train.sh"

    for config_name, experiment in (
        ("dino-wm", "dino-wm"),
        ("dreamer-wm", "dreamer-wm"),
    ):
        result = subprocess.run(
            [
                "bash",
                str(script),
                "--config",
                config_name,
                "dry_run=true",
                "ngpu=1",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert f"experiment={experiment}" in result.stdout


def test_world_model_training_launcher_maps_batch_size_to_dino_dataloader() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "python",
            "-m",
            "dreamervla.launchers.train",
            "--config-name",
            "world_model_training",
            "--config",
            "dino-wm",
            "dry_run=true",
            "ngpu=1",
            "batch_size=7",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "dataloader.batch_size=7" in result.stdout
    assert "training.global_batch_size=7" not in result.stdout


def test_cotrain_eval_protocol_lives_in_hydra_config() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts/experiments/cotrain/eval.sh").read_text(
        encoding="utf-8"
    )
    cfg = OmegaConf.load(root / "configs/experiment/eval_cotrain.yaml")

    assert "--config-name cotrain_eval" in script
    assert "eval.num_episodes_per_task" not in script
    assert cfg.eval.ckpt_path is None
    assert cfg.eval.ckpt_kind == "vla_policy"
    assert cfg.eval.num_episodes_per_task == 10
    assert cfg.eval.num_envs == 25
    assert cfg.eval.require_strict_component_load is True


def test_cotrain_eval_script_rejects_missing_checkpoint() -> None:
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("COTRAIN_CKPT", None)
    result = subprocess.run(
        ["bash", str(root / "scripts/experiments/cotrain/eval.sh")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "missing required Hydra override" in result.stderr
    assert "eval.ckpt_path=<value>" in result.stderr


def test_hidden_token_preprocess_uses_configured_torchrun_world_size() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "preprocess" / "10_oft_hidden_token.sh"
    text = script.read_text(encoding="utf-8")
    assert '--nproc-per-node="${OFT_HIDDEN_TOKEN_GPUS}"' in text
    assert "dreamervla.preprocess.preprocess_oft_hidden_token" in text
    assert "obs_hidden_source=hidden_token" in text


def test_cotrain_world_model_ddp_keeps_online_defaults_configurable() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runtime.world_model_training_common import (
        _world_model_ddp_wrap_kwargs,
    )

    # Mixed-mode online cotrain retains the historical safety settings. The
    # static-graph optimization is selected only by the offline WM recipe.
    assert _world_model_ddp_wrap_kwargs(OmegaConf.create({"training": {}})) == {
        "find_unused_parameters": True,
        "broadcast_buffers": True,
    }


def test_experiment_stage_check_module_exposes_required_commands() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "dreamervla" / "diagnostics" / "experiment_stage_checks.py").read_text(
        encoding="utf-8"
    )

    for command in (
        "collect-check",
        "collect-run",
        "collect-output",
        "cls-check",
        "cls-eval",
        "wm-check",
        "pack-init",
        "cotrain-check",
        "libero-original-check",
        "libero-original-cls-run",
        "libero-original-warmup-run",
        "libero-original-rl-run",
    ):
        assert command in source


def test_classifier_check_treats_missing_failure_dirs_as_optional(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    success_raw = tmp_path / "success_raw"
    success_hidden = tmp_path / "success_hidden"
    success_raw.mkdir()
    success_hidden.mkdir()
    with h5py.File(success_raw / "demo.hdf5", "w") as handle:
        handle.attrs["complete"] = True
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("actions", shape=(1, 7), dtype="float32")
        demo.create_dataset("rewards", shape=(1,), dtype="float32")
        demo.create_dataset("dones", shape=(1,), dtype="uint8")
        demo.create_dataset("robot_states", shape=(1, 9), dtype="float32")
        demo.create_dataset("states", shape=(1, 5), dtype="float32")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", shape=(1, 1, 1, 3), dtype="uint8")
        obs.create_dataset("eye_in_hand_rgb", shape=(1, 1, 1, 3), dtype="uint8")
        obs.create_dataset("ee_pos", shape=(1, 3), dtype="float32")
        obs.create_dataset("ee_ori", shape=(1, 3), dtype="float32")
        obs.create_dataset("ee_states", shape=(1, 6), dtype="float32")
        obs.create_dataset("gripper_states", shape=(1, 2), dtype="float32")
        obs.create_dataset("joint_states", shape=(1, 7), dtype="float32")
    with h5py.File(success_hidden / "demo.hdf5", "w") as handle:
        handle.attrs["complete"] = True
        handle.create_dataset(
            "data/demo_0/obs_embedding",
            shape=(1, 256, 4096),
            dtype="float16",
            fillvalue=0,
        )
    (success_hidden / "preprocess_config.json").write_text(
        json.dumps(
            {
                "action_head_type": "oft_discrete_token",
                "obs_hidden_source": "hidden_token",
                "hidden_key": "obs_embedding",
                "token_count": 256,
                "token_dim": 4096,
                "hidden_dim": 1_048_576,
                "obs_embedding_shape": [256, 4096],
                "hidden_storage_format": "tokenized",
                "num_images_in_input": 1,
                "patches_per_image": 256,
                "history": 1,
                "include_state": False,
                "sidecar_schema_version": 1,
                "required_demo_datasets": ["obs_embedding"],
            }
        ),
        encoding="utf-8",
    )
    failure_raw = tmp_path / "missing_failures"
    failure_hidden = tmp_path / "missing_failure_hidden"
    cfg = OmegaConf.create(
        {
            "data": {
                "success_dir_raw": str(success_raw),
                "success_dir_hidden": str(success_hidden),
                "failure_dir_raw": str(failure_raw),
                "failure_dir_hidden": str(failure_hidden),
                "window": 8,
                "sampling_protocol": "wmpo",
            },
            "training": {"out_dir": str(tmp_path / "out")},
            "classifier": {"_target_": "classifier.Target"},
        }
    )
    monkeypatch.setattr(
        experiment_stage_checks,
        "_compose_train_config",
        lambda _experiment, _overrides: cfg,
    )

    exit_code = experiment_stage_checks.cls_check(
        argparse.Namespace(experiment="classifier_exp", overrides=[])
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["hdf5_counts"]["success_dir_raw"] == 1
    assert payload["hdf5_counts"]["success_dir_hidden"] == 1
    assert payload["hdf5_counts"]["failure_dir_raw"] == 0
    assert payload["hdf5_counts"]["failure_dir_hidden"] == 0
    assert payload["optional_directories"]["failure_dir_raw"] == "missing"
    assert payload["optional_directories"]["failure_dir_hidden"] == "missing"
