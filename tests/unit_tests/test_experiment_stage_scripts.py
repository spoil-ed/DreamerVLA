from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_cotrain_experiment_directory_contains_train_and_eval() -> None:
    root = Path(__file__).resolve().parents[2]
    experiments = root / "scripts" / "experiments"
    cotrain = experiments / "cotrain"

    assert sorted(path.name for path in experiments.iterdir()) == [
        "classifier_training",
        "collect_rollouts",
        "cotrain",
        "openvla_oft_official_eval",
        "single_trajectory_overfit",
        "world_model_training",
    ]
    assert sorted(path.name for path in cotrain.iterdir()) == ["eval.sh", "train.sh"]
    assert sorted(path.name for path in (experiments / "classifier_training").iterdir()) == [
        "train.sh"
    ]
    for folder in ("single_trajectory_overfit", "world_model_training"):
        assert (experiments / folder / "train.sh").is_file()
        assert (experiments / folder / "eval.sh").is_file()


def test_stale_classifier_summarizer_is_removed() -> None:
    root = Path(__file__).resolve().parents[2]

    assert not (root / "dreamervla/diagnostics/experiment_stage_checks.py").exists()
    assert not (root / "scripts/experiments/classifier_training/eval.sh").exists()


def test_cotrain_train_script_uses_train_only_recipe_without_pinned_warm_states() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/experiments/cotrain/train.sh").read_text(encoding="utf-8")

    assert "dreamervla.launchers.train" in text
    assert "dreamervla.launchers.cotrain" not in text
    assert "experiment=openvla_libero" not in text
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
    script = (root / "scripts" / "experiments" / "world_model_training" / "train.sh").read_text(
        encoding="utf-8"
    )

    assert "--config dreamer-wm" in script
    assert "wm_dino_token_official" not in script
    assert "wm_official_upper_bound" not in script
    assert {path.name for path in (root / "configs" / "scripts").iterdir()} == {
        "download",
        "install",
        "preprocess",
        "reproduce",
    }
    assert (root / "configs" / "experiment" / "dino-wm.yaml").is_file()
    assert (root / "configs" / "experiment" / "dreamer-wm.yaml").is_file()
    assert (root / "configs" / "worldmodel" / "dino-wm.yaml").is_file()
    assert (root / "configs" / "worldmodel" / "dreamer-wm.yaml").is_file()


def test_mainline_experiments_compose_only_the_components_their_stage_needs() -> None:
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        classifier = compose(
            config_name="train",
            overrides=["experiment=classifier_official_upper_bound"],
        )
        world_model = compose(
            config_name="train",
            overrides=["experiment=dreamer-wm"],
        )
        cotrain = compose(
            config_name="train",
            overrides=["experiment=openvla_libero"],
        )

    OmegaConf.resolve(classifier)
    OmegaConf.resolve(world_model)
    OmegaConf.resolve(cotrain)

    assert classifier._target_ == "dreamervla.runners.SuccessClassifierTrainingRunner"
    assert classifier.classifier._target_
    assert classifier.task.classifier.dataset.train._target_
    assert world_model._target_ == "dreamervla.runners.WorldModelTrainingRunner"
    assert world_model.world_model._target_
    assert cotrain._target_ == "dreamervla.runners.DreamerRunner"
    assert cotrain.world_model._target_
    assert cotrain.classifier._target_
    assert cotrain.manual_cotrain.real_env_enabled is True
    assert cotrain.task.openvla_oft.hidden_token.token_dim == 4096


def test_active_experiments_use_expected_run_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("RUN_ROOT", str(tmp_path))
    expected = {
        "collect_rollouts": "collect_rollouts",
        "dino-wm": "dino-wm",
        "dreamer-wm": "dreamer-wm",
        "classifier_official_upper_bound": "classifier_official_upper_bound",
        "openvla_libero": "openvla_libero",
    }
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        configs = {
            experiment: compose(
                config_name="train",
                overrides=[f"experiment={experiment}"],
            )
            for experiment in expected
        }
        eval_cfg = compose(
            config_name="train",
            overrides=["experiment=eval_cotrain"],
        )

    for experiment, run_name in expected.items():
        cfg = configs[experiment]
        OmegaConf.resolve(cfg)
        out_dir = Path(cfg.training.out_dir)
        assert cfg.run.name == run_name
        assert out_dir.parent == tmp_path / run_name
        assert re.fullmatch(r"\d{8}_\d{6}", out_dir.name)
        assert "pre_mainline" not in out_dir.parts

    OmegaConf.resolve(eval_cfg)
    assert eval_cfg.run.name == "eval_cotrain"
    assert Path(eval_cfg.training.out_dir) == tmp_path / "eval" / "libero_goal"


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


def test_world_model_training_launcher_rejects_batch_alias() -> None:
    from dreamervla.launchers.train import build_launch

    with pytest.raises(SystemExit, match="Hydra key=value"):
        build_launch(["--config", "dino-wm", "batch_size=7"])


def test_experiment_launcher_resume_reuses_original_run_root(
    tmp_path: Path,
    capsys,
) -> None:
    from dreamervla.launchers.train import main

    run_dir = tmp_path / "dreamer-wm" / "20260714_120000"
    checkpoint = run_dir / "checkpoints" / "latest.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    result = main(
        [
            "--config",
            "dreamer-wm",
            "--resume",
            str(checkpoint),
            "dry_run=true",
            "ngpu=1",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "training.resume=true" in output
    assert f"training.resume_path={json.dumps(str(checkpoint.resolve()))}" in output
    assert f"training.resume_dir={json.dumps(str(run_dir.resolve()))}" in output
    assert f"training.out_dir={json.dumps(str(run_dir.resolve()))}" in output


def test_experiment_launcher_rejects_resume_with_output_override(
    tmp_path: Path,
) -> None:
    from dreamervla.launchers.train import main

    run_dir = tmp_path / "run"
    latest = run_dir / "checkpoints" / "latest.ckpt"
    latest.parent.mkdir(parents=True)
    latest.touch()

    with pytest.raises(ValueError, match="--resume.*out_dir"):
        main(
            [
                "--config",
                "dreamer-wm",
                "--resume",
                str(run_dir),
                "training.out_dir=/tmp/fork",
                "dry_run=true",
            ]
        )


def test_cotrain_eval_protocol_lives_in_hydra_config() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts/experiments/cotrain/eval.sh").read_text(encoding="utf-8")
    cfg = OmegaConf.load(root / "configs/experiment/eval_cotrain.yaml")

    assert "--config eval_cotrain" in script
    assert "eval.num_episodes_per_task" not in script
    assert cfg.eval.ckpt_path is None
    assert cfg.eval.ckpt_kind == "vla_policy"
    assert cfg.eval.num_episodes_per_task == 10
    assert cfg.eval.num_envs == 25
    assert cfg.eval.distributed is True
    assert cfg.eval.require_strict_component_load is True
    assert cfg.launch.distributed is True
    assert cfg.launch.ngpu == 8
    assert str(cfg.launch.gpus) == "0,1,2,3,4,5,6,7"


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


def test_offline_world_model_ddp_defaults_remain_configurable() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runtime.world_model_training_common import (
        _world_model_ddp_wrap_kwargs,
    )

    # Offline recipes retain conservative defaults and may opt into static graph.
    assert _world_model_ddp_wrap_kwargs(OmegaConf.create({"training": {}})) == {
        "find_unused_parameters": True,
        "broadcast_buffers": True,
    }
