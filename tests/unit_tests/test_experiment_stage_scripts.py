from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
from omegaconf import OmegaConf

from dreamervla.diagnostics import experiment_stage_checks


def test_experiment_directory_contains_only_three_experiment_folders() -> None:
    root = Path(__file__).resolve().parents[2]
    experiments_dir = root / "scripts" / "experiments"

    assert sorted(path.name for path in experiments_dir.iterdir() if path.is_file()) == []
    assert sorted(path.name for path in experiments_dir.iterdir() if path.is_dir()) == [
        "classifier_training",
        "single_trajectory_overfit",
        "world_model_training",
    ]
    for folder in (
        "classifier_training",
        "single_trajectory_overfit",
        "world_model_training",
    ):
        assert (experiments_dir / folder / "train.sh").is_file()
        assert (experiments_dir / folder / "eval.sh").is_file()


def test_experiment_train_scripts_are_hydra_centered_and_include_checks() -> None:
    root = Path(__file__).resolve().parents[2]
    experiments_dir = root / "scripts" / "experiments"
    expected = {
        "single_trajectory_overfit/train.sh": (
            "dreamervla.diagnostics.wm_single_trajectory_overfit",
            "SINGLE_TRAJECTORY_TASK",
            "--run",
        ),
        "classifier_training/train.sh": (
            "experiment_stage_checks cls-check",
            "dreamervla.train",
            "experiment=${CLASSIFIER_EXPERIMENT}",
            "torch.distributed.run",
            '--nproc-per-node="${GPU_COUNT}"',
            "++training.distributed_strategy=ddp",
        ),
        "world_model_training/train.sh": (
            "dreamervla.train",
            "experiment=${WORLD_MODEL_EXPERIMENT}",
            "torch.distributed.run",
            '--nproc-per-node="${GPU_COUNT}"',
        ),
    }

    for name, markers in expected.items():
        script = experiments_dir / name
        text = script.read_text(encoding="utf-8")
        assert "DVLA_DATA_ROOT" in text, name
        assert "PYTHON_EXECUTABLE" in text, name
        for marker in markers:
            assert marker in text, name
        assert "cls_" not in script.name
        assert "wm_" not in script.name

    full_world_model = (experiments_dir / "world_model_training" / "train.sh").read_text(
        encoding="utf-8"
    )
    for config_owned_name in (
        "WORLD_MODEL_BATCH_SIZE",
        "WORLD_MODEL_LR",
        "WARMUP_REPLAY_EPOCHS",
        "WORLD_MODEL_SEQUENCE_LENGTH",
        "WORLD_MODEL_CHUNK_ROLLOUT_CHUNKS",
    ):
        assert config_owned_name not in full_world_model


def test_component_training_scripts_default_to_official_hydra_experiments() -> None:
    root = Path(__file__).resolve().parents[2]
    experiments_dir = root / "scripts" / "experiments"
    classifier = (experiments_dir / "classifier_training" / "train.sh").read_text(encoding="utf-8")
    world_model = (experiments_dir / "world_model_training" / "train.sh").read_text(
        encoding="utf-8"
    )

    assert (
        'CLASSIFIER_EXPERIMENT="${CLASSIFIER_EXPERIMENT:-classifier_official_upper_bound}"'
        in classifier
    )
    assert (
        'WORLD_MODEL_EXPERIMENT="${WORLD_MODEL_EXPERIMENT:-wm_official_upper_bound}"' in world_model
    )
    assert 'training.out_dir="${CLASSIFIER_RUN_ROOT}"' in classifier
    assert 'training.out_dir="${WORLD_MODEL_RUN_ROOT}"' in world_model
    assert "training.batch_size=" not in classifier
    assert "training.lr=" not in classifier
    assert "dataloader.batch_size=" not in world_model
    assert "optim.world_model.lr=" not in world_model


def test_world_model_profile_script_is_a_thin_one_click_hydra_launcher() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "experiments" / "world_model_training" / "profile.sh"

    assert script.is_file()
    assert os.access(script, os.X_OK)
    text = script.read_text(encoding="utf-8")
    for marker in (
        'WORLD_MODEL_EXPERIMENT="${WORLD_MODEL_EXPERIMENT:-wm_official_upper_bound_profile}"',
        'WORLD_MODEL_CHECKPOINT_EVERY="${WORLD_MODEL_CHECKPOINT_EVERY:-0}"',
        "world_model_profile",
        'exec bash "${SCRIPT_DIR}/train.sh" "$@"',
    ):
        assert marker in text
    for config_owned_override in (
        "training.wm_warmup_steps=",
        "training.wm_profile_steps=",
        "training.wm_prefetch_workers=",
        "training.warmup_replay_epochs=",
        "dataloader.batch_size=",
        "optim.world_model.lr=",
    ):
        assert config_owned_override not in text


def test_frozen_cotrain_script_only_requires_component_paths_at_handoff() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "e2e_frozen_model_cotrain.sh"

    assert script.is_file()
    assert os.access(script, os.X_OK)
    text = script.read_text(encoding="utf-8")
    for marker in (
        "dreamervla.launchers.frozen_model_cotrain_ray",
        'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"',
        "WORLD_MODEL_CKPT=/path",
        "CLASSIFIER_CKPT=/path",
        '"$@"',
    ):
        assert marker in text
    assert "positional" not in text
    assert "dreamervla.train" not in text


def test_eight_card_training_scripts_embed_h100_runtime_defaults() -> None:
    root = Path(__file__).resolve().parents[2]
    scripts = [
        root / "scripts" / "experiments" / "classifier_training" / "train.sh",
        root / "scripts" / "experiments" / "world_model_training" / "train.sh",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        for expected in (
            'export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"',
            'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"',
            'export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"',
            'export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"',
            'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"',
            'export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"',
            'export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"',
            'export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"',
            'GPU_COUNT="${GPU_COUNT:-${NGPU:-8}}"',
        ):
            assert expected in text, script


def test_classifier_and_world_model_train_scripts_expose_resume_and_periodic_ckpts() -> None:
    root = Path(__file__).resolve().parents[2]
    experiments_dir = root / "scripts" / "experiments"

    classifier_text = (experiments_dir / "classifier_training" / "train.sh").read_text(
        encoding="utf-8"
    )
    assert 'CLASSIFIER_RESUME="${CLASSIFIER_RESUME:-${RESUME:-false}}"' in classifier_text
    assert 'export CLASSIFIER_RUN_ROOT="${CLASSIFIER_RUN_ROOT:-' in classifier_text
    assert 'CLASSIFIER_CHECKPOINT_EVERY="${CLASSIFIER_CHECKPOINT_EVERY:-250}"' in classifier_text
    assert 'training.out_dir="${CLASSIFIER_RUN_ROOT}"' in classifier_text
    assert 'training.resume="${CLASSIFIER_RESUME}"' in classifier_text
    assert '++training.resume_dir="${CLASSIFIER_RESUME_DIR}"' in classifier_text
    assert 'training.ckpt_every="${CLASSIFIER_CHECKPOINT_EVERY}"' in classifier_text
    assert "checkpoints/latest.ckpt" in classifier_text
    assert "ckpt/latest.ckpt" in classifier_text

    world_model_text = (experiments_dir / "world_model_training" / "train.sh").read_text(
        encoding="utf-8"
    )
    assert 'WORLD_MODEL_RESUME="${WORLD_MODEL_RESUME:-${RESUME:-false}}"' in world_model_text
    assert 'WORLD_MODEL_CHECKPOINT_EVERY="${WORLD_MODEL_CHECKPOINT_EVERY:-500}"' in world_model_text
    assert 'WORLD_MODEL_TOPK_K="${WORLD_MODEL_TOPK_K:-3}"' in world_model_text
    assert 'training.resume="${WORLD_MODEL_RESUME}"' in world_model_text
    assert 'training.warmup_checkpoint_every="${WORLD_MODEL_CHECKPOINT_EVERY}"' in world_model_text
    assert 'training.warmup_topk_k="${WORLD_MODEL_TOPK_K}"' in world_model_text
    assert "wm_warmup.ckpt" in world_model_text
    assert "wm_step_*.ckpt" in world_model_text


def test_hidden_token_preprocess_uses_configured_torchrun_world_size() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "preprocess" / "35_oft_hidden_token.sh"
    text = script.read_text(encoding="utf-8")
    assert '--nproc-per-node="${OFT_HIDDEN_TOKEN_GPUS}"' in text
    assert "dreamervla.preprocess.preprocess_oft_hidden_token" in text
    assert "obs_hidden_source=hidden_token" in text


def test_cotrain_world_model_ddp_tracks_unused_parameters() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "dreamervla" / "runners" / "online_cotrain_runner.py").read_text(
        encoding="utf-8"
    )
    assert "self.world_model," in source
    assert "find_unused_parameters=True" in source


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
