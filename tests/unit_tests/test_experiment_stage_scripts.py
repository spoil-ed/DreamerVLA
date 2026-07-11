from __future__ import annotations

from pathlib import Path


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
            "experiment_stage_checks libero-original-check",
            "dreamervla.train",
            "experiment=wm_full_dataset_train",
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

    classifier_text = (
        experiments_dir / "classifier_training" / "train.sh"
    ).read_text(encoding="utf-8")
    assert 'CLASSIFIER_RESUME="${CLASSIFIER_RESUME:-${RESUME:-false}}"' in classifier_text
    assert 'export CLASSIFIER_RUN_ROOT="${CLASSIFIER_RUN_ROOT:-' in classifier_text
    assert 'CLASSIFIER_CHECKPOINT_EVERY="${CLASSIFIER_CHECKPOINT_EVERY:-250}"' in classifier_text
    assert 'training.out_dir="${CLASSIFIER_RUN_ROOT}"' in classifier_text
    assert 'training.resume="${CLASSIFIER_RESUME}"' in classifier_text
    assert '++training.resume_dir="${CLASSIFIER_RESUME_DIR}"' in classifier_text
    assert 'training.ckpt_every="${CLASSIFIER_CHECKPOINT_EVERY}"' in classifier_text
    assert "checkpoints/latest.ckpt" in classifier_text
    assert "ckpt/latest.ckpt" in classifier_text

    world_model_text = (
        experiments_dir / "world_model_training" / "train.sh"
    ).read_text(encoding="utf-8")
    assert 'WORLD_MODEL_RESUME="${WORLD_MODEL_RESUME:-${RESUME:-false}}"' in world_model_text
    assert 'WORLD_MODEL_CHECKPOINT_EVERY="${WORLD_MODEL_CHECKPOINT_EVERY:-500}"' in world_model_text
    assert 'WORLD_MODEL_TOPK_K="${WORLD_MODEL_TOPK_K:-3}"' in world_model_text
    assert 'training.resume="${WORLD_MODEL_RESUME}"' in world_model_text
    assert (
        'training.warmup_checkpoint_every="${WORLD_MODEL_CHECKPOINT_EVERY}"'
        in world_model_text
    )
    assert 'training.warmup_topk_k="${WORLD_MODEL_TOPK_K}"' in world_model_text
    assert "wm_warmup.ckpt" in world_model_text
    assert "wm_step_*.ckpt" in world_model_text


def test_pretokenize_quotes_multi_gpu_hydra_value() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "preprocess" / "20_pretokenize_dataset.sh"
    text = script.read_text(encoding="utf-8")
    assert 'gpu_devices="\'${GPUS}\'"' in text


def test_cotrain_world_model_ddp_tracks_unused_parameters() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "dreamervla" / "runners" / "online_cotrain_runner.py").read_text(
        encoding="utf-8"
    )
    assert "self.world_model," in source
    assert "find_unused_parameters=True" in source


def test_experiment_stage_check_module_exposes_required_commands() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "dreamervla" / "diagnostics" / "experiment_stage_checks.py"
    ).read_text(encoding="utf-8")

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
