from __future__ import annotations

from pathlib import Path


def test_experiment_stage_scripts_cover_mainline_plan() -> None:
    root = Path(__file__).resolve().parents[2]
    experiments_dir = root / "scripts" / "experiments"
    expected = {
        "collect_00_check.sh": "experiment_stage_checks collect-check",
        "collect_01_run.sh": "experiment_stage_checks collect-run",
        "collect_02_check.sh": "experiment_stage_checks collect-output",
        "cls_00_check.sh": "experiment_stage_checks cls-check",
        "cls_01_train.sh": "dreamervla.train",
        "cls_02_eval.sh": "experiment_stage_checks cls-eval",
        "wm_00_check.sh": "experiment_stage_checks wm-check",
        "wm_01_train.sh": "coldstart_warmup_cotrain",
        "wm_02_eval.sh": "eval_chunkwm_closeloop",
        "wm_cls_init_00_pack.sh": "experiment_stage_checks pack-init",
        "cotrain_00_check.sh": "experiment_stage_checks cotrain-check",
        "cotrain_01_run.sh": "coldstart_warmup_cotrain",
        "cotrain_02_eval.sh": "eval_libero_vla",
        "libero_original_00_reprocess_data.sh": "prepare_libero_data.sh",
        "libero_original_00_check.sh": "libero-original-check",
        "libero_original_01_train_cls_best.sh": "libero-original-cls-run",
        "libero_original_02_warmup_wm_cls_best.sh": "libero-original-warmup-run",
        "wm_full_dataset_train.sh": "libero-original-warmup-run",
        "wm_full_dataset_prepare.sh": "libero_original_00_reprocess_data.sh",
        "libero_original_03_rl_from_best.sh": "libero-original-rl-run",
        "libero_original_04_eval_rl.sh": "eval_libero_vla",
    }

    for name, marker in expected.items():
        script = experiments_dir / name
        assert script.is_file(), name
        text = script.read_text(encoding="utf-8")
        assert marker in text, name
        assert "DVLA_DATA_ROOT" in text, name
        assert "PYTHON_BIN" in text, name

    full_wm = (experiments_dir / "wm_full_dataset_train.sh").read_text(encoding="utf-8")
    assert "--classifier-steps 0" in full_wm
    assert "--classifier-batch-size 1" in full_wm
    assert "--wm-steps" in full_wm

    prepare = (experiments_dir / "wm_full_dataset_prepare.sh").read_text(encoding="utf-8")
    assert 'PREPROCESS_OVERWRITE="${PREPROCESS_OVERWRITE:-false}"' in prepare
    assert 'OFT_LATENT_SCHEME="${OFT_LATENT_SCHEME:-input_tokens}"' in prepare
    assert 'exec "${DVLA_ROOT}/scripts/experiments/libero_original_00_reprocess_data.sh"' in prepare


def test_libero_original_reprocess_script_targets_artifact_root() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "experiments" / "libero_original_00_reprocess_data.sh"
    text = script.read_text(encoding="utf-8")

    assert "task=libero_goal" in text
    assert "libero_suite=libero_goal" in text
    assert "task_name=openvla_onetraj_libero" in text
    assert "artifact_name=OpenVLA_Onetraj_LIBERO_libero_goal" in text
    assert "only=[10_hdf5_reward,20_pretokenize_dataset,35_oft_action_hidden,40_validate]" in text
    assert 'PREPROCESS_OVERWRITE="${PREPROCESS_OVERWRITE:-false}"' in text
    assert "OFT_LATENT_SCHEME" in text
    assert "both" in text
    assert "Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1" in text


def test_libero_original_reprocess_infers_ngpu_from_cuda_visible_devices() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "experiments" / "libero_original_00_reprocess_data.sh"
    text = script.read_text(encoding="utf-8")

    assert "_infer_gpu_count" in text
    assert 'PREPROCESS_NGPU="${PREPROCESS_NGPU:-${NGPU:-$(_infer_gpu_count "${PREPROCESS_GPUS}")}}"' in text
    assert 'OFT_ACTION_HIDDEN_GPUS_VALUE="${OFT_ACTION_HIDDEN_GPUS:-${PREPROCESS_NGPU}}"' in text


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
