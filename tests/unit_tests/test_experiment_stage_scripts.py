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
        "libero_original_00_check.sh": "libero-original-check",
        "libero_original_01_train_cls_best.sh": "libero-original-cls-run",
        "libero_original_02_warmup_wm_cls_best.sh": "libero-original-warmup-run",
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
