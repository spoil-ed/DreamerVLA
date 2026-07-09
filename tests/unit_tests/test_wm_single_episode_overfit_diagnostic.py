from __future__ import annotations

from pathlib import Path

import pytest


def test_default_artifact_paths_follow_dvla_data_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dreamervla.diagnostics import wm_single_episode_overfit as diag

    data_root = tmp_path / "asset-root"
    monkeypatch.setenv("DVLA_ROOT", "/repo/root")
    monkeypatch.setenv("DVLA_DATA_ROOT", str(data_root))
    monkeypatch.setattr("sys.argv", ["wm_single_episode_overfit"])

    args = diag.parse_args()

    assert args.resolved_config == data_root / (
        "outputs/coldstart_warmup_cotrain/"
        "fixed_cls_wm_vla_eval_g7_component_20260707_205109/"
        "cotrain/resolved_config.yaml"
    )
    assert args.wm_ckpt == data_root / (
        "outputs/world_model_probe/current_actions_reward0_20260708_01/"
        "wm_probe_step1200.ckpt"
    )
    assert args.classifier_ckpt == data_root / (
        "outputs/coldstart_warmup_cotrain/"
        "fixed_wm_wmpo_cls_mainline_20260707_01/init/"
        "fixed_wm_wmpo_cls_init.ckpt"
    )
    assert args.hidden_hdf5 == data_root / (
        "processed_data/"
        "libero_goal_no_noops_t_256_oft_input_token_embedding_vla_policy_h1/"
        "open_the_middle_drawer_of_the_cabinet_demo.hdf5"
    )
    assert args.raw_hdf5 == data_root / (
        "processed_data/libero_goal_no_noops_t_256/"
        "open_the_middle_drawer_of_the_cabinet_demo.hdf5"
    )
    assert args.out_dir == data_root / "outputs/world_model_probe/single_episode_overfit"


def test_split_stage_cli_accepts_trained_checkpoint_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dreamervla.diagnostics import wm_single_episode_overfit as diag

    trained_ckpt = tmp_path / "wm_single_episode_step1200.ckpt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "wm_single_episode_overfit",
            "--stage",
            "eval",
            "--trained-wm-ckpt",
            str(trained_ckpt),
        ],
    )

    args = diag.parse_args()

    assert args.stage == "eval"
    assert args.trained_wm_ckpt == trained_ckpt


def test_split_experiment_scripts_are_stage_specific() -> None:
    root = Path(__file__).resolve().parents[2]
    experiments_dir = root / "scripts" / "experiments"
    expected = {
        "wm_single_episode_00_check.sh": "--stage check",
        "wm_single_episode_01_train.sh": "--stage train",
        "wm_single_episode_02_eval.sh": "--stage eval",
    }

    for name, marker in expected.items():
        script = experiments_dir / name
        assert script.is_file()
        text = script.read_text(encoding="utf-8")
        assert marker in text
        assert "dreamervla.diagnostics.wm_single_episode_overfit" in text
