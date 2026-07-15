from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf


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

    assert args.run_config == data_root / (
        "outputs/coldstart_warmup_cotrain/"
        "fixed_cls_wm_vla_eval_g7_component_20260707_205109/"
        "cotrain/.hydra/config.yaml"
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
        "OpenVLA_Onetraj_LIBERO_libero_goal/"
        "no_noops_t_256_oft_hidden_token_vla_policy_h1/"
        "open_the_middle_drawer_of_the_cabinet_demo.hdf5"
    )
    assert args.raw_hdf5 == data_root / (
        "processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/"
        "no_noops_t_256_remaining_reward/"
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


def test_legacy_resolved_config_flag_is_hidden_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dreamervla.diagnostics import wm_single_episode_overfit as diag

    legacy = tmp_path / "resolved_config.yaml"
    monkeypatch.setattr(
        "sys.argv",
        ["wm_single_episode_overfit", "--resolved-config", str(legacy)],
    )

    args = diag.parse_args()

    assert args.run_config == legacy
    monkeypatch.setattr("sys.argv", ["wm_single_episode_overfit", "--help"])
    with pytest.raises(SystemExit):
        diag.parse_args()
    help_text = capsys.readouterr().out
    assert "--run-config" in help_text
    assert "--resolved-config" not in help_text


def test_component_config_loading_uses_shared_run_config_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dreamervla.diagnostics import wm_single_episode_overfit as diag

    run_config = tmp_path / ".hydra" / "config.yaml"
    observed: list[Path] = []
    persisted = OmegaConf.create(
        {
            "ray_components": {
                "world_model": {
                    "target": "world.Model",
                    "kwargs": {
                        "reward_loss_scale": 1.0,
                        "chunk_rollout_chunks": 4,
                        "chunk_rollout_loss_scale": 0.5,
                    },
                },
                "classifier": {
                    "target": "classifier.Model",
                    "kwargs": {"hidden_dim": 8},
                },
            }
        }
    )

    def spy_load(path: str | Path) -> DictConfig:
        observed.append(Path(path))
        return persisted

    monkeypatch.setattr(diag, "load_run_config", spy_load)

    world_model, classifier = diag._load_component_configs(run_config)

    assert observed == [run_config]
    assert world_model["target"] == "world.Model"
    assert world_model["kwargs"]["reward_loss_scale"] == 0.0
    assert world_model["kwargs"]["chunk_rollout_chunks"] == 1
    assert world_model["kwargs"]["chunk_rollout_loss_scale"] == 0.0
    assert classifier["target"] == "classifier.Model"
