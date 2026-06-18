from __future__ import annotations

from pathlib import Path

import pytest


def test_ray_launcher_plan_wires_coldstart_outputs_into_cotrain_warmup(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="ray", run_root=tmp_path, python="python")

    reward_dir = str(tmp_path / "coldstart" / "reward")
    hidden_dir = str(tmp_path / "coldstart" / "hidden")
    assert plan.mode == "ray"
    assert "experiment=collect_rollouts_ray" in plan.collect_cmd
    assert "rollout.target_episodes=4" in plan.collect_cmd
    assert f"task.openvla_oft.hdf5_reward_dir={reward_dir}" in plan.collect_cmd
    assert f"task.openvla_oft.action_hidden_dir={hidden_dir}" in plan.collect_cmd
    assert "experiment=online_cotrain_pipeline_oft_action_hidden" in plan.cotrain_cmd
    assert f"offline_warmup.data_dir={reward_dir}" in plan.cotrain_cmd
    assert f"offline_warmup.hidden_dir={hidden_dir}" in plan.cotrain_cmd
    assert "training.debug=false" in plan.cotrain_cmd
    assert "training.wm_warmup_steps=1" in plan.cotrain_cmd
    assert "training.classifier_warmup_steps=1" in plan.cotrain_cmd
    assert "online_rollout.total_env_steps=0" in plan.cotrain_cmd


@pytest.mark.parametrize(
    ("task", "hydra_task", "suite"),
    [
        ("goal", "OpenVLA_Onetraj_ColdStart_LIBERO", "libero_goal"),
        ("object", "OpenVLA_Onetraj_ColdStart_LIBERO_Object", "libero_object"),
        ("spatial", "OpenVLA_Onetraj_ColdStart_LIBERO_Spatial", "libero_spatial"),
    ],
)
def test_launcher_plan_accepts_libero_task_input(tmp_path, task, hydra_task, suite) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="ray", task=task, run_root=tmp_path, python="python")

    assert plan.task == task
    assert f"task={hydra_task}" in plan.collect_cmd
    assert f"+task={hydra_task}" in plan.cotrain_cmd
    assert f"env.task_suite_name={suite}" in plan.cotrain_cmd


@pytest.mark.parametrize(
    ("task", "ckpt_name"),
    [
        ("goal", "Openvla-oft-SFT-libero-goal-traj1"),
        ("object", "Openvla-oft-SFT-libero-object-traj1"),
        ("spatial", "Openvla-oft-SFT-libero-spatial-traj1"),
    ],
)
def test_launcher_plan_uses_oft_discrete_token_cotrain_smoke_defaults(
    tmp_path,
    task,
    ckpt_name,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="noray", task=task, run_root=tmp_path, python="python")

    assert "world_model.obs_dim=229376" in plan.cotrain_cmd
    assert "world_model.token_count=56" in plan.cotrain_cmd
    assert "world_model.token_dim=4096" in plan.cotrain_cmd
    assert "world_model.chunk_size=8" in plan.cotrain_cmd
    assert "+world_model.time_horizon=8" in plan.cotrain_cmd
    assert "world_model.chunk_rollout_chunks=1" in plan.cotrain_cmd
    assert "policy._target_=dreamervla.models.actor.OpenVLADiscreteTokenActor" in plan.cotrain_cmd
    assert "policy.action_hidden_dim=4096" in plan.cotrain_cmd
    assert "policy.time_horizon=8" in plan.cotrain_cmd
    assert "policy.head_type=oft_discrete_token" in plan.cotrain_cmd
    assert any(
        override.startswith("+policy.init_lm_head_ckpt=") and override.endswith(ckpt_name)
        for override in plan.cotrain_cmd
    )
    assert "classifier.window=2" in plan.cotrain_cmd
    assert "+classifier.token_dim=4096" in plan.cotrain_cmd
    assert "+classifier.token_pool=mean" in plan.cotrain_cmd
    assert "+classifier.token_count=56" in plan.cotrain_cmd
    assert "algorithm.wmpo.chunk_size=8" in plan.cotrain_cmd


def test_noray_launcher_plan_uses_pure_hydra_collector(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="noray", run_root=tmp_path, python="python")

    reward_dir = str(tmp_path / "coldstart" / "reward")
    hidden_dir = str(tmp_path / "coldstart" / "hidden")
    assert plan.mode == "noray"
    assert "experiment=collect_rollouts_onetraj" in plan.collect_cmd
    assert "experiment=collect_rollouts_ray" not in plan.collect_cmd
    assert "collect.envs_per_gpu=1" in plan.collect_cmd
    assert f"task.openvla_oft.hdf5_reward_dir={reward_dir}" in plan.collect_cmd
    assert f"task.openvla_oft.action_hidden_dir={hidden_dir}" in plan.collect_cmd
    assert f"offline_warmup.data_dir={reward_dir}" in plan.cotrain_cmd
    assert f"offline_warmup.hidden_dir={hidden_dir}" in plan.cotrain_cmd


def test_launcher_dry_run_prints_both_commands(tmp_path, capsys) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import main

    exit_code = main(["--mode", "noray", "--task", "object", "--run-root", str(tmp_path), "--dry-run"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "mode: noray" in out
    assert "task: object" in out
    assert "collect:" in out
    assert "cotrain:" in out
    assert "task=OpenVLA_Onetraj_ColdStart_LIBERO_Object" in out
    assert "+task=OpenVLA_Onetraj_ColdStart_LIBERO_Object" in out
    assert "offline_warmup.data_dir" in out


def test_asset_validation_reports_missing_inputs(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import validate_input_assets

    errors = validate_input_assets(data_root=tmp_path)

    assert any("OpenVLA-OFT checkpoint" in error for error in errors)
    assert any("LIBERO dataset" in error for error in errors)


@pytest.mark.parametrize(
    ("task", "ckpt_name", "suite", "stats_key"),
    [
        ("goal", "Openvla-oft-SFT-libero-goal-traj1", "libero_goal", "libero_goal_no_noops"),
        ("object", "Openvla-oft-SFT-libero-object-traj1", "libero_object", "libero_object_no_noops"),
        ("spatial", "Openvla-oft-SFT-libero-spatial-traj1", "libero_spatial", "libero_spatial_no_noops"),
    ],
)
def test_asset_validation_accepts_minimal_expected_layout(
    tmp_path,
    task,
    ckpt_name,
    suite,
    stats_key,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import validate_input_assets

    ckpt = tmp_path / "checkpoints" / "Openvla-oft-SFT-traj1" / ckpt_name
    libero = tmp_path / "datasets" / "libero" / suite
    ckpt.mkdir(parents=True)
    libero.mkdir(parents=True)
    (ckpt / "dataset_statistics.json").write_text(f'{{"{stats_key}": {{}}}}', encoding="utf-8")
    (libero / "demo.hdf5").touch()

    assert validate_input_assets(task=task, data_root=tmp_path) == []


def test_e2e_shell_scripts_select_expected_modes() -> None:
    root = Path(__file__).resolve().parents[2]
    ray = root / "scripts" / "e2e_coldstart_warmup_cotrain_ray.sh"
    noray = root / "scripts" / "e2e_coldstart_warmup_cotrain_noray.sh"

    assert ray.is_file()
    assert noray.is_file()
    ray_text = ray.read_text(encoding="utf-8")
    noray_text = noray.read_text(encoding="utf-8")
    assert "--mode ray" in ray_text
    assert "--mode noray" in noray_text
    assert 'CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"' in ray_text
    assert 'CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"' in noray_text
    assert 'conda activate "${CONDA_ENV_NAME}"' in ray_text
    assert 'conda activate "${CONDA_ENV_NAME}"' in noray_text
