from __future__ import annotations

from pathlib import Path


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
    assert "training.debug=true" in plan.cotrain_cmd


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

    exit_code = main(["--mode", "noray", "--run-root", str(tmp_path), "--dry-run"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "mode: noray" in out
    assert "collect:" in out
    assert "cotrain:" in out
    assert "offline_warmup.data_dir" in out


def test_asset_validation_reports_missing_inputs(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import validate_input_assets

    errors = validate_input_assets(data_root=tmp_path)

    assert any("OpenVLA-OFT checkpoint" in error for error in errors)
    assert any("LIBERO dataset" in error for error in errors)


def test_asset_validation_accepts_minimal_expected_layout(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import validate_input_assets

    ckpt = tmp_path / "checkpoints" / "Openvla-oft-SFT-traj1" / "Openvla-oft-SFT-libero-goal-traj1"
    libero = tmp_path / "datasets" / "libero" / "libero_goal"
    ckpt.mkdir(parents=True)
    libero.mkdir(parents=True)
    (ckpt / "dataset_statistics.json").write_text('{"libero_goal_no_noops": {}}', encoding="utf-8")
    (libero / "demo.hdf5").touch()

    assert validate_input_assets(data_root=tmp_path) == []


def test_e2e_shell_scripts_select_expected_modes() -> None:
    root = Path(__file__).resolve().parents[2]
    ray = root / "scripts" / "e2e_coldstart_warmup_cotrain_ray.sh"
    noray = root / "scripts" / "e2e_coldstart_warmup_cotrain_noray.sh"

    assert ray.is_file()
    assert noray.is_file()
    assert "--mode ray" in ray.read_text(encoding="utf-8")
    assert "--mode noray" in noray.read_text(encoding="utf-8")
