from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _launcher_cfg() -> dict:
    from dreamervla.utils.hydra_config import script_config

    return script_config("coldstart_warmup_cotrain")


def _plan_context(plan, cfg: dict) -> dict:
    task_spec = dict(cfg["tasks"][plan.task])
    return {
        **task_spec,
        "task": plan.task,
        "mode": plan.mode,
        "profile": plan.profile,
        "run_root": str(plan.run_root),
        "reward_dir": str(plan.reward_dir),
        "hidden_dir": str(plan.hidden_dir),
        "collect_out": str(plan.run_root / "collect"),
        "cotrain_out": str(plan.run_root / "cotrain"),
    }


def _render(items, context: dict) -> list[str]:
    from dreamervla.launchers.coldstart_warmup_cotrain import _render_overrides

    return _render_overrides(items, context)


def _assert_items_in_command(items: list[str], command: list[str]) -> None:
    for item in items:
        assert item in command


def _override_int(overrides: list[str], key: str) -> int:
    prefix = f"{key}="
    matches = [item for item in overrides if item.startswith(prefix)]
    assert len(matches) == 1
    return int(matches[0].split("=", 1)[1])


def _override_key(override: str) -> str:
    return override.split("=", 1)[0].removeprefix("+")


def test_ray_launcher_plan_wires_coldstart_outputs_into_cotrain_warmup(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="ray", run_root=tmp_path, python="python", profile="smoke")
    cfg = _launcher_cfg()
    context = _plan_context(plan, cfg)

    assert plan.mode == "ray"
    assert f"task.openvla_oft.hdf5_reward_dir={plan.reward_dir}" in plan.collect_cmd
    assert f"task.openvla_oft.action_hidden_dir={plan.hidden_dir}" in plan.collect_cmd
    assert f"offline_warmup.data_dir={plan.reward_dir}" in plan.cotrain_cmd
    assert f"offline_warmup.hidden_dir={plan.hidden_dir}" in plan.cotrain_cmd
    _assert_items_in_command(
        _render(cfg["modes"][plan.mode]["collect"], context),
        plan.collect_cmd,
    )
    _assert_items_in_command(
        _render(cfg["profiles"][plan.profile]["collect"][plan.mode], context),
        plan.collect_cmd,
    )
    _assert_items_in_command(
        _render(cfg["profiles"][plan.profile]["cotrain"], context),
        plan.cotrain_cmd,
    )


@pytest.mark.parametrize(
    "task",
    list(_launcher_cfg()["tasks"]),
)
def test_launcher_plan_accepts_libero_task_input(tmp_path, task) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="ray", task=task, run_root=tmp_path, python="python")
    task_spec = _launcher_cfg()["tasks"][task]

    assert plan.task == task
    assert f"task={task_spec['hydra_task']}" in plan.collect_cmd
    assert f"task={task_spec['hydra_task']}" in plan.cotrain_cmd
    assert not any(item.startswith("env.task_suite_name=") for item in plan.cotrain_cmd)


@pytest.mark.parametrize(
    "task",
    list(_launcher_cfg()["tasks"]),
)
def test_launcher_plan_uses_profile_runtime_cotrain_overrides(
    tmp_path,
    task,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="noray",
        task=task,
        run_root=tmp_path,
        python="python",
        profile="smoke",
    )
    cfg = _launcher_cfg()
    context = _plan_context(plan, cfg)

    _assert_items_in_command(
        _render(cfg["profiles"][plan.profile]["cotrain"], context),
        plan.cotrain_cmd,
    )


def test_noray_launcher_plan_uses_pure_hydra_collector(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="noray", run_root=tmp_path, python="python")
    cfg = _launcher_cfg()
    context = _plan_context(plan, cfg)

    assert plan.mode == "noray"
    assert "experiment=collect_rollouts_ray" not in plan.collect_cmd
    assert f"task.openvla_oft.hdf5_reward_dir={plan.reward_dir}" in plan.collect_cmd
    assert f"task.openvla_oft.action_hidden_dir={plan.hidden_dir}" in plan.collect_cmd
    assert f"offline_warmup.data_dir={plan.reward_dir}" in plan.cotrain_cmd
    assert f"offline_warmup.hidden_dir={plan.hidden_dir}" in plan.cotrain_cmd
    _assert_items_in_command(
        _render(cfg["modes"][plan.mode]["collect"], context),
        plan.collect_cmd,
    )
    _assert_items_in_command(
        _render(cfg["profiles"][plan.profile]["collect"][plan.mode], context),
        plan.collect_cmd,
    )


def test_default_launcher_profile_uses_single_gpu_utilization_defaults(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="noray", run_root=tmp_path, python="python")
    cfg = _launcher_cfg()
    context = _plan_context(plan, cfg)

    assert plan.profile == cfg["profile"]
    _assert_items_in_command(
        _render(cfg["profiles"][plan.profile]["collect"][plan.mode], context),
        plan.collect_cmd,
    )
    _assert_items_in_command(
        _render(cfg["profiles"][plan.profile]["cotrain"], context),
        plan.cotrain_cmd,
    )


def test_default_launcher_profile_is_release_parallel_collector(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    plan = build_pipeline_plan(mode="noray", run_root=tmp_path, python="python")
    root = Path(__file__).resolve().parents[2]
    collect_recipe = OmegaConf.load(root / "configs" / "experiment" / "collect_rollouts_onetraj.yaml")
    baseline = int(collect_recipe.collect.envs_per_gpu)
    default_profile = cfg["default_profile"]
    default_collect = cfg["profiles"][default_profile]["collect"][plan.mode]

    assert cfg["profile"] == default_profile
    assert plan.profile == default_profile
    assert _override_int(default_collect, "collect.envs_per_gpu") > baseline


def test_launcher_profiles_do_not_override_model_structure() -> None:
    cfg = _launcher_cfg()
    runtime_keys = {
        "training.debug",
        "training.wm_warmup_steps",
        "training.classifier_warmup_steps",
        "training.classifier_batch_size",
        "dataloader.batch_size",
        "online_rollout.buffer_size",
        "online_rollout.total_env_steps",
    }

    for profile in cfg["profiles"].values():
        assert {_override_key(item) for item in profile["cotrain"]} <= runtime_keys


@pytest.mark.parametrize(
    "task_name",
    list(_launcher_cfg()["tasks"]),
)
def test_oft_cotrain_recipe_derives_structure_from_task_vla_config(task_name) -> None:
    hydra_task = _launcher_cfg()["tasks"][task_name]["hydra_task"]
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=online_cotrain_pipeline_oft_action_hidden",
                f"task={hydra_task}",
            ],
        )
    OmegaConf.resolve(cfg)
    oft = cfg.task.openvla_oft

    assert cfg.env.task_suite_name == cfg.task.suite
    assert cfg.world_model._target_ == oft.wm_target
    assert cfg.world_model.obs_dim == oft.wm_obs_dim
    assert cfg.world_model.token_count == oft.token_count
    assert cfg.world_model.token_dim == oft.token_dim
    assert cfg.world_model.chunk_size == oft.chunk_size
    assert cfg.world_model.time_horizon == oft.time_horizon
    assert cfg.policy._target_ == oft.actor_target
    assert cfg.policy.action_hidden_dim == oft.token_dim
    assert cfg.policy.time_horizon == oft.chunk_size
    assert cfg.policy.head_type == oft.actor_head_type
    assert cfg.policy.adapter_type == oft.actor_adapter_type
    assert cfg.policy.adapter_hidden_dim == oft.actor_adapter_hidden_dim
    assert cfg.policy.init_lm_head_ckpt == oft.ckpt_path
    assert cfg.policy.vocab_size == oft.vocab_size
    assert cfg.policy.action_token_bins == oft.action_token_bins
    assert cfg.policy.min_action == oft.min_action
    assert cfg.policy.max_action == oft.max_action
    assert cfg.classifier.latent_dim == oft.token_dim
    assert cfg.classifier.chunk_size == oft.chunk_size
    assert cfg.algorithm.wmpo.chunk_size == oft.chunk_size


def test_launcher_dry_run_prints_both_commands(tmp_path, capsys) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import main

    cfg = _launcher_cfg()
    mode = "noray"
    task = next(task_name for task_name in cfg["tasks"] if task_name != cfg["task"])
    exit_code = main(
        [
            f"mode={mode}",
            f"task={task}",
            f"run_root={tmp_path}",
            "dry_run=true",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert f"mode: {mode}" in out
    assert f"task: {task}" in out
    assert "collect:" in out
    assert "cotrain:" in out
    assert f"task={cfg['tasks'][task]['hydra_task']}" in out
    assert "offline_warmup.data_dir" in out


def test_launcher_accepts_hydra_list_overrides(tmp_path, capsys) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import main

    cfg = _launcher_cfg()
    collect_override = cfg["modes"]["ray"]["collect"][3].replace("[0]", "all")
    profile_override = cfg["profiles"][cfg["profile"]]["collect"]["ray"][-1]
    profile_key = profile_override.split("=", 1)[0]
    collect_override_2 = f"{profile_key}=9"
    cotrain_profile = cfg["profiles"][cfg["profile"]]["cotrain"]
    cotrain_key = next(item.split("=", 1)[0] for item in cotrain_profile if item.startswith("online_rollout.total_env_steps="))
    cotrain_override = f"{cotrain_key}=10"
    exit_code = main(
        [
            "mode=ray",
            f"run_root={tmp_path}",
            "dry_run=true",
            f'collect_overrides=["{collect_override}","{collect_override_2}"]',
            f'cotrain_overrides=["{cotrain_override}"]',
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert collect_override in out
    assert collect_override_2 in out
    assert cotrain_override in out


@pytest.mark.parametrize(
    ("mode", "concurrency_override", "expected_concurrency_key"),
    [
        ("noray", "collect.envs_per_gpu=6", "collect.envs_per_gpu"),
        ("ray", "collect.num_workers=5", "env.num_workers"),
    ],
)
def test_launcher_exposes_direct_hydra_controls_for_collection_and_warmup(
    tmp_path,
    capsys,
    mode,
    concurrency_override,
    expected_concurrency_key,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import main

    episodes_per_task = 7
    episode_horizon = 123
    wm_steps = 11
    classifier_steps = 12
    batch_size = 13
    classifier_batch_size = 14
    concurrency_value = concurrency_override.split("=", 1)[1]
    exit_code = main(
        [
            f"mode={mode}",
            f"run_root={tmp_path}",
            "dry_run=true",
            f"collect.episodes_per_task={episodes_per_task}",
            f"collect.episode_horizon={episode_horizon}",
            concurrency_override,
            f"warmup.wm_steps={wm_steps}",
            f"warmup.classifier_steps={classifier_steps}",
            f"warmup.batch_size={batch_size}",
            f"warmup.classifier_batch_size={classifier_batch_size}",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert f"collect.episodes_per_task={episodes_per_task}" in out
    assert f"collect.episode_horizon={episode_horizon}" in out
    assert f"{expected_concurrency_key}={concurrency_value}" in out
    assert f"training.wm_warmup_steps={wm_steps}" in out
    assert f"training.classifier_warmup_steps={classifier_steps}" in out
    assert f"dataloader.batch_size={batch_size}" in out
    assert f"training.classifier_batch_size={classifier_batch_size}" in out


def test_launcher_prints_phase_start_banners(tmp_path, monkeypatch, capsys) -> None:
    import dreamervla.launchers.coldstart_warmup_cotrain as mod

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, cmd, **_kwargs):
            self.calls.append(list(cmd))

    rec = _Recorder()
    monkeypatch.setattr(mod, "subprocess", rec)

    exit_code = mod.main(
        [f"run_root={tmp_path}", f"data_root={tmp_path}", "skip_asset_check=true"]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    collect_at = out.find("PHASE 1/2 START: cold-start collection")
    cotrain_at = out.find("PHASE 2/2 START: offline-warmup online cotrain")
    assert collect_at != -1, out
    assert cotrain_at != -1, out
    assert collect_at < cotrain_at  # collect banner streams before cotrain banner
    assert len(rec.calls) == 2  # collect then cotrain both launched


def test_launcher_banner_marks_skipped_collection(tmp_path, monkeypatch, capsys) -> None:
    import dreamervla.launchers.coldstart_warmup_cotrain as mod

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, cmd, **_kwargs):
            self.calls.append(list(cmd))

    rec = _Recorder()
    monkeypatch.setattr(mod, "subprocess", rec)

    exit_code = mod.main(
        [
            f"run_root={tmp_path}",
            f"data_root={tmp_path}",
            "skip_asset_check=true",
            "skip_collect=true",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "PHASE 1/2 SKIPPED: cold-start collection" in out
    assert "PHASE 2/2 START: offline-warmup online cotrain" in out
    assert len(rec.calls) == 1  # only cotrain launched when collection is skipped


def test_ray_launcher_does_not_freeze_total_episodes_when_episodes_per_task_changes(
    tmp_path,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="ray", run_root=tmp_path, python="python")

    assert not any(item.startswith("rollout.target_episodes=") for item in plan.collect_cmd)


def test_plan_uses_unified_collected_rollouts_space(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DVLA_DATA_ROOT", str(tmp_path))
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    run_root = tmp_path / "run"
    plan = build_pipeline_plan(mode="noray", task="goal", run_root=run_root, python="python")

    # Collected data lives in the stable, per-suite unified space (not the run_root).
    assert plan.reward_dir == tmp_path / "collected_rollouts" / "libero_goal" / "reward"
    assert plan.hidden_dir == tmp_path / "collected_rollouts" / "libero_goal" / "hidden"
    assert plan.collected_root == tmp_path / "collected_rollouts" / "libero_goal"
    # Training outputs stay isolated under the timestamped run_root.
    assert f"training.out_dir={run_root / 'collect'}" in plan.collect_cmd
    assert f"training.out_dir={run_root / 'cotrain'}" in plan.cotrain_cmd


def test_launcher_resume_skips_collection_and_writes_manifest(tmp_path, monkeypatch, capsys) -> None:
    import h5py

    import dreamervla.launchers.coldstart_warmup_cotrain as mod
    from dreamervla.dataset.collection_manifest import read_manifest

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, cmd, **_kwargs):
            self.calls.append(list(cmd))

    rec = _Recorder()
    monkeypatch.setattr(mod, "subprocess", rec)

    reward = tmp_path / "collected_rollouts" / "libero_goal" / "reward"
    reward.mkdir(parents=True)
    with h5py.File(str(reward / "shard_000.hdf5"), "w") as f:
        data = f.create_group("data")
        data.attrs["num_demos"] = 6
        for i in range(6):
            data.create_group(f"demo_{i}")

    exit_code = mod.main(
        [
            f"run_root={tmp_path / 'run'}",
            f"data_root={tmp_path}",
            "task=goal",
            "skip_asset_check=true",
            "collect_target_episodes=6",
            "collect_num_tasks=2",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "PHASE 1/2 SKIPPED" in out
    assert len(rec.calls) == 1  # only cotrain runs; collection target already met

    manifest = read_manifest(tmp_path / "collected_rollouts" / "libero_goal")
    assert manifest["collected_episodes"] == 6
    assert manifest["status"] == "complete"


def test_asset_validation_reports_missing_inputs(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import validate_input_assets

    errors = validate_input_assets(data_root=tmp_path)

    assert any("OpenVLA-OFT checkpoint" in error for error in errors)
    assert any("LIBERO dataset" in error for error in errors)


@pytest.mark.parametrize(
    "task",
    list(_launcher_cfg()["tasks"]),
)
def test_asset_validation_accepts_minimal_expected_layout(
    tmp_path,
    task,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import validate_input_assets

    task_spec = _launcher_cfg()["tasks"][task]
    ckpt = tmp_path / "checkpoints" / "Openvla-oft-SFT-traj1" / task_spec["ckpt_name"]
    libero = tmp_path / "datasets" / "libero" / task_spec["suite"]
    ckpt.mkdir(parents=True)
    libero.mkdir(parents=True)
    (ckpt / "dataset_statistics.json").write_text(f'{{"{task_spec["stats_key"]}": {{}}}}', encoding="utf-8")
    (libero / "demo.hdf5").touch()

    assert validate_input_assets(task=task, data_root=tmp_path) == []


def test_reused_coldstart_output_validation_requires_sidecar_metadata(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import validate_collected_outputs

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    reward_dir.mkdir()
    hidden_dir.mkdir()
    (reward_dir / "shard_000.hdf5").touch()
    (hidden_dir / "shard_000.hdf5").touch()

    errors = validate_collected_outputs(reward_dir=reward_dir, hidden_dir=hidden_dir)

    assert any("preprocess_config.json" in error for error in errors)

    (hidden_dir / "preprocess_config.json").write_text(
        '{"hidden_key": "obs_embedding"}',
        encoding="utf-8",
    )

    assert validate_collected_outputs(reward_dir=reward_dir, hidden_dir=hidden_dir) == []


def test_e2e_shell_scripts_select_expected_modes() -> None:
    root = Path(__file__).resolve().parents[2]
    ray = root / "scripts" / "e2e_coldstart_warmup_cotrain_ray.sh"
    noray = root / "scripts" / "e2e_coldstart_warmup_cotrain_noray.sh"

    assert ray.is_file()
    assert noray.is_file()
    ray_text = ray.read_text(encoding="utf-8")
    noray_text = noray.read_text(encoding="utf-8")
    assert "--mode" not in ray_text
    assert "--mode" not in noray_text
    assert "mode=ray" in ray_text
    assert "mode=noray" in noray_text
    assert 'CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"' in ray_text
    assert 'CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"' in noray_text
    assert 'conda activate "${CONDA_ENV_NAME}"' in ray_text
    assert 'conda activate "${CONDA_ENV_NAME}"' in noray_text


def test_coldstart_launcher_has_no_argparse_cli() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "dreamervla" / "launchers" / "coldstart_warmup_cotrain.py").read_text(
        encoding="utf-8"
    )

    assert "import argparse" not in text
    assert "ArgumentParser" not in text
    assert "parse_args" not in text


def test_multi_gpu_wraps_cotrain_in_torchrun_but_not_collect(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="noray", run_root=tmp_path, python="python", ngpu=4
    )

    # cotrain runs under torchrun DDP across the requested GPUs ...
    assert "torch.distributed.run" in plan.cotrain_cmd
    assert "--nproc-per-node=4" in plan.cotrain_cmd
    assert plan.cotrain_cmd.count("dreamervla.train") == 1
    # ... while collection stays single-process (vectorized / ray fan-out).
    assert "torch.distributed.run" not in plan.collect_cmd


def test_single_gpu_cotrain_has_no_torchrun(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="noray", run_root=tmp_path, python="python", ngpu=1
    )

    assert "torch.distributed.run" not in plan.cotrain_cmd
    assert plan.cotrain_cmd[:3] == ["python", "-m", "dreamervla.train"]


def test_multi_gpu_profile_is_gpu_count_agnostic(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    assert "multi_gpu" in cfg["profiles"]
    # The profile bakes in NO GPU count — no fixed ray worker count tied to GPUs.
    profile = cfg["profiles"]["multi_gpu"]
    profile_keys = {_override_key(x) for x in profile["cotrain"]}
    for mode_overrides in profile["collect"].values():
        profile_keys |= {_override_key(x) for x in mode_overrides}
    assert "env.num_workers" not in profile_keys

    # GPU count is the variable `ngpu`: the same profile drives any count via torchrun.
    for n in (2, 4, 8):
        plan = build_pipeline_plan(
            mode="ray", run_root=tmp_path, python="python", profile="multi_gpu", ngpu=n
        )
        assert f"--nproc-per-node={n}" in plan.cotrain_cmd
        assert "torch.distributed.run" not in plan.collect_cmd


@pytest.mark.parametrize("profile", ["release", "multi_gpu"])
def test_full_profiles_run_full_pipeline_by_default(profile) -> None:
    cfg = _launcher_cfg()
    cotrain = cfg["profiles"][profile]["cotrain"]

    assert "training.debug=false" in cotrain
    assert "online_rollout.total_env_steps=200000" in cotrain
    assert "training.wm_warmup_steps=2000" in cotrain
    assert "training.classifier_warmup_steps=2000" in cotrain


def test_launcher_debug_control_appends_training_debug_to_cotrain(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan_default = build_pipeline_plan(mode="noray", run_root=tmp_path, python="python")
    assert "training.debug=true" not in plan_default.cotrain_cmd

    plan_debug = build_pipeline_plan(
        mode="noray", run_root=tmp_path, python="python", debug=True
    )
    assert "training.debug=true" in plan_debug.cotrain_cmd
    # debug only affects cotrain, never the collect command.
    assert "training.debug=true" not in plan_debug.collect_cmd
