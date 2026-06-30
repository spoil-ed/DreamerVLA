from __future__ import annotations

from pathlib import Path
import sys

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


def _override_values(overrides: list[str], key: str) -> list[str]:
    prefix = f"{key}="
    return [item.split("=", 1)[1] for item in overrides if item.startswith(prefix)]


def _override_key(override: str) -> str:
    return override.split("=", 1)[0].lstrip("+~")


def _assert_no_duplicate_override_keys(overrides: list[str]) -> None:
    keys = [
        _override_key(item)
        for item in overrides
        if "=" in item
    ]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    assert duplicates == []


def _write_complete_collected_pair(reward, hidden, shard_name: str, task_ids: list[int]) -> None:
    import h5py
    import numpy as np

    reward.mkdir(parents=True, exist_ok=True)
    hidden.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(reward / shard_name), "w") as rf, h5py.File(
        str(hidden / shard_name), "w"
    ) as hf:
        rdata = rf.create_group("data")
        hdata = hf.create_group("data")
        for idx, tid in enumerate(task_ids):
            demo = rdata.create_group(f"demo_{idx}")
            demo.attrs["task_id"] = int(tid)
            demo.attrs["episode_id"] = int(idx)
            demo.attrs["num_samples"] = "1"
            demo.attrs["complete"] = True
            demo.create_dataset("actions", data=np.zeros((1, 7), dtype=np.float32))
            demo.create_dataset("dones", data=np.ones((1,), dtype=np.uint8))
            demo.create_dataset("rewards", data=np.zeros((1,), dtype=np.float32))
            demo.create_dataset("sparse_rewards", data=np.zeros((1,), dtype=np.uint8))
            demo.create_dataset("robot_states", data=np.zeros((1, 9), dtype=np.float32))
            demo.create_dataset("states", data=np.zeros((1, 5), dtype=np.float32))
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_rgb", data=np.zeros((1, 1, 1, 3), dtype=np.uint8))
            obs.create_dataset("eye_in_hand_rgb", data=np.zeros((1, 1, 1, 3), dtype=np.uint8))
            obs.create_dataset("ee_pos", data=np.zeros((1, 3), dtype=np.float32))
            obs.create_dataset("ee_ori", data=np.zeros((1, 3), dtype=np.float32))
            obs.create_dataset("ee_states", data=np.zeros((1, 6), dtype=np.float32))
            obs.create_dataset("gripper_states", data=np.zeros((1, 2), dtype=np.float32))
            obs.create_dataset("joint_states", data=np.zeros((1, 7), dtype=np.float32))
            hdemo = hdata.create_group(f"demo_{idx}")
            hdemo.create_dataset("obs_embedding", data=np.zeros((1, 8), dtype=np.float16))
            hdemo.attrs["complete"] = True
        rdata.attrs["num_demos"] = len(task_ids)
        hdata.attrs["num_demos"] = len(task_ids)


def test_ray_launcher_plan_wires_coldstart_outputs_into_cotrain_warmup(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="ray", run_root=tmp_path, python="python", profile="smoke")
    cfg = _launcher_cfg()
    context = _plan_context(plan, cfg)

    assert plan.mode == "ray"
    assert f"task.openvla_oft.hdf5_reward_dir={plan.reward_dir}" in plan.collect_cmd
    assert f"task.openvla_oft.input_token_hidden_dir={plan.hidden_dir}" in plan.collect_cmd
    assert f"++collect.hidden_dir={plan.hidden_dir}" in plan.collect_cmd
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


def test_launcher_main_defaults_child_python_to_current_interpreter(
    monkeypatch,
    tmp_path,
) -> None:
    import dreamervla.launchers.coldstart_warmup_cotrain as mod

    captured: dict[str, str] = {}

    def fake_build_pipeline_plan(**kwargs):
        captured["python"] = kwargs["python"]
        return mod.PipelinePlan(
            mode="ray",
            profile="smoke",
            task="goal",
            run_root=tmp_path,
            collected_root=tmp_path / "collected",
            reward_dir=tmp_path / "reward",
            hidden_dir=tmp_path / "hidden",
            collect_cmd=[kwargs["python"], "-m", "dreamervla.train"],
            cotrain_cmd=[kwargs["python"], "-m", "dreamervla.train"],
        )

    monkeypatch.setattr(mod, "build_pipeline_plan", fake_build_pipeline_plan)

    assert mod.main(["dry_run=true", f"run_root={tmp_path}"]) == 0
    assert captured["python"] == sys.executable


def test_launcher_main_preserves_explicit_python_override(monkeypatch, tmp_path) -> None:
    import dreamervla.launchers.coldstart_warmup_cotrain as mod

    captured: dict[str, str] = {}

    def fake_build_pipeline_plan(**kwargs):
        captured["python"] = kwargs["python"]
        return mod.PipelinePlan(
            mode="ray",
            profile="smoke",
            task="goal",
            run_root=tmp_path,
            collected_root=tmp_path / "collected",
            reward_dir=tmp_path / "reward",
            hidden_dir=tmp_path / "hidden",
            collect_cmd=[kwargs["python"], "-m", "dreamervla.train"],
            cotrain_cmd=[kwargs["python"], "-m", "dreamervla.train"],
        )

    monkeypatch.setattr(mod, "build_pipeline_plan", fake_build_pipeline_plan)

    assert mod.main(["dry_run=true", f"run_root={tmp_path}", "python=python"]) == 0
    assert captured["python"] == "python"


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
    assert f"task.openvla_oft.input_token_hidden_dir={plan.hidden_dir}" in plan.collect_cmd
    assert f"++collect.hidden_dir={plan.hidden_dir}" in plan.collect_cmd
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
        "training.warmup_replay_epochs",
        "training.warmup_replay_max_steps",
        "training.warmup_checkpoint_every",
        "training.classifier_batch_size",
        "dataloader.batch_size",
        "online_rollout.buffer_size",
        "online_rollout.total_env_steps",
    }

    for profile in cfg["profiles"].values():
        assert {_override_key(item) for item in profile["cotrain"]} <= runtime_keys


def test_release_profiles_keep_full_coldstart_pool_for_warmup() -> None:
    cfg = _launcher_cfg()

    for profile_name in ("release", "multi_gpu"):
        buffer_size = _override_int(
            cfg["profiles"][profile_name]["cotrain"],
            "online_rollout.buffer_size",
        )
        assert buffer_size >= 160_000


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


@pytest.mark.parametrize("mode", ["noray", "ray"])
def test_launcher_forwards_demos_per_shard_to_collect(tmp_path, capsys, mode) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import main

    exit_code = main(
        [
            f"mode={mode}",
            f"run_root={tmp_path}",
            "dry_run=true",
            "collect.demos_per_shard=25",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    # The slicing knob lands in the collect command (both backends read it), never cotrain.
    collect_line = next(line for line in out.splitlines() if line.startswith("collect:"))
    cotrain_line = next(line for line in out.splitlines() if line.startswith("cotrain:"))
    assert "collect.demos_per_shard=25" in collect_line
    assert "demos_per_shard" not in cotrain_line


def test_ray_collect_scales_env_workers_with_ngpu(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="multi_gpu",
        ngpu=6,
    )
    # ray env-worker count scales with ngpu (6 * 4) when the profile does not set it.
    assert "env.num_workers=24" in plan.collect_cmd
    # noray sizes env concurrency via envs_per_gpu, not env.num_workers.
    noray = build_pipeline_plan(mode="noray", run_root=tmp_path, python="python", ngpu=6)
    assert not any(item.startswith("env.num_workers=") for item in noray.collect_cmd)


def test_ray_smoke_profile_num_workers_is_not_overwritten_by_ngpu_autoscale(
    tmp_path,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="smoke",
        ngpu=1,
    )

    env_worker_overrides = [
        item for item in plan.collect_cmd if item.startswith("env.num_workers=")
    ]
    assert env_worker_overrides == ["env.num_workers=2"]


def test_multi_gpu_profile_scales_sync_cotrain_rollout_envs_with_ngpu(
    tmp_path,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="noray",
        run_root=tmp_path,
        python="python",
        profile="multi_gpu",
        ngpu=6,
    )

    assert "online_rollout.num_envs=12" in plan.cotrain_cmd


def test_ray_explicit_num_workers_overrides_ngpu_scaling(tmp_path, capsys) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import main

    exit_code = main(
        [
            "mode=ray",
            f"run_root={tmp_path}",
            "dry_run=true",
            "collect.num_workers=3",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    collect_line = next(line for line in out.splitlines() if line.startswith("collect:"))
    assert "env.num_workers=3" in collect_line
    assert "env.num_workers=4" not in collect_line  # auto-scale suppressed by the override


def test_launcher_aggregates_collection_after_collect(tmp_path, monkeypatch, capsys) -> None:
    import dreamervla.launchers.coldstart_warmup_cotrain as mod

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, cmd, **_kwargs):
            self.calls.append(list(cmd))

    monkeypatch.setattr(mod, "subprocess", _Recorder())

    exit_code = mod.main(
        [f"run_root={tmp_path}", f"data_root={tmp_path}", "skip_asset_check=true"]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    # A single aggregate summary follows the multi-process collect.
    assert "PHASE 1/2 collected (aggregate across all processes):" in out


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
    hidden = tmp_path / "collected_rollouts" / "libero_goal" / "hidden"
    _write_complete_collected_pair(
        reward,
        hidden,
        "shard_000.hdf5",
        [0, 1, 0, 1, 0, 1],
    )

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


def test_launcher_prints_inspection_report_then_resumes(tmp_path, monkeypatch, capsys) -> None:
    import dreamervla.launchers.coldstart_warmup_cotrain as mod

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, cmd, **_kwargs):
            self.calls.append(list(cmd))

    rec = _Recorder()
    monkeypatch.setattr(mod, "subprocess", rec)

    reward = tmp_path / "collected_rollouts" / "libero_goal" / "reward"
    hidden = tmp_path / "collected_rollouts" / "libero_goal" / "hidden"
    _write_complete_collected_pair(reward, hidden, "shard_000.hdf5", [0, 1])

    exit_code = mod.main(
        [
            f"run_root={tmp_path / 'run'}",
            f"data_root={tmp_path}",
            "task=goal",
            "skip_asset_check=true",
            "collect_target_episodes=10",
            "collect_num_tasks=2",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    # Existing data is identified and reported before collecting (no blind load).
    assert "inspecting" in out
    assert "collected: 2 / 10" in out
    assert "task0=1" in out and "task1=1" in out
    assert "[resume] topping up 8" in out
    assert "collect.episodes_per_task=5" in rec.calls[0]
    assert len(rec.calls) == 2  # not complete -> collect + cotrain both run


def test_collect_resume_function_handles_skip_and_manifest(tmp_path, monkeypatch, capsys) -> None:
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
    monkeypatch.setenv("DVLA_DATA_ROOT", str(tmp_path))

    plan = mod.build_pipeline_plan(
        mode="ray",
        run_root=tmp_path / "run",
        python="python",
        launcher_cfg=_launcher_cfg(),
    )
    reward = plan.reward_dir
    hidden = plan.hidden_dir
    reward.mkdir(parents=True)
    hidden.mkdir(parents=True)
    with h5py.File(str(reward / "ray_shard_000.hdf5"), "w") as rf, h5py.File(
        str(hidden / "ray_shard_000.hdf5"), "w"
    ) as hf:
        rdata = rf.create_group("data")
        hdata = hf.create_group("data")
        for idx in range(2):
            demo = rdata.create_group(f"demo_{idx}")
            demo.attrs["task_id"] = idx
            demo.attrs["episode_id"] = 0
            demo.attrs["num_samples"] = "1"
            demo.create_dataset("actions", data=[[0, 0, 0, 0, 0, 0, 0]])
            demo.create_dataset("dones", data=[1])
            demo.create_dataset("rewards", data=[0])
            demo.create_dataset("sparse_rewards", data=[0])
            demo.create_dataset("robot_states", data=[[0] * 9])
            demo.create_dataset("states", data=[[0] * 5])
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_rgb", data=[[[[0, 0, 0]]]])
            obs.create_dataset("eye_in_hand_rgb", data=[[[[0, 0, 0]]]])
            obs.create_dataset("ee_pos", data=[[0, 0, 0]])
            obs.create_dataset("ee_ori", data=[[0, 0, 0]])
            obs.create_dataset("ee_states", data=[[0] * 6])
            obs.create_dataset("gripper_states", data=[[0] * 2])
            obs.create_dataset("joint_states", data=[[0] * 7])
            hdata.create_group(f"demo_{idx}").create_dataset("obs_embedding", data=[[0] * 8])
        rdata.attrs["num_demos"] = 2
        hdata.attrs["num_demos"] = 2

    result = mod.collect_resume(
        plan,
        target_episodes=2,
        num_tasks=2,
        skip_collect=False,
    )
    out = capsys.readouterr().out

    assert result["ran_collect"] is False
    assert "target 2 already collected" in out
    assert rec.calls == []
    manifest = read_manifest(plan.collected_root)
    assert manifest["status"] == "complete"
    assert manifest["collected_episodes"] == 2


def test_collect_resume_derives_target_from_collect_command_when_target_is_null(
    tmp_path, monkeypatch, capsys
) -> None:
    import dreamervla.launchers.coldstart_warmup_cotrain as mod

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, cmd, **_kwargs):
            self.calls.append(list(cmd))

    rec = _Recorder()
    monkeypatch.setattr(mod, "subprocess", rec)
    monkeypatch.setenv("DVLA_DATA_ROOT", str(tmp_path))

    plan = mod.build_pipeline_plan(
        mode="ray",
        profile="smoke",
        run_root=tmp_path / "run",
        python="python",
        launcher_cfg=_launcher_cfg(),
    )
    _write_complete_collected_pair(
        plan.reward_dir,
        plan.hidden_dir,
        "ray_shard_000.hdf5",
        [0, 1, 0, 1],
    )

    result = mod.collect_resume(
        plan,
        target_episodes=None,
        num_tasks=2,
        skip_collect=False,
    )
    out = capsys.readouterr().out

    assert result["ran_collect"] is False
    assert "target 4 already collected" in out
    assert rec.calls == []


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


def test_multi_gpu_torchrun_wrapping_by_mode(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    # no-Ray multi-GPU: BOTH collect and cotrain run under torchrun DDP. The collector
    # shards its work list by torchrun rank and binds gpu_id=local_rank.
    noray = build_pipeline_plan(mode="noray", run_root=tmp_path, python="python", ngpu=4)
    assert "torch.distributed.run" in noray.cotrain_cmd
    assert "--nproc-per-node=4" in noray.cotrain_cmd
    assert noray.cotrain_cmd.count("dreamervla.train") == 1
    assert "torch.distributed.run" in noray.collect_cmd
    assert "--nproc-per-node=4" in noray.collect_cmd
    assert noray.collect_cmd.count("dreamervla.train") == 1

    # Ray multi-GPU: cotrain uses torchrun DDP, but collection stays a Ray worker
    # fan-out (no torchrun); inference scales via collect.num_inference_workers.
    ray = build_pipeline_plan(mode="ray", run_root=tmp_path, python="python", ngpu=4)
    assert "torch.distributed.run" in ray.cotrain_cmd
    assert "--nproc-per-node=4" in ray.cotrain_cmd
    assert "torch.distributed.run" not in ray.collect_cmd


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
    assert "training.wm_warmup_steps=1200" in cotrain
    if profile == "release":
        assert "training.classifier_warmup_steps=1200" in cotrain
        assert "training.warmup_replay_epochs=1" in cotrain
    else:
        assert "training.classifier_warmup_steps=42" in cotrain
        assert "training.warmup_replay_epochs=0" in cotrain
        assert "training.warmup_checkpoint_every=200" in cotrain
        assert "training.classifier_batch_size=12" in cotrain
        assert "dataloader.batch_size=12" in cotrain
    assert not any(
        item.startswith("training.warmup_replay_max_steps=") for item in cotrain
    )


def test_smoke_profile_uses_fixed_step_warmup_not_replay_epoch() -> None:
    cfg = _launcher_cfg()
    cotrain = cfg["profiles"]["smoke"]["cotrain"]

    assert "training.wm_warmup_steps=1" in cotrain
    assert "training.classifier_warmup_steps=1" in cotrain
    assert "training.warmup_replay_epochs=0" in cotrain


def test_smoke_profile_replay_can_retain_one_episode_per_task() -> None:
    cfg = _launcher_cfg()
    smoke = cfg["profiles"]["smoke"]
    task_count = int(cfg["collect_num_tasks"])
    horizon = _override_int(smoke["collect"]["ray"], "collect.episode_horizon")
    buffer_size = _override_int(smoke["cotrain"], "online_rollout.buffer_size")

    assert buffer_size >= task_count * horizon


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


def test_async_cotrain_engine_splits_warmup_and_ray_online(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    plan = build_pipeline_plan(
        mode="ray", run_root=tmp_path, python="python", launcher_cfg=cfg
    )

    assert plan.cotrain_engine == "async"
    # 2a: sync pipeline warmup, online RL disabled (total_env_steps=0).
    assert "experiment=openvla_onetraj_libero_cotrain_noray" in plan.cotrain_warmup_cmd
    assert "online_rollout.total_env_steps=0" in plan.cotrain_warmup_cmd
    # 2c: target manual-cotrain ray online, NOT torchrun, init from the consolidated warmup ckpt.
    assert "experiment=openvla_onetraj_libero_cotrain_ray" in plan.cotrain_online_cmd
    assert "torch.distributed.run" not in plan.cotrain_online_cmd
    assert any(x.startswith("init.warmup_ckpt_path=") for x in plan.cotrain_online_cmd)
    assert "inference.init_ckpt.path=null" not in plan.cotrain_online_cmd
    assert plan.ray_init_ckpt is not None


def test_async_cotrain_online_command_targets_manual_cotrain_runner(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="smoke",
        ngpu=2,
        launcher_cfg=cfg,
    )

    assert "experiment=openvla_onetraj_libero_cotrain_ray" in plan.cotrain_online_cmd
    assert "manual_cotrain.ngpu=2" in plan.cotrain_online_cmd
    assert "+cluster.num_gpus=2" in plan.cotrain_online_cmd
    assert "manual_cotrain.envs_per_worker=8" in plan.cotrain_online_cmd
    assert "cluster.component_placement=null" in plan.cotrain_online_cmd


@pytest.mark.parametrize("ngpu", [0, 1, 2, 3, 4, 5])
def test_async_manual_cotrain_online_command_supports_zero_to_five_gpus(
    tmp_path,
    ngpu: int,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="smoke",
        ngpu=ngpu,
        launcher_cfg=cfg,
    )

    assert "experiment=openvla_onetraj_libero_cotrain_ray" in plan.cotrain_online_cmd
    assert f"manual_cotrain.ngpu={ngpu}" in plan.cotrain_online_cmd
    assert f"+cluster.num_gpus={ngpu}" in plan.cotrain_online_cmd
    assert "torch.distributed.run" not in plan.cotrain_online_cmd
    if ngpu == 0:
        assert "actor.train_cfg.fsdp.strategy=none" in plan.cotrain_online_cmd
    else:
        assert "actor.train_cfg.fsdp.strategy=none" not in plan.cotrain_online_cmd


def test_async_manual_cotrain_envs_per_worker_uses_profile_and_explicit_override(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"

    profile_plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path / "profile",
        python="python",
        profile="multi_gpu",
        ngpu=2,
        launcher_cfg=cfg,
    )
    assert "manual_cotrain.envs_per_worker=2" in profile_plan.cotrain_online_cmd
    assert "manual_cotrain.envs_per_worker=8" not in profile_plan.cotrain_online_cmd

    override_plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path / "override",
        python="python",
        profile="smoke",
        ngpu=2,
        launcher_cfg=cfg,
        cotrain_overrides=["manual_cotrain.envs_per_worker=5"],
    )
    assert "manual_cotrain.envs_per_worker=5" in override_plan.cotrain_online_cmd
    assert "manual_cotrain.envs_per_worker=8" not in override_plan.cotrain_online_cmd


def test_async_manual_cotrain_maps_online_budget_to_global_steps(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path / "budget",
        python="python",
        profile="release",
        ngpu=1,
        launcher_cfg=cfg,
    )

    # release profile: ceil(200000 / (envs_per_worker=8 * rollout_epoch=16 * 256)).
    assert "manual_cotrain.global_steps=7" in plan.cotrain_online_cmd

    override_plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path / "override",
        python="python",
        profile="release",
        ngpu=1,
        launcher_cfg=cfg,
        cotrain_overrides=["manual_cotrain.global_steps=3"],
    )
    assert "manual_cotrain.global_steps=3" in override_plan.cotrain_online_cmd
    assert "manual_cotrain.global_steps=7" not in override_plan.cotrain_online_cmd


def test_ngpu_zero_does_not_emit_torchrun_or_gpu_ray_placement(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="multi_gpu",
        ngpu=0,
        launcher_cfg={
            **_launcher_cfg(),
            "cotrain_engine": "async",
            "render_backend": "osmesa",
        },
    )

    assert "torch.distributed.run" not in plan.cotrain_cmd
    assert "torch.distributed.run" not in plan.cotrain_warmup_cmd
    assert _override_values(plan.collect_cmd, "collect.num_inference_workers") == ["1"]
    assert _override_values(plan.collect_cmd, "env.num_workers") == ["1"]
    assert "++inference.device=cpu" in plan.collect_cmd
    assert _override_values(plan.cotrain_cmd, "trainer.device") == ["cpu"]
    assert _override_values(plan.cotrain_cmd, "training.distributed_strategy") == [
        "ddp"
    ]
    assert _override_values(plan.cotrain_cmd, "online_rollout.num_envs") == ["1"]
    assert "cluster.component_placement.env=0" not in plan.cotrain_online_cmd
    assert "cluster.component_placement.rollout=0" not in plan.cotrain_online_cmd
    assert "cluster.component_placement.actor=0" not in plan.cotrain_online_cmd
    assert "cluster.component_placement=null" in plan.cotrain_online_cmd
    assert "+cluster.num_gpus=0" in plan.cotrain_online_cmd
    assert "inference.placement.strategy=node" in plan.cotrain_online_cmd
    assert "++inference.device=cpu" in plan.cotrain_online_cmd
    assert "learner.placement.strategy=node" in plan.cotrain_online_cmd
    assert "learner.train_cfg.device=cpu" in plan.cotrain_online_cmd
    assert "learner.train_cfg.precision=fp32" in plan.cotrain_online_cmd
    assert "env.num_workers=1" in plan.cotrain_online_cmd
    _assert_no_duplicate_override_keys(plan.cotrain_online_cmd)


def test_ngpu_zero_noray_collect_is_rejected(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    with pytest.raises(ValueError, match="mode=noray does not support ngpu=0"):
        build_pipeline_plan(
            mode="noray",
            run_root=tmp_path,
            python="python",
            profile="smoke",
            ngpu=0,
        )


def test_ngpu_zero_rejects_egl_async_online(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    with pytest.raises(
        ValueError,
        match="render_backend=egl requires ngpu>=1; use render_backend=osmesa for ngpu=0",
    ):
        build_pipeline_plan(
            mode="ray",
            run_root=tmp_path,
            python="python",
            profile="multi_gpu",
            ngpu=0,
            launcher_cfg={
                **_launcher_cfg(),
                "cotrain_engine": "async",
                "render_backend": "egl",
            },
        )


def test_ngpu_zero_rejects_egl_sync_cotrain(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    with pytest.raises(
        ValueError,
        match="render_backend=egl requires ngpu>=1; use render_backend=osmesa for ngpu=0",
    ):
        build_pipeline_plan(
            mode="ray",
            run_root=tmp_path,
            python="python",
            profile="multi_gpu",
            ngpu=0,
            launcher_cfg={**_launcher_cfg(), "render_backend": "egl"},
        )


def test_ngpu_zero_rejects_async_cotrain_render_backend_override_to_egl(
    tmp_path,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    with pytest.raises(
        ValueError,
        match="render_backend=egl requires ngpu>=1; use render_backend=osmesa for ngpu=0",
    ):
        build_pipeline_plan(
            mode="ray",
            run_root=tmp_path,
            python="python",
            profile="multi_gpu",
            ngpu=0,
            launcher_cfg={
                **_launcher_cfg(),
                "cotrain_engine": "async",
                "render_backend": "osmesa",
            },
            cotrain_overrides=["render_backend=egl"],
        )


def test_ngpu_zero_rejects_sync_online_rollout_backend_override_to_egl(
    tmp_path,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    with pytest.raises(
        ValueError,
        match="render_backend=egl requires ngpu>=1; use render_backend=osmesa for ngpu=0",
    ):
        build_pipeline_plan(
            mode="ray",
            run_root=tmp_path,
            python="python",
            profile="multi_gpu",
            ngpu=0,
            launcher_cfg={**_launcher_cfg(), "render_backend": "osmesa"},
            cotrain_overrides=["online_rollout.render_backend=egl"],
        )


def test_ngpu_zero_rejects_nested_component_placement_override(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    with pytest.raises(
        ValueError,
        match="cluster.component_placement.* overrides are not supported when ngpu=0",
    ):
        build_pipeline_plan(
            mode="ray",
            run_root=tmp_path,
            python="python",
            profile="multi_gpu",
            ngpu=0,
            launcher_cfg={
                **_launcher_cfg(),
                "cotrain_engine": "async",
                "render_backend": "osmesa",
            },
            cotrain_overrides=["cluster.component_placement.env=0"],
        )


def test_ngpu_zero_rejects_root_component_placement_override(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    with pytest.raises(
        ValueError,
        match="cluster.component_placement.* overrides are not supported when ngpu=0",
    ):
        build_pipeline_plan(
            mode="ray",
            run_root=tmp_path,
            python="python",
            profile="multi_gpu",
            ngpu=0,
            launcher_cfg={
                **_launcher_cfg(),
                "cotrain_engine": "async",
                "render_backend": "osmesa",
            },
            cotrain_overrides=["cluster.component_placement={env:0}"],
        )


def test_multi_gpu_profile_scales_async_ray_online_envs_with_ngpu(
    tmp_path,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        launcher_cfg=cfg,
        profile="multi_gpu",
        ngpu=6,
    )

    assert "env.num_workers=12" in plan.cotrain_online_cmd
    assert "env.envs_per_worker=12" not in plan.cotrain_online_cmd
    assert "render_backend=osmesa" in plan.cotrain_online_cmd
    assert "++env.cfg.egl_spawn_stagger_s=2.0" not in plan.cotrain_online_cmd
    assert "++env.cfg.egl_spawn_init_timeout_s=900" not in plan.cotrain_online_cmd
    assert "++env.cfg.egl_max_respawns=5" not in plan.cotrain_online_cmd


@pytest.mark.parametrize("ngpu", [1, 2, 3, 4, 5, 6])
def test_multi_gpu_profile_scales_async_ray_egl_slots_with_ngpu(
    tmp_path,
    ngpu: int,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    cfg["render_backend"] = "egl"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        launcher_cfg=cfg,
        profile="multi_gpu",
        ngpu=ngpu,
    )

    expected_env_slots = "0" if ngpu == 1 else f"0-{ngpu - 1}"
    expected_actor_slot = str(ngpu - 1)

    assert f"env.num_workers={ngpu}" in plan.cotrain_online_cmd
    assert "env.envs_per_worker=2" in plan.cotrain_online_cmd
    assert "env.envs_per_worker=12" not in plan.cotrain_online_cmd
    assert f"cluster.component_placement.env={expected_env_slots}" in plan.cotrain_online_cmd
    assert "cluster.component_placement.rollout=0" in plan.cotrain_online_cmd
    assert f"cluster.component_placement.actor={expected_actor_slot}" in plan.cotrain_online_cmd
    assert "cluster.component_placement=null" not in plan.cotrain_online_cmd
    assert "render_backend=egl" in plan.cotrain_online_cmd
    assert "++env.cfg.egl_spawn_stagger_s=2.0" in plan.cotrain_online_cmd
    assert "++env.cfg.egl_spawn_init_timeout_s=900" in plan.cotrain_online_cmd
    assert "++env.cfg.egl_max_respawns=5" not in plan.cotrain_online_cmd


def test_async_ray_egl_respects_explicit_online_worker_topology(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    cfg["render_backend"] = "egl"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        launcher_cfg=cfg,
        profile="multi_gpu",
        ngpu=6,
        cotrain_overrides=[
            "env.num_workers=3",
            "env.envs_per_worker=4",
            "cluster.component_placement.env=0-2",
            "cluster.component_placement.rollout=3",
            "cluster.component_placement.actor=5",
        ],
    )

    assert plan.cotrain_online_cmd.count("env.num_workers=3") == 1
    assert plan.cotrain_online_cmd.count("env.envs_per_worker=4") == 1
    assert "env.num_workers=6" not in plan.cotrain_online_cmd
    assert "env.envs_per_worker=2" not in plan.cotrain_online_cmd
    assert plan.cotrain_online_cmd.count("cluster.component_placement.env=0-2") == 1
    assert plan.cotrain_online_cmd.count("cluster.component_placement.rollout=3") == 1
    assert plan.cotrain_online_cmd.count("cluster.component_placement.actor=5") == 1


def test_sync_cotrain_phase_warmup_only_has_no_online_steps(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_phase"] = "warmup"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        launcher_cfg=cfg,
    )

    assert plan.cotrain_phase == "warmup"
    assert "online_rollout.total_env_steps=0" in plan.cotrain_warmup_cmd
    assert "training.resume=true" not in plan.cotrain_warmup_cmd


def test_sync_cotrain_phase_online_only_resumes_warmup_ckpts(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_phase"] = "online"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        launcher_cfg=cfg,
    )

    assert plan.cotrain_phase == "online"
    assert "training.resume=true" in plan.cotrain_online_cmd
    assert "online_rollout.total_env_steps=0" not in plan.cotrain_online_cmd


def test_async_cotrain_phase_online_only_consolidates_missing_ray_init_ckpt(
    tmp_path,
    monkeypatch,
) -> None:
    import torch

    import dreamervla.launchers.coldstart_warmup_cotrain as mod

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, cmd, **_kwargs):
            self.calls.append(list(cmd))

    rec = _Recorder()
    monkeypatch.setattr(mod, "subprocess", rec)

    run_root = tmp_path / "run"
    ckpt_dir = run_root / "cotrain" / "ckpt"
    ckpt_dir.mkdir(parents=True)
    torch.save({"world_model": {"wm": torch.ones(1)}}, ckpt_dir / "wm_warmup.ckpt")
    torch.save({"classifier": {"cls": torch.zeros(1)}}, ckpt_dir / "classifier_warmup.ckpt")

    exit_code = mod.main(
        [
            f"run_root={run_root}",
            f"data_root={tmp_path}",
            "cotrain_engine=async",
            "cotrain_phase=online",
            "skip_asset_check=false",
            "collect_num_tasks=1",
        ]
    )

    ray_init = ckpt_dir / "ray_async_init.ckpt"
    assert exit_code == 0
    assert ray_init.is_file()
    assert len(rec.calls) == 1
    assert f"init.warmup_ckpt_path={ray_init}" in rec.calls[0]


def test_validate_warmup_outputs_requires_split_ckpts(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import validate_warmup_outputs

    errors = validate_warmup_outputs(cotrain_out=tmp_path)
    assert "wm_warmup.ckpt" in "\n".join(errors)
    assert "classifier_warmup.ckpt" in "\n".join(errors)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "wm_warmup.ckpt").touch()
    (ckpt_dir / "classifier_warmup.ckpt").touch()

    assert validate_warmup_outputs(cotrain_out=tmp_path) == []


def test_async_cotrain_online_command_uses_valid_ray_hydra_keys(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        launcher_cfg=cfg,
        cotrain_overrides=["manual_cotrain.global_steps=4", "env.num_workers=1"],
    )
    overrides = [
        item
        for item in plan.cotrain_online_cmd
        if "=" in item and not item.startswith("-")
    ]

    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg_obj = compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg_obj)

    assert cfg_obj.init.warmup_ckpt_path == str(plan.ray_init_ckpt)
    assert cfg_obj.learner.init_ckpt.path == str(plan.ray_init_ckpt)


def test_sync_cotrain_engine_has_no_async_subcommands(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="noray", run_root=tmp_path, python="python")
    assert plan.cotrain_engine == "sync"
    assert plan.cotrain_phase == "all"
    assert plan.cotrain_warmup_cmd == []
    assert plan.cotrain_online_cmd == []


def test_consolidate_warmup_state_dicts_merges_components(tmp_path) -> None:
    import torch

    from dreamervla.launchers.coldstart_warmup_cotrain import _consolidate_warmup_state_dicts

    wm = tmp_path / "wm_warmup.ckpt"
    cls = tmp_path / "classifier_warmup.ckpt"
    out = tmp_path / "ray_async_init.ckpt"
    torch.save({"global_step": 3, "world_model": {"w": torch.zeros(2)}}, wm)
    torch.save({"classifier": {"b": torch.ones(1)}, "classifier_threshold": 0.5}, cls)

    _consolidate_warmup_state_dicts(wm, cls, out)

    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert set(payload["state_dicts"]) == {"world_model", "classifier"}
    assert set(payload["state_dicts"]["world_model"]) == {"w"}
    assert payload["classifier_threshold"] == 0.5
