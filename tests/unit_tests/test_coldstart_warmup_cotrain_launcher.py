from __future__ import annotations

import json
import os
import sys
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
        "render_backend": cfg["render_backend"],
    }


def _render(items, context: dict) -> list[str]:
    from dreamervla.launchers.coldstart_warmup_cotrain import _render_overrides

    return _render_overrides(items, context)


def _assert_items_in_command(items: list[str], command: list[str]) -> None:
    for item in items:
        assert item in command


def _override_int(overrides: list[str], key: str) -> int:
    matches = [item for item in overrides if _override_key(item) == key and "=" in item]
    assert len(matches) == 1
    return int(matches[0].split("=", 1)[1])


def _override_values(overrides: list[str], key: str) -> list[str]:
    return [
        item.split("=", 1)[1]
        for item in overrides
        if _override_key(item) == key and "=" in item
    ]


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


def _expected_ray_online_rollout_placement(ngpu: int, real_env_workers: int = 4) -> str:
    if ngpu <= 1:
        return "0"
    compute_gpus = list(range(min(real_env_workers, ngpu - 1), ngpu))
    if not compute_gpus:
        compute_gpus = [ngpu - 1]
    if len(compute_gpus) >= 3 and ngpu > 1:
        actor_gpu = compute_gpus[-1]
        worker_count = max(0, ngpu - 1)
        non_actor_gpus = compute_gpus[:-1]
    else:
        actor_gpu = None
        worker_count = ngpu
        non_actor_gpus = compute_gpus
    segments: list[str] = []
    next_rank = 0
    base, extra = divmod(worker_count, len(non_actor_gpus))
    for idx, gpu in enumerate(non_actor_gpus):
        count = base + (1 if idx < extra else 0)
        if count <= 0:
            continue
        end = next_rank + count - 1
        processes = str(next_rank) if next_rank == end else f"{next_rank}-{end}"
        segments.append(f"{gpu}:{processes}")
        next_rank = end + 1
    if actor_gpu is not None:
        segments.append(f"{actor_gpu}:{next_rank}")
    return ",".join(segments)


def _hydra_string_value(value: str) -> str:
    return f"'{value}'" if "," in value else value


def _write_complete_collected_pair(
    reward,
    hidden,
    shard_name: str,
    task_ids: list[int],
    *,
    obs_hidden_source: str = "hidden_token",
    token_count: int = 256,
    token_dim: int = 4096,
) -> None:
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
            hdemo.create_dataset(
                "obs_embedding",
                shape=(1, token_count, token_dim),
                dtype=np.float16,
                fillvalue=0,
            )
            hdemo.attrs["complete"] = True
        rdata.attrs["num_demos"] = len(task_ids)
        hdata.attrs["num_demos"] = len(task_ids)
    (hidden / "preprocess_config.json").write_text(
        json.dumps(
            {
                "action_head_type": "oft_discrete_token",
                "obs_hidden_source": obs_hidden_source,
                "hidden_key": "obs_embedding",
                "token_count": token_count,
                "token_dim": token_dim,
                "hidden_dim": token_count * token_dim,
                "obs_embedding_shape": [token_count, token_dim],
                "hidden_storage_format": "tokenized",
                "num_images_in_input": 1,
                "patches_per_image": token_count,
                "history": 1,
                "include_state": False,
                "sidecar_schema_version": 1,
                "required_demo_datasets": ["obs_embedding"],
            }
        ),
        encoding="utf-8",
    )


def test_ray_launcher_plan_wires_coldstart_outputs_into_cotrain_warmup(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="ray", run_root=tmp_path, python="python", profile="smoke")
    cfg = _launcher_cfg()
    context = _plan_context(plan, cfg)

    assert plan.mode == "ray"
    assert f"task.openvla_oft.hdf5_reward_dir={plan.reward_dir}" in plan.collect_cmd
    assert f"task.openvla_oft.hidden_token_dir={plan.hidden_dir}" in plan.collect_cmd
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
    assert f"task.openvla_oft.hidden_token_dir={plan.hidden_dir}" in plan.collect_cmd
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
        "training.warmup_topk_k",
        "training.wm_profile_steps",
        "training.classifier_batch_size",
        "dataloader.batch_size",
        "optim.world_model.lr",
        "offline_warmup.infer_task_id_from_shard",
        "online_rollout.buffer_size",
        "online_rollout.sequence_length",
        "online_rollout.total_env_steps",
        "world_model.chunk_rollout_chunks",
        "world_model.chunk_rollout_loss_scale",
        "world_model.proprio_reconstruction_loss_scale",
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


def test_multi_gpu_profile_aligns_world_model_full_dataset_recipe() -> None:
    cfg = _launcher_cfg()
    cotrain = cfg["profiles"]["multi_gpu"]["cotrain"]

    expected = {
        "training.wm_warmup_steps=20000",
        "training.warmup_replay_epochs=10",
        "training.warmup_checkpoint_every=0",
        "training.warmup_topk_k=0",
        "training.wm_profile_steps=0",
        "dataloader.batch_size=16",
        "optim.world_model.lr=3.0e-5",
        "online_rollout.buffer_size=160000",
        "online_rollout.sequence_length=36",
        "offline_warmup.infer_task_id_from_shard=true",
        "world_model.chunk_rollout_chunks=4",
        "world_model.chunk_rollout_loss_scale=0.2",
        "world_model.proprio_reconstruction_loss_scale=0.0",
    }

    assert expected <= set(cotrain)


def test_multi_gpu_cotrain_warmup_command_composes_with_hydra(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    launcher_cfg = _launcher_cfg()
    launcher_cfg["cotrain_engine"] = "async"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="multi_gpu",
        ngpu=8,
        launcher_cfg=launcher_cfg,
    )
    train_module_at = len(plan.cotrain_warmup_cmd) - 1 - list(
        reversed(plan.cotrain_warmup_cmd)
    ).index("dreamervla.train")
    overrides = plan.cotrain_warmup_cmd[train_module_at + 1 :]
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)

    assert cfg.offline_warmup.infer_task_id_from_shard is True
    assert cfg.online_rollout.sequence_length == 36
    assert cfg.world_model.chunk_rollout_chunks == 4


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
    topk_k = 2
    wm_profile_steps = 3
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
            f"warmup.topk_k={topk_k}",
            f"warmup.wm_profile_steps={wm_profile_steps}",
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
    assert f"training.warmup_topk_k={topk_k}" in out
    assert f"training.wm_profile_steps={wm_profile_steps}" in out
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
            if len(self.calls) == 1:
                _write_complete_collected_pair(
                    tmp_path / "collected_rollouts/libero_goal/reward",
                    tmp_path / "collected_rollouts/libero_goal/hidden",
                    "ray_shard_000.hdf5",
                    [0],
                )

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
            if len(self.calls) == 1:
                _write_complete_collected_pair(
                    tmp_path / "collected_rollouts/libero_goal/reward",
                    tmp_path / "collected_rollouts/libero_goal/hidden",
                    "ray_shard_000.hdf5",
                    [0],
                )

    rec = _Recorder()
    monkeypatch.setattr(mod, "subprocess", rec)
    _write_complete_collected_pair(
        tmp_path / "collected_rollouts/libero_goal/reward",
        tmp_path / "collected_rollouts/libero_goal/hidden",
        "ray_shard_000.hdf5",
        [0],
    )

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
    _write_complete_collected_pair(
        tmp_path / "collected_rollouts/libero_goal/reward",
        tmp_path / "collected_rollouts/libero_goal/hidden",
        "ray_shard_000.hdf5",
        [0],
    )

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
        rf.attrs["complete"] = True
        hf.attrs["complete"] = True
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
            hdata.create_group(f"demo_{idx}").create_dataset(
                "obs_embedding",
                shape=(1, 256, 4096),
                dtype="float16",
                fillvalue=0,
            )
        rdata.attrs["num_demos"] = 2
        hdata.attrs["num_demos"] = 2
    (hidden / "preprocess_config.json").write_text(
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
    _write_complete_collected_pair(
        reward_dir,
        hidden_dir,
        "shard_000.hdf5",
        [0],
    )
    config_path = hidden_dir / "preprocess_config.json"
    config_text = config_path.read_text(encoding="utf-8")
    config_path.unlink()

    errors = validate_collected_outputs(reward_dir=reward_dir, hidden_dir=hidden_dir)

    assert any("preprocess_config.json" in error for error in errors)

    config_path.write_text(config_text, encoding="utf-8")

    assert validate_collected_outputs(reward_dir=reward_dir, hidden_dir=hidden_dir) == []


def test_reused_coldstart_output_validation_rejects_56_token_sidecar(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import validate_collected_outputs

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    _write_complete_collected_pair(
        reward_dir,
        hidden_dir,
        "shard_000.hdf5",
        [0],
        obs_hidden_source="hidden_token",
        token_count=56,
    )

    errors = validate_collected_outputs(reward_dir=reward_dir, hidden_dir=hidden_dir)

    assert not any("obs_hidden_source" in error for error in errors)
    assert any("token_count" in error for error in errors)
    assert any("obs_embedding_shape" in error for error in errors)


def test_collect_resume_rejects_count_complete_56_token_collection(
    tmp_path,
    monkeypatch,
) -> None:
    import dreamervla.launchers.coldstart_warmup_cotrain as mod

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
        [0, 1],
        obs_hidden_source="hidden_token",
        token_count=56,
    )

    with pytest.raises(ValueError, match="hidden-token schema"):
        mod.collect_resume(
            plan,
            target_episodes=2,
            num_tasks=2,
            skip_collect=False,
        )


def test_collected_validation_rejects_56x1024_in_later_shard(tmp_path) -> None:
    import h5py

    from dreamervla.launchers.coldstart_warmup_cotrain import validate_collected_outputs

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    _write_complete_collected_pair(
        reward_dir, hidden_dir, "shard_000.hdf5", [0]
    )
    _write_complete_collected_pair(
        reward_dir, hidden_dir, "shard_001.hdf5", [1]
    )
    with h5py.File(hidden_dir / "shard_001.hdf5", "r+") as handle:
        demo = handle["data/demo_0"]
        del demo["obs_embedding"]
        demo.create_dataset(
            "obs_embedding",
            shape=(1, 56, 1024),
            dtype="float16",
            fillvalue=0,
        )

    errors = validate_collected_outputs(
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
    )

    assert any("shard_001.hdf5" in error for error in errors)
    assert any("(1, 56, 1024)" in error for error in errors)


def test_collected_validation_rejects_56x1024_in_later_demo(tmp_path) -> None:
    import h5py

    from dreamervla.launchers.coldstart_warmup_cotrain import validate_collected_outputs

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    _write_complete_collected_pair(
        reward_dir, hidden_dir, "shard_000.hdf5", [0, 1]
    )
    with h5py.File(hidden_dir / "shard_000.hdf5", "r+") as handle:
        demo = handle["data/demo_1"]
        del demo["obs_embedding"]
        demo.create_dataset(
            "obs_embedding",
            shape=(1, 56, 1024),
            dtype="float16",
            fillvalue=0,
        )

    errors = validate_collected_outputs(
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
    )

    assert any("data/demo_1/obs_embedding" in error for error in errors)
    assert any("(1, 56, 1024)" in error for error in errors)


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
    assert "profile=multi_gpu" in ray_text
    # ngpu is derived from CUDA_VISIBLE_DEVICES (so 8-GPU runs without a manual
    # override) with a 6-GPU fallback when the variable is unset.
    assert 'ngpu="${_DVLA_NGPU}"' in ray_text
    assert "CUDA_VISIBLE_DEVICES" in ray_text
    assert "_DVLA_NGPU=6" in ray_text
    assert "cotrain_engine=async" in ray_text
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


@pytest.mark.parametrize(
    ("ngpu", "expected_inference_workers", "expected_real_env_workers"),
    [(2, "2", "1"), (4, "4", "3"), (8, "4", "4")],
)
def test_multi_gpu_profile_caps_env_worker_counts_at_ngpu(
    tmp_path,
    ngpu: int,
    expected_inference_workers: str,
    expected_real_env_workers: str,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    cfg["render_backend"] = "osmesa"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="multi_gpu",
        ngpu=ngpu,
        launcher_cfg=cfg,
    )

    # collect inference workers cap at min(4, ngpu): ngpu=2 -> 2, ngpu>=4 -> 4.
    assert _override_values(plan.collect_cmd, "collect.num_inference_workers") == [
        expected_inference_workers
    ]
    # online real-env workers cap at min(4, ngpu-1) so a wm_env worker always keeps a
    # GPU: ngpu=2 -> 1, ngpu=4 -> 3, ngpu>=5 -> 4 (unchanged from the profile 4).
    assert (
        f"manual_cotrain.real_env_workers={expected_real_env_workers}"
        in plan.cotrain_online_cmd
    )


@pytest.mark.parametrize("ngpu", [2, 3, 4, 5, 6, 8])
def test_multi_gpu_profile_reserves_a_wm_env_gpu(tmp_path, ngpu: int) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    cfg["render_backend"] = "osmesa"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="multi_gpu",
        ngpu=ngpu,
        launcher_cfg=cfg,
    )

    workers = _override_int(plan.cotrain_online_cmd, "manual_cotrain.real_env_workers")
    # ngpu>=2 always leaves at least one GPU for a wm_env worker.
    assert 1 <= workers <= ngpu - 1
    # ngpu>=5 is unchanged from the profile-declared 4 (6/8-GPU mainline byte-identical).
    if ngpu >= 5:
        assert workers == 4


@pytest.mark.parametrize(
    ("ngpu", "expected_real_env_workers", "expected_envs_per_worker"),
    [(2, 1, 16), (3, 2, 16), (4, 3, 8), (8, 4, 8)],
)
def test_multi_gpu_profile_caps_real_envs_per_worker_for_guard(
    tmp_path,
    ngpu: int,
    expected_real_env_workers: int,
    expected_envs_per_worker: int,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    cfg["render_backend"] = "osmesa"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="multi_gpu",
        ngpu=ngpu,
        launcher_cfg=cfg,
    )

    target = 32  # ray_online_real_rollout_target_trajectories in the multi_gpu profile
    workers = _override_int(plan.cotrain_online_cmd, "manual_cotrain.real_env_workers")
    envs = _override_int(plan.cotrain_online_cmd, "manual_cotrain.envs_per_worker")
    assert workers == expected_real_env_workers
    assert envs == expected_envs_per_worker
    # Both runner rollout-distribution guard invariants must hold at every GPU count.
    assert target % envs == 0
    assert target // envs >= workers


def test_async_online_guard_rejects_real_env_workers_leaving_no_wm_gpu(
    tmp_path,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    cfg["render_backend"] = "osmesa"
    with pytest.raises(ValueError, match="need ngpu > real_env_workers"):
        build_pipeline_plan(
            mode="ray",
            run_root=tmp_path,
            python="python",
            profile="multi_gpu",
            ngpu=4,
            launcher_cfg=cfg,
            cotrain_overrides=["manual_cotrain.real_env_workers=4"],
        )


@pytest.mark.parametrize("profile", ["release", "multi_gpu"])
def test_full_profiles_run_full_pipeline_by_default(profile) -> None:
    cfg = _launcher_cfg()
    cotrain = cfg["profiles"][profile]["cotrain"]

    assert "training.debug=false" in cotrain
    assert "online_rollout.total_env_steps=200000" in cotrain
    if profile == "release":
        assert "training.wm_warmup_steps=1200" in cotrain
        assert "training.classifier_warmup_steps=1200" in cotrain
        assert "training.warmup_replay_epochs=1" in cotrain
    else:
        assert "training.wm_warmup_steps=20000" in cotrain
        assert "training.classifier_warmup_steps=42" in cotrain
        assert "training.warmup_replay_epochs=10" in cotrain
        assert "training.warmup_checkpoint_every=0" in cotrain
        assert "training.warmup_topk_k=0" in cotrain
        assert "training.wm_profile_steps=0" in cotrain
        assert "training.classifier_batch_size=16" in cotrain
        assert "dataloader.batch_size=16" in cotrain
        assert "optim.world_model.lr=3.0e-5" in cotrain
        assert "online_rollout.sequence_length=36" in cotrain
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


def test_launcher_debug_control_covers_collection_warmup_and_async_online(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan_default = build_pipeline_plan(mode="ray", run_root=tmp_path, python="python")
    assert "training.debug=true" not in plan_default.cotrain_cmd

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    plan_debug = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        launcher_cfg=cfg,
        debug=True,
        profile="multi_gpu",
        ngpu=6,
    )
    assert "training.debug=true" in plan_debug.cotrain_cmd
    assert _override_values(plan_debug.collect_cmd, "collect.episodes_per_task")[-1] == "1"
    assert _override_values(plan_debug.collect_cmd, "collect.episode_horizon")[-1] == "16"
    assert _override_values(plan_debug.collect_cmd, "env.num_workers")[-1] == "2"
    assert _override_values(plan_debug.cotrain_cmd, "training.wm_warmup_steps")[-1] == "1"
    assert _override_values(plan_debug.cotrain_cmd, "training.classifier_warmup_steps")[-1] == "1"
    assert _override_values(plan_debug.cotrain_cmd, "online_rollout.num_envs")[-1] == "1"
    assert _override_values(plan_debug.cotrain_cmd, "online_rollout.total_env_steps")[-1] == "0"
    assert _override_values(plan_debug.cotrain_online_cmd, "env.num_workers")[-1] == "12"
    assert _override_values(plan_debug.cotrain_online_cmd, "manual_cotrain.global_steps")[-1] == "1"
    assert _override_values(plan_debug.cotrain_online_cmd, "manual_cotrain.real_env_workers")[-1] == "4"
    assert _override_values(plan_debug.cotrain_online_cmd, "manual_cotrain.max_steps_per_rollout_epoch")[-1] == "64"
    assert _override_values(plan_debug.cotrain_online_cmd, "manual_cotrain.envs_per_worker")[-1] == "4"
    assert _override_values(plan_debug.cotrain_online_cmd, "manual_cotrain.wm_envs_per_worker")[-1] == "4"
    assert _override_values(plan_debug.cotrain_online_cmd, "manual_cotrain.real_rollout_target_trajectories")[-1] == "16"
    assert _override_values(plan_debug.cotrain_online_cmd, "manual_cotrain.wm_rollout_target_trajectories")[-1] == "48"
    assert _override_values(plan_debug.cotrain_online_cmd, "actor.train_cfg.global_batch_size")[-1] == "384"
    assert _override_values(plan_debug.cotrain_online_cmd, "actor.train_cfg.micro_batch_size")[-1] == "32"


def test_launcher_debug_control_keeps_async_env_width_adjustable(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        launcher_cfg=cfg,
        debug=True,
        profile="multi_gpu",
        ngpu=6,
        cotrain_overrides=[
            "manual_cotrain.real_env_workers=3",
            "manual_cotrain.envs_per_worker=6",
            "manual_cotrain.wm_envs_per_worker=6",
        ],
    )

    assert _override_values(plan.cotrain_online_cmd, "manual_cotrain.real_env_workers")[-1] == "3"
    assert _override_values(plan.cotrain_online_cmd, "manual_cotrain.envs_per_worker")[-1] == "6"
    assert _override_values(plan.cotrain_online_cmd, "manual_cotrain.wm_envs_per_worker")[-1] == "6"
    assert _override_values(plan.cotrain_online_cmd, "manual_cotrain.real_rollout_target_trajectories")[-1] == "18"
    assert _override_values(plan.cotrain_online_cmd, "manual_cotrain.wm_rollout_target_trajectories")[-1] == "48"


def test_launcher_debug_control_preserves_explicit_overrides(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        launcher_cfg=cfg,
        debug=True,
        profile="multi_gpu",
        ngpu=2,
        collect_overrides=["collect.episodes_per_task=3"],
        cotrain_overrides=[
            "online_rollout.total_env_steps=64",
            "manual_cotrain.global_steps=4",
        ],
    )

    assert _override_values(plan.collect_cmd, "collect.episodes_per_task")[-1] == "3"
    assert _override_values(plan.cotrain_cmd, "online_rollout.total_env_steps")[-1] == "64"
    assert _override_values(plan.cotrain_online_cmd, "manual_cotrain.global_steps")[-1] == "4"


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
    assert "cluster.component_placement.env=0-1" not in plan.cotrain_online_cmd


@pytest.mark.parametrize("ngpu", [0, 1, 2, 3, 4, 5])
def test_async_manual_cotrain_online_command_supports_zero_to_five_gpus(
    tmp_path,
    ngpu: int,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    if ngpu == 0:
        cfg["render_backend"] = "osmesa"
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
    # ngpu=2 -> 1 real_env_worker, target=32, profile width=16 -> capped to 16.
    assert "manual_cotrain.envs_per_worker=16" in profile_plan.cotrain_online_cmd
    assert "manual_cotrain.envs_per_worker=1" not in profile_plan.cotrain_online_cmd

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
    cfg["render_backend"] = "osmesa"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path / "budget",
        python="python",
        profile="release",
        ngpu=1,
        launcher_cfg=cfg,
    )

    # release profile: ceil(200000 / (wm_target=128 * rollout_horizon=512)).
    assert "manual_cotrain.global_steps=4" in plan.cotrain_online_cmd

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
    assert "manual_cotrain.global_steps=4" not in override_plan.cotrain_online_cmd


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
    cfg["render_backend"] = "osmesa"
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


@pytest.mark.parametrize("ngpu", [1, 2, 3, 4, 5, 6, 7])
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
    expected_rollout_slots = _hydra_string_value(
        _expected_ray_online_rollout_placement(ngpu)
    )

    assert f"env.num_workers={ngpu}" in plan.cotrain_online_cmd
    assert "env.envs_per_worker=16" in plan.cotrain_online_cmd
    assert "env.envs_per_worker=1" not in plan.cotrain_online_cmd
    assert f"cluster.component_placement.env={expected_env_slots}" in plan.cotrain_online_cmd
    assert f"cluster.component_placement.rollout={expected_rollout_slots}" in plan.cotrain_online_cmd
    assert f"cluster.component_placement.actor={expected_actor_slot}" in plan.cotrain_online_cmd
    assert "cluster.component_placement=null" not in plan.cotrain_online_cmd
    assert "render_backend=egl" in plan.cotrain_online_cmd
    assert "env.cfg.render_backend=egl" in plan.cotrain_online_cmd
    # real-env workers cap at min(4, ngpu-1) so a wm_env worker always keeps a GPU.
    expected_real_env_workers = min(4, max(1, ngpu - 1))
    assert (
        f"manual_cotrain.real_env_workers={expected_real_env_workers}"
        in plan.cotrain_online_cmd
    )
    # envs_per_worker: profile width 16, capped to the largest divisor of target=32
    # that keeps target//envs >= real_env_workers.
    expected_envs = max(
        d
        for d in (1, 2, 4, 8, 16, 32)
        if d <= min(16, 32 // expected_real_env_workers)
    )
    assert (
        f"manual_cotrain.envs_per_worker={expected_envs}" in plan.cotrain_online_cmd
    )
    assert 32 % expected_envs == 0
    assert 32 // expected_envs >= expected_real_env_workers
    assert not any(
        item.startswith("manual_cotrain.real_render_backend=")
        for item in plan.cotrain_online_cmd
    )
    assert "manual_cotrain.wm_envs_per_worker=16" in plan.cotrain_online_cmd


def test_multi_gpu_profile_limits_rollout_pressure_on_actor_gpu(tmp_path) -> None:
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
        ngpu=7,
    )

    assert "cluster.component_placement.rollout='4:0-2,5:3-5,6:6'" in (
        plan.cotrain_online_cmd
    )
    assert (
        "manual_cotrain.real_rollout_target_trajectories=32"
        in plan.cotrain_online_cmd
    )
    assert (
        "manual_cotrain.wm_rollout_target_trajectories=1024"
        in plan.cotrain_online_cmd
    )
    assert "manual_cotrain.max_steps_per_rollout_epoch=512" in plan.cotrain_online_cmd
    assert "++env.cfg.egl_spawn_stagger_s=2.0" not in plan.cotrain_online_cmd
    assert "++env.cfg.egl_spawn_init_timeout_s=900" not in plan.cotrain_online_cmd
    assert "manual_cotrain.env_rollout_timeout_s=5400" in plan.cotrain_online_cmd
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


def test_async_online_debug_override_is_struct_safe(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"

    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        launcher_cfg=cfg,
        debug=True,
    )

    assert "++training.debug=true" in plan.cotrain_online_cmd
    assert "training.debug=true" not in plan.cotrain_online_cmd


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
    _write_complete_collected_pair(
        tmp_path / "collected_rollouts/libero_goal/reward",
        tmp_path / "collected_rollouts/libero_goal/hidden",
        "ray_shard_000.hdf5",
        [0],
    )

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


def test_launcher_main_restores_dvla_data_root_after_inline_call(
    tmp_path,
    monkeypatch,
) -> None:
    import torch

    import dreamervla.launchers.coldstart_warmup_cotrain as mod

    class _Recorder:
        def run(self, *_args, **_kwargs):
            return None

    original_data_root = tmp_path / "original_data"
    monkeypatch.setenv("DVLA_DATA_ROOT", str(original_data_root))
    monkeypatch.setattr(mod, "subprocess", _Recorder())

    run_root = tmp_path / "run"
    ckpt_dir = run_root / "cotrain" / "ckpt"
    ckpt_dir.mkdir(parents=True)
    torch.save({"world_model": {"wm": torch.ones(1)}}, ckpt_dir / "wm_warmup.ckpt")
    torch.save({"classifier": {"cls": torch.zeros(1)}}, ckpt_dir / "classifier_warmup.ckpt")
    inline_data = tmp_path / "inline_data"
    _write_complete_collected_pair(
        inline_data / "collected_rollouts/libero_goal/reward",
        inline_data / "collected_rollouts/libero_goal/hidden",
        "ray_shard_000.hdf5",
        [0],
    )

    exit_code = mod.main(
        [
            f"run_root={run_root}",
            f"data_root={inline_data}",
            "cotrain_engine=async",
            "cotrain_phase=online",
            "skip_asset_check=false",
            "collect_num_tasks=1",
        ]
    )

    assert exit_code == 0
    assert os.environ["DVLA_DATA_ROOT"] == str(original_data_root)


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
    cfg["render_backend"] = "egl"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="multi_gpu",
        ngpu=2,
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
    assert cfg_obj.env.real.cfg.target == cfg_obj.env.cfg.target
    assert cfg_obj.env.real.cfg.render_backend == "egl"


def test_async_eval_runs_segmented_post_step_libero_eval_and_writes_trend_summary(
    tmp_path,
    monkeypatch,
) -> None:
    import dreamervla.launchers.coldstart_warmup_cotrain as mod

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    cfg["eval"]["enabled"] = True
    cfg["eval"]["interval_global_steps"] = 1
    plan = mod.build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="multi_gpu",
        ngpu=2,
        launcher_cfg=cfg,
        cotrain_overrides=["manual_cotrain.global_steps=2"],
    )

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []
            self.envs: list[dict[str, str] | None] = []
            self.eval_rates = [0.25, 0.35]

        def run(self, cmd, check, **kwargs):
            assert check is True
            cmd = list(cmd)
            self.calls.append(cmd)
            self.envs.append(kwargs.get("env"))
            if "dreamervla.train" in cmd:
                target = _override_int(cmd, "manual_cotrain.global_steps")
                ckpt_dir = (
                    tmp_path
                    / "cotrain"
                    / "checkpoints"
                    / f"manual_cotrain_step_{target}"
                )
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                (ckpt_dir / "manual_cotrain.ckpt").touch()
                return
            out_dir = next(
                item.split("=", 1)[1] for item in cmd if item.startswith("out_dir=")
            )
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            rate = self.eval_rates.pop(0)
            (Path(out_dir) / "eval_libero_metrics.json").write_text(
                json.dumps({"eval_success_rate": rate}),
                encoding="utf-8",
            )

    rec = _Recorder()
    monkeypatch.setattr(mod, "subprocess", rec)

    mod.run_async_online_with_in_run_eval_metrics(plan)

    train_calls = [call for call in rec.calls if "dreamervla.train" in call]
    eval_calls = [call for call in rec.calls if "dreamervla.launchers.train" in call]
    eval_envs = [
        env
        for call, env in zip(rec.calls, rec.envs, strict=True)
        if "dreamervla.launchers.train" in call
    ]
    assert [rec.calls.index(call) for call in train_calls] == [0, 2]
    assert [rec.calls.index(call) for call in eval_calls] == [1, 3]
    assert [_override_int(call, "manual_cotrain.global_steps") for call in train_calls] == [1, 2]
    assert all(_override_int(call, "manual_cotrain.checkpoint_every") == 1 for call in train_calls)
    assert len(eval_calls) == 2
    assert all(env is not None for env in eval_envs)
    assert all(env["MUJOCO_GL"] == "osmesa" for env in eval_envs if env is not None)
    assert all(
        env["PYOPENGL_PLATFORM"] == "osmesa"
        for env in eval_envs
        if env is not None
    )
    assert all(
        "MUJOCO_EGL_DEVICE_ID" not in env
        for env in eval_envs
        if env is not None
    )
    assert all("experiment=eval_libero_vla" in call for call in eval_calls)
    assert all("eval.ckpt_kind=dreamer" in call for call in eval_calls)
    assert all("eval.action_postprocess=openvla_oft" in call for call in eval_calls)
    assert any(
        item.endswith("manual_cotrain_step_1/manual_cotrain.ckpt")
        for item in eval_calls[0]
    )
    assert "+manual_cotrain.resume_ckpt" not in " ".join(train_calls[0])
    assert any(
        item.startswith("+manual_cotrain.resume_ckpt=")
        and item.endswith("manual_cotrain_step_1/manual_cotrain.ckpt")
        for item in train_calls[1]
    )
    summary = json.loads(
        (tmp_path / "cotrain" / "eval" / "eval_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert [record["global_step"] for record in summary["records"]] == [1, 2]
    assert [record["eval_success_rate"] for record in summary["records"]] == [0.25, 0.35]
    assert [record["eval_success_rate_trend"] for record in summary["records"]] == [0.25, 0.35]
    assert [record["eval_success_rate_delta"] for record in summary["records"]] == [0.0, 0.10]
    assert summary["records"][-1]["eval_best_success_rate"] == 0.35
    assert summary["records"][-1]["eval_success_rate_drop"] == 0.0
    assert summary["records"][-1]["eval_significant_drop"] == 0.0


def test_debug_async_cotrain_enables_post_step_eval_each_global_step(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"
    cfg["debug"] = True
    cfg["eval"]["enabled"] = False
    cfg["eval"]["debug_interval_global_steps"] = 1

    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="multi_gpu",
        ngpu=2,
        launcher_cfg=cfg,
        cotrain_overrides=["manual_cotrain.global_steps=2"],
    )

    assert plan.eval_enabled is True
    assert plan.eval_interval_global_steps == 1


def test_post_step_eval_egl_device_defaults_to_last_eval_gpu_for_split_render() -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import (
        _post_step_eval_egl_device_id,
    )

    assert _post_step_eval_egl_device_id({"gpus": "7,0"}) == "0"
    assert _post_step_eval_egl_device_id({"gpus": "7,0", "egl_device_id": 3}) == "3"


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


def test_multi_gpu_profile_does_not_force_async_real_rollout_actor_alignment(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _launcher_cfg()
    cfg["cotrain_engine"] = "async"

    profile_plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path / "profile",
        python="python",
        profile="multi_gpu",
        ngpu=6,
        launcher_cfg=cfg,
    )
    # RealEnv trajectories feed replay/learner, not ActorGroup, so actor group_size
    # alignment must be enforced only on WMEnv trajectories.
    # ngpu=6 -> 4 real_env_workers, target=32, profile width=16 -> capped to 8.
    assert "manual_cotrain.envs_per_worker=8" in profile_plan.cotrain_online_cmd
    assert "manual_cotrain.real_env_workers=4" in profile_plan.cotrain_online_cmd
    assert not any(
        item.startswith("manual_cotrain.real_render_backend=")
        for item in profile_plan.cotrain_online_cmd
    )
    assert (
        "manual_cotrain.real_rollout_target_trajectories=32"
        in profile_plan.cotrain_online_cmd
    )
    assert (
        "manual_cotrain.wm_rollout_target_trajectories=1024"
        in profile_plan.cotrain_online_cmd
    )
    assert not any(
        item.startswith("manual_cotrain.real_rollout_epoch=")
        for item in profile_plan.cotrain_online_cmd
    )

    override_plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path / "override",
        python="python",
        profile="multi_gpu",
        ngpu=6,
        launcher_cfg=cfg,
        cotrain_overrides=[
            "manual_cotrain.envs_per_worker=8",
            "manual_cotrain.real_rollout_epoch=1",
            "manual_cotrain.wm_rollout_target_trajectories=512",
        ],
    )
    assert "manual_cotrain.real_rollout_epoch=1" in override_plan.cotrain_online_cmd
    assert (
        "manual_cotrain.wm_rollout_target_trajectories=512"
        in override_plan.cotrain_online_cmd
    )
    assert (
        "manual_cotrain.wm_rollout_target_trajectories=1024"
        not in override_plan.cotrain_online_cmd
    )

    release_plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path / "release",
        python="python",
        profile="release",
        ngpu=6,
        launcher_cfg=cfg,
    )
    assert not any(
        item.startswith("manual_cotrain.real_rollout_epoch=")
        for item in release_plan.cotrain_online_cmd
    )
