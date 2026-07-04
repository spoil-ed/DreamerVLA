# Step 3b Report: tiny manual cotrain 5-step smoke

## 本步目标

让 `experiment=manual_cotrain_ray_tiny` 在 CPU/Ray tiny 路径下跑满
`manual_cotrain.global_steps=5`，并确认 run root 有可检查产物。

## 改动文件

- `configs/experiment/manual_cotrain_ray_tiny.yaml`：在 tiny real env cfg 中显式设置
  `emit_actor_trajectories: true`，让无 WMEnvGroup 的 tiny route 可以给 ActorGroup 提供训练 shard。
- `dreamervla/workers/env/trajectory_env_worker.py`：新增 real/env 通用的
  `emit_actor_trajectories` 显式 opt-in；默认仍保持 `real_env=false`、`wm_env=true`。
- `dreamervla/runners/manual_cotrain_ray_runner.py`：在当前在途 manual cotrain role-key worktree 上，
  修正 zero-GPU env placement 读取空 `gpu_ids` 的问题，并让 real-only metrics 使用 `real_env` actor channel key。
- `tests/unit_tests/test_manual_cotrain_ray_runner.py`：增加 zero-GPU group build、real-only actor key 推断、
  tiny compose 的回归断言。
- `tests/unit_tests/test_trajectory_env_worker.py`：增加 real env actor shard opt-in 回归测试；把 EGL worker 测试对齐到当前 in-process helper 契约。
- `logs/loop_progress.md`：Step 3b 标为 DONE。

## 验证命令与输出摘要

- RED: `conda run -n dreamervla python -m pytest tests/unit_tests/test_manual_cotrain_ray_runner.py::test_runner_launches_zero_gpu_tiny_groups_on_node_placement -q`
  - 失败符合预期：`IndexError: list index out of range`，来源是 zero-GPU `spec.gpu_ids[0]`。
- RED: `conda run -n dreamervla python -m pytest tests/unit_tests/test_manual_cotrain_ray_runner.py::test_actor_shard_role_counts_use_real_env_key_without_wm_group -q`
  - 失败符合预期：返回 `("default", 2)` 而不是 `("real_env", 2)`。
- RED: `conda run -n dreamervla python -m pytest tests/unit_tests/test_trajectory_env_worker.py::test_real_env_interact_can_emit_actor_trajectory_when_configured -q`
  - 失败符合预期：`channels["actor"].puts` 为空。
- RED: `conda run -n dreamervla python -m pytest tests/unit_tests/test_manual_cotrain_ray_runner.py::test_manual_cotrain_tiny_wm_env_num_envs_tracks_envs_per_worker_and_disables_loggers -q`
  - 失败符合预期：`env.real.cfg.emit_actor_trajectories` 不存在。
- GREEN focused: `conda run -n dreamervla python -m pytest ... -q`
  - 5 个新增/相关 focused tests 通过。
- Full affected tests:
  - `conda run -n dreamervla python -m pytest tests/unit_tests/test_manual_cotrain_ray_runner.py tests/unit_tests/test_trajectory_env_worker.py -q`
  - 结果：`103 passed in 3.99s`。
- Ruff:
  - `conda run -n dreamervla ruff check dreamervla/runners/manual_cotrain_ray_runner.py dreamervla/workers/env/trajectory_env_worker.py tests/unit_tests/test_manual_cotrain_ray_runner.py tests/unit_tests/test_trajectory_env_worker.py`
  - 结果：`All checks passed!`
- Step 3b smoke:
  - `WANDB_MODE=offline HYDRA_FULL_ERROR=1 conda run -n dreamervla python -m dreamervla.train experiment=manual_cotrain_ray_tiny manual_cotrain.global_steps=5 manual_cotrain.learner_update_step=1 training.out_dir=/tmp/dvla-step3b-manual-tiny-5`
  - 结果：exit 0；stdout 包含 `[manual-cotrain] groups=LearnerGroup,ActorGroup,RolloutGroup,EnvGroup`。
  - `/tmp/dvla-step3b-manual-tiny-5/resolved_config.yaml` 确认 `global_steps: 5`、`learner_update_step: 1`、`emit_actor_trajectories: true`。
  - `/tmp/dvla-step3b-manual-tiny-5/run_manifest.json` 存在。
  - `diagnostics/manual_cotrain_progress/global_step_00000001` 到 `global_step_00000005` 均有 `real_env_0.json`，内容均为 `finished: true, done: 2, total: 2`。

## 结论

DONE。`manual_cotrain_ray_tiny` CPU/Ray 路径已跑满 5 个 global step。

## 下一步建议

推进 Step 3c：真实 32/256/512 配置跑满 5 global_step，并与 Step 3a 的 base-VLA SR 做趋势对比。

## 残留风险

本轮 runner 侧小修叠在当前工作树已有的 manual cotrain role-key / multi-real-worker 在途 diff 上；提交时需要继续避免把无关在途改动整文件带入。
