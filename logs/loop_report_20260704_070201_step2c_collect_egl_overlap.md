# Step 2c Report - Collect EGL Render/Inference GPU Overlap

## 本步目标

继续处理 Step 2c 的阻塞点：三处默认 `render_backend=egl` 已接线，但真实 Ray collect EGL smoke 在 EnvWorker 子进程 step 阶段死亡。本轮目标是定位并消除 collect 默认 EGL 的原生子进程死亡，保留 osmesa 显式回退与零/单 GPU 不满足条件时的明确拒绝。

## 改动文件

- `dreamervla/runners/cold_start_ray_collect_runner.py`
  - `_ensure_collect_render_device_pool()` 增加 `collect_cfg` 输入，默认 render pool 不再使用全部 cluster GPU，而是排除 collect inference worker 使用的 GPU。
  - 当 `render_backend=egl` 且没有可与 inference GPU 分离的 render GPU 时，提前抛 `ValueError`，提示暴露额外 GPU / 显式 `+env.cfg.render_devices` / 使用 `render_backend=osmesa`。
  - `_build_oft_components()` 传入 collect 配置，保证默认 pool 与实际 inference GPU 分配一致。
- `tests/unit_tests/test_ray_coldstart_real_config.py`
  - 将 collect EGL 默认 pool 单测改为断言默认选择 non-inference GPUs。
  - 新增单 GPU/无 spare render GPU 的 fail-fast 单测。
  - 保留显式 render pool 优先级单测。
- `tests/unit_tests/test_libero_render_regime_wiring.py`
  - 增加 manual cotrain real env 在 cfg 无显式 render pool 时复用 worker 可见 GPU 的覆盖，防止 Ray actor 分配 GPU 后 real env 入口回退成“空池”。
- `logs/loop_progress.md`
  - 更正 loop 事实源路径为 `docs/superpowers/plans/2026-07-04-mainline-deprecation-egl-align.md`。
  - Step 2c 标为 DONE。

## 验证命令与真实输出摘要

- RED（collect pool 新行为）
  - 命令：`conda run -n dreamervla python -m pytest tests/unit_tests/test_ray_coldstart_real_config.py::test_collect_egl_render_pool_defaults_to_non_inference_gpus tests/unit_tests/test_ray_coldstart_real_config.py::test_collect_egl_render_pool_rejects_inference_overlap_without_spare_gpu tests/unit_tests/test_ray_coldstart_real_config.py::test_collect_egl_render_pool_preserves_explicit_pool -q`
  - 初始结果：3 failed，原因是 `_ensure_collect_render_device_pool()` 尚不接受 `collect_cfg`。
- GREEN（collect pool 新行为）
  - 同命令重跑：`3 passed in 1.23s`。
- focused 单测
  - 命令：`conda run -n dreamervla python -m pytest tests/unit_tests/test_ray_coldstart_real_config.py tests/unit_tests/test_libero_render_regime_wiring.py tests/unit_tests/test_egl_device.py tests/unit_tests/test_mainline_egl_defaults.py tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py tests/unit_tests/test_manual_cotrain_ray_runner.py::test_manual_cotrain_oft_real_rollout_uses_oft_encoder_and_action_postprocess tests/unit_tests/test_manual_cotrain_ray_runner.py::test_manual_runner_injects_top_level_render_backend_into_real_env_cfg tests/unit_tests/test_manual_cotrain_ray_runner.py::test_manual_runner_real_render_backend_overrides_real_env_cfg_backend -q`
  - 结果：`119 passed in 10.19s`。
- ruff
  - 命令：`conda run -n dreamervla ruff check dreamervla/runners/cold_start_ray_collect_runner.py tests/unit_tests/test_ray_coldstart_real_config.py tests/unit_tests/test_libero_render_regime_wiring.py`
  - 结果：`All checks passed!`
- Hydra compose 默认值
  - 命令：compose `collect_rollouts_ray`、`openvla_onetraj_libero_cotrain_ray`、`scripts/coldstart_warmup_cotrain` 并打印 render keys。
  - 结果：`collect_env_render egl`、`ray_top_render egl`、`ray_real_render egl`、`script_render egl`、`script_eval_render egl`、`script_profile_real_render None`。
- 真实 collect EGL smoke（2 GPU）
  - 命令要点：`CUDA_VISIBLE_DEVICES=0,1 ... dreamervla.train experiment=collect_rollouts_ray ... env.cfg.render_backend=egl ... collect.episodes_per_task=1 ... rollout.target_episodes=1`
  - 结果：退出码 0；输出包含 `COLDSTART COLLECT - done · 1 episodes · succ 0.000`，未再出现 EnvWorker child death。
- 单 GPU 默认 EGL fail-fast
  - 命令要点：同 collect smoke，但 `CUDA_VISIBLE_DEVICES=0`。
  - 结果：退出码 1；明确报错 `collect render_backend=egl needs a render GPU disjoint from inference GPU(s) [0] ... or use render_backend=osmesa.`，不是 native SIGABRT/EOF。
- cotrain-real 直接 EGL reset+step smoke
  - 命令要点：`CUDA_VISIBLE_DEVICES=1` 下 compose `openvla_onetraj_libero_cotrain_ray`，直接构造 `RealEnvWorker`，执行 `init()`、`bootstrap_obs()` 和一次零动作 `_step_slot()`。
  - 结果：退出码 0；输出包含 `real_env_render egl`、`mujoco_gl egl`、`mujoco_egl_device 1`、`bootstrap_task 0`、`next_step 1`。

## 结论

DONE。Step 2c 的阻塞点收敛为 collect 默认 render pool 与 inference GPU 重叠；默认策略现已改为选择 non-inference GPUs，并在没有 spare render GPU 时 fail-fast 指向 osmesa 回退。collect 真实 EGL 通过，cotrain-real 的 LIBERO EGL env 构造与 reset/step 通过，三处默认仍为 EGL。

## 下一步建议

进入 Step 3：跑 base-VLA `eval.ckpt_kind=vla` 基线，以及 `manual_cotrain_ray_tiny` 和真实 32/256/512 的 5 global_step cotrain/SR 验收。

## 残留风险

- 本轮 direct cotrain-real smoke 覆盖了 `RealEnvWorker` 的 LIBERO EGL reset+step；完整 manual cotrain 5-step、SR 趋势和 eval 落盘属于 Step 3。
- 单 GPU机器默认 EGL collect 现在会拒绝运行；这符合“EGL 需要 disjoint render GPU”的 fail-fast 策略，显式回退为 `render_backend=osmesa`。
