# Step 3a Report - Base OpenVLA-OFT Eval Harness

## 本步目标

启动 R1 的第一项要求：先用同一个 `EmbodiedEvalRunner` harness 跑原始 OpenVLA-OFT base VLA（`eval.ckpt_kind=vla`），确认最小 LIBERO eval 能执行并把 SR 指标落到 run root。完整同配置基线和 cotrain 后趋势对比留给 Step 3c。

## 改动文件

- `dreamervla/runners/embodied_eval_runner.py`
  - 增加 `_OFTBaseEvalAdapter`，让现有 LIBERO eval loop 能驱动 OpenVLA-OFT extractor。
  - 新增 `_oft_base_policy_cfg()` / `_use_oft_base_eval()` / `_build_oft_base_eval_adapter()`，当 `eval.ckpt_kind=vla` 且 checkpoint 是 HF OpenVLA-OFT 时，从 `task.openvla_oft` 元数据构造 `OFTRolloutBundle`。
  - 在 `_generate_actions()` 中增加 base-OFT 分支：用当前 LIBERO raw obs 构造 OFT obs，调用 extractor，按 `action_steps` 返回并复用 `process_action()` 做 gripper 后处理。
- `tests/unit_tests/test_openvla_oft_base_eval_runner.py`
  - 新增 RED/GREEN 单测，钉住 policy cfg 来自 `task.openvla_oft`，以及 base-OFT eval action chunk 会经过 gripper 后处理。
- `logs/loop_progress.md`
  - 将 Step 3 拆成 Step 3a/3b/3c，并把 Step 3a 标为 DONE。

## 验证命令与真实输出摘要

- RED
  - 命令：`conda run -n dreamervla python -m pytest tests/unit_tests/test_openvla_oft_base_eval_runner.py -q`
  - 初始结果：2 failed；失败点分别是 `EmbodiedEvalRunner` 缺 `_oft_base_policy_cfg`，以及 `_generate_actions()` 仍落到 RynnVLA/Pretokenize path。
- GREEN
  - 命令：同上。
  - 结果：`2 passed, 2 warnings in 5.21s`。
- Compose
  - 命令：compose `experiment=eval_libero_vla +task=openvla_onetraj_coldstart_libero eval.ckpt_kind=vla ...` 并打印分支判断。
  - 结果：`target dreamervla.runners.EmbodiedEvalRunner`、`use_oft_base True`，policy cfg 指向 `Openvla-oft-SFT-libero-goal-traj1` 与 `unnorm_key=libero_goal_no_noops`。
- 真实 base-VLA eval smoke
  - 命令要点：`CUDA_VISIBLE_DEVICES=0,1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 ... python -m dreamervla.train experiment=eval_libero_vla +task=openvla_onetraj_coldstart_libero eval.ckpt_kind=vla eval.ckpt_path=<Openvla-oft-SFT-libero-goal-traj1> eval.task_ids='[0]' eval.num_episodes_per_task=1 eval.max_steps=1 eval.num_steps_wait=0 training.out_dir=/tmp/dvla-step3a-base-vla-eval-smoke`
  - 结果：退出码 0；输出包含 `Loading OFT policy (discrete)`、`EVALUATION - done · succ 0.000`、`wrote metrics -> /tmp/dvla-step3a-base-vla-eval-smoke/eval_libero_metrics.json`。
- 指标落盘
  - 命令：`python -m json.tool /tmp/dvla-step3a-base-vla-eval-smoke/eval_libero_metrics.json`
  - 结果字段包含 `eval_success_rate: 0.0`、`eval_total_episodes: 1.0`、`eval_tasks: 1.0`、`results/task_macro_success_rate: 0.0`。
- focused 单测
  - 命令：`conda run -n dreamervla python -m pytest tests/unit_tests/test_openvla_oft_base_eval_runner.py tests/unit_tests/test_libero_eval_protocol_compat.py::test_eval_libero_config_uses_rlinf_protocol_defaults tests/unit_tests/test_libero_eval_protocol_compat.py::test_eval_summary_averages_three_trials_per_task -q`
  - 结果：`4 passed, 2 warnings in 4.62s`。
- ruff
  - 命令：`conda run -n dreamervla ruff check dreamervla/runners/embodied_eval_runner.py tests/unit_tests/test_openvla_oft_base_eval_runner.py`
  - 结果：`All checks passed!`

## 结论

DONE。`EmbodiedEvalRunner` 现在能用 `eval.ckpt_kind=vla` 跑 OpenVLA-OFT base checkpoint，并在最小 LIBERO eval smoke 中成功写出 SR JSON。

## 下一步建议

进入 Step 3b：运行 `manual_cotrain_ray_tiny`，覆盖 `manual_cotrain.global_steps=5`，确认 tiny cotrain 端到端绿并落盘基础 artifacts。

## 残留风险

- 本轮是 one-task / one-episode / `max_steps=1` 的 harness smoke；R1 的完整 base SR 和 cotrain 后趋势对比仍需 Step 3c 用真实评测配置完成。
- 当前工作区存在未提交的 eval/cotrain 相关在途改动；本轮提交只应包含 Step 3a 的新 adapter、单测和 loop 日志。
