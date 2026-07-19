# Online Cotrain Metrics Inventory

本文档盘点 online cotrain 路径当前记录的 scalar metrics。原则是：`rollout/`
只表示真实环境 completed episodes；imagined rollout 与 classifier score 只能放在
`rl/` 或 `LUMOS/` 诊断命名空间。

## rollout

1. `rollout/success_rate`
   Source: real env completed episodes. Keep. Cumulative success rate.
2. `rollout/success_rate_valid`
   Source: real env completed episodes. Keep. No completed episode 时为 0。
3. `rollout/recent_success_rate`
   Source: real env completed episodes. Keep. Recent-window online movement signal.
4. `rollout/recent_success_rate_valid`
   Source: real env completed episodes. Keep.
5. `rollout/episodes`
   Source: real env completed episodes. Keep.
6. `rollout/successes`
   Source: real env completed episodes. Keep.
7. `rollout/env_steps`
   Source: real env step counter. Keep.
8. `rollout/num_envs`, `rollout/episode_horizon`
   Source: config/run context. Keep as low-risk diagnostics.
9. `rollout/active_episode_step_min`, `rollout/active_episode_step_mean`,
   `rollout/active_episode_step_max`, `rollout/episode_progress_max`
   Source: active real rollout progress. Debug diagnostic; not success evidence.

Removed from current Ray runner scalar metrics:
`rollout/avg_success_rate` and `rollout/current_success_rate` because they duplicate or
muddy the cumulative/recent semantics.

## wm

1. `wm/loss`
   Source: replay WM learner update. Keep.
2. `wm/hidden_rec_loss`, `wm/hidden_cosine_loss`
   Source: WM implementation when emitted. Keep when present.
3. `wm/full_hidden_rec_loss`, `wm/full_hidden_cosine_loss`
   Source: sequence/full-hidden WM targets. Keep only when the selected WM emits them.

## cls

1. `cls/loss`
   Source: classifier replay update. Keep.
2. `cls/acc`
   Source: classifier replay update. Keep.
3. `cls/f1`
   Source: classifier replay update. Keep; primary classifier readiness metric.
4. `cls/pos_frac`, `cls/prob_mean`, `cls/grad_norm`
   Source: classifier replay update. Useful diagnostics; not real rollout success.

## rl

1. `rl/returns_mean`, `rl/returns_std`
   Source: imagined rollout outcome. Keep; actor-signal diagnostics, not real success.
2. `rl/actor_loss`
   Source: PPO/GRPO actor update. Keep.
3. `rl/policy_grad_norm`
   Source: PPO/GRPO actor update. Keep.
4. `rl/skipped_zero_variance_groups`
   Source: imagined rollout groups with no outcome variance. Keep.
5. `rl/ppo_step_applied`
   Source: PPO optimizer step gate. Keep.
6. `rl/advantage_std`, `rl/advantage_mag`
   Source: PPO advantage diagnostics. Keep as debug/diagnostic.
7. `rl/actor_signal_ready`, `rl/skipped_no_signal`, `rl/classifier_f1_gate`,
   `rl/classifier_updates`
   Source: Ray learner actor-signal gate. Keep as Ray diagnostics.

## actor（manual Ray PPO）

1. `actor/ppo_optimizer_steps`
   Source: completed policy optimizer steps in the current manual global step. This is
   the authoritative “PPO updated” counter; `actor/ppo_updates` is a compatibility alias.
2. `actor/ppo_forward_backward_steps`, `actor/ppo_progress_ops`
   Source: completed micro-batch forward/backward operations and the sum of
   forward/backward plus optimizer operations. These are progress counters, not quality
   metrics.
3. `actor/global_rollout_trajectories`, `actor/global_ppo_samples`,
   `actor/global_loss_mask_sum`
   Source: ActorGroup-wide trajectory count, flattened chunk-sample count, and valid
   sample count after termination/reward filtering.
4. `actor/global_batch_size`, `actor/per_rank_global_batch_size`,
   `actor/micro_batch_size`
   Source: resolved Hydra/FSDP batch hierarchy. These are run-contract diagnostics.
5. `actor/policy_loss`, `actor/total_loss`
   Source: micro-batch mean PPO surrogate and the optimized objective after KL/entropy
   terms. `actor/loss` is a compatibility alias of `actor/total_loss`.
6. `actor/ratio`, `actor/ratio_abs`, `actor/clipped_ratio`, `actor/approx_kl`,
   `actor/clip_fraction`, `actor/dual_clip_fraction`
   Source: RLinf-style PPO stability diagnostics averaged over micro batches and Actor
   ranks.
7. `actor/grad_norm`, `actor/lr`
   Source: policy gradient norm and optimizer learning rate after each global batch.
   `actor/policy_grad_norm` is a compatibility alias.
8. `actor/skipped_zero_valid_update`, `actor/zero_loss_micro_batches`
   Source: explicit no-signal/update diagnostics. A skipped update must have
   `actor/ppo_optimizer_steps=0`.

## LUMOS

1. `LUMOS/success_rate`
   Source: imagined classifier-complete fraction. Keep only as LUMOS diagnostic.
   It must never be treated as `rollout/success_rate`.
2. `LUMOS/score_mean`, `LUMOS/score_std`
   Source: imagined classifier score/probability. Keep.
3. `LUMOS/group_var_keep_frac`, `LUMOS/skipped_zero_variance_groups`,
   `LUMOS/num_mixed_groups`, `LUMOS/num_all_success_groups`,
   `LUMOS/num_all_fail_groups`
   Source: imagined GRPO group composition. Keep.
4. `LUMOS/group_success_rates`, `LUMOS/group_success_counts`,
   `LUMOS/group_rollout_successes`, `LUMOS/group_finish_steps`,
   `LUMOS/group_has_variance`
   Source: per-group diagnostic payloads. These are structured diagnostics and should
   not be logged as scalar TensorBoard series.

## eval（resident real-LIBERO）

1. `eval/success_rate`, `eval/successes`, `eval/episodes`
   Source: resident RolloutGroup 在固定真实 LIBERO protocol 上完成的 episodes。
2. `eval/wm_trajectory_cosine`（alias: `eval/wm_closed_loop_cosine`）,
   `eval/wm_closed_loop_mse`
   Source: 同一批真实 eval 轨迹。WM 只用真实 history 初始化，随后全程递归预测；
   先对每条轨迹的有效 horizon 求均值，再对轨迹等权平均。
3. `eval/cls_trajectory_f1`, `eval/cls_trajectory_accuracy`（aliases:
   `eval/classifier_real_f1`, `eval/classifier_real_accuracy`）
   Source: checkpoint threshold 下，CLS 对完整真实 hidden 轨迹的 success 分类；label
   是对应真实 episode 的 success。
4. `eval/classifier_wm_f1`, `eval/classifier_wm_accuracy`
   Source: 相同 CLS 和 threshold 对 WM 闭环预测 hidden 轨迹的 success 分类，用于区分
   CLS 本身效果与 WM->CLS 组合效果。
5. `eval/classifier_{real,wm}_{precision,recall,roc_auc,pr_auc}` 及 confusion counts
   Source: 同一条 trajectory-level 分类协议的辅助诊断。AUC 只有同时存在正负样本时定义。
6. `eval/chunk_per_s`, `time/eval_s`, `time/eval_diagnostics_s`
   Source: 真实环境 rollout throughput、完整 eval 用时和其中 WM/CLS 诊断用时。

这些指标在启用 resident eval 的每个 eval global step 一起写入 logger；eval 轨迹不写
replay，诊断不做 optimizer step，也不重新校准 classifier threshold。

## train/time

1. `train/<phase>_loss`, `train/rl_loss`
   Source: synthetic or phase-updater learner paths, not the main cotrain loop. Keep
   only for those modes.
2. `time/*`
   Source: timing diagnostics when present. Keep.

## sync（manual Ray）

1. `sync/learner_state_dicts_s`
   Source: LearnerWorker materializing the CPU WM/classifier snapshot after an update.
2. `sync/learner_state_share_s`
   Source: the driver placing that snapshot in Ray's object store exactly once. This is
   serialization/share cost, not model compute.
3. `sync/wm_env_load_component_states_s`
   Source: end-to-end wait until all WMEnv workers have loaded the shared snapshot.
4. `sync/world_model_load_s`, `sync/classifier_load_s`
   Source: worker-side component load timings. Keep as state-distribution diagnostics.

## Screened Set

The main online cotrain dashboard should prioritize:

1. `rollout/success_rate`
2. `rollout/success_rate_valid`
3. `rollout/recent_success_rate`
4. `rollout/recent_success_rate_valid`
5. `rollout/episodes`
6. `rollout/env_steps`
7. `rl/returns_mean`
8. `rl/returns_std`
9. `rl/actor_loss`
10. `rl/policy_grad_norm`
11. `rl/skipped_zero_variance_groups`
12. `cls/f1`
13. `cls/acc`
14. `wm/loss`
15. WM hidden reconstruction/cosine losses when emitted by the selected WM.
16. `actor/ppo_optimizer_steps`
17. `actor/policy_loss`
18. `actor/approx_kl`
19. `actor/clip_fraction`
20. `actor/grad_norm`
21. `eval/success_rate`
22. `eval/wm_trajectory_cosine`
23. `eval/cls_trajectory_f1`
24. `eval/cls_trajectory_accuracy`
