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

Removed from current sync and Ray runner scalar metrics:
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

## train/time/eval

1. `train/<phase>_loss`, `train/rl_loss`
   Source: synthetic or phase-updater learner paths, not the main cotrain loop. Keep
   only for those modes.
2. `time/*`
   Source: timing diagnostics when present. Keep.
3. `eval/*`
   Source: real eval windows when present. Keep; eval success must come from completed
   real episodes.

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
