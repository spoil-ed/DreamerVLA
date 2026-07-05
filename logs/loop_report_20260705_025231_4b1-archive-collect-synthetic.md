# Loop 报告 — Step 4b-1:归档 collect_rollouts_ray_synthetic(§3.1 首批)

## 本步目标
用户已定 Step 4b 走「小步分批」。本批取 §3.1 中**对单测基线零耦合**的最轻单元:归档 synthetic 后端冒烟 experiment `collect_rollouts_ray_synthetic`,并同批移除唯一绑定它的 e2e 测试用例,保持套件不新增失败。作为归档流水线(grep→git mv→manifest→verify)的首个端到端验证。

## 改了哪些文件 + 理由
- `configs/experiment/collect_rollouts_ray_synthetic.yaml` → `archive/configs/experiment/`(git mv,绝不 rm):SPEC §3.1 明列;grep 确认无 mainline 源引用(仅 e2e test_s6 + superpowers 计划文档)。
- `tests/e2e_tests/test_s6_ray_coldstart_collect.py`:删 `test_ray_coldstart_synthetic_experiment_runs_through_train_entry`——它是唯一 compose+run 该 archived experiment 的用例(局部 import 随函数移除,无孤儿);其余 3 个用例直接构造 mainline `ColdStartRayCollectRunner`,不受影响。
- `docs/superpowers/DEPRECATION-manifest.md`:追加 1 行(原路径→archive→理由→迁移标记 4b-1)。
- `logs/loop_progress.md`:4b-1 行 DOING→DONE。
- 本报告。

## verify 命令与真实输出
- ruff:`ruff check tests/e2e_tests/test_s6_ray_coldstart_collect.py` → `All checks passed!`
- test_s6 可导入:`pytest ... --collect-only` → `3 tests collected`(synthetic 用例已干净移除)。
- compose 主线 6:`scratchpad/compose_mainline6.py` → 6/6 PASS,`COMPOSE6_EXIT= 0`。
- 全量单测:`pytest tests/unit_tests -q` →
  ```
  4 failed, 1350 passed, 7 skipped, 43 warnings in 164.02s
  ```
  4 个失败**全部 ⊆ BASELINE-0 allowlist**(test_env_full_record、test_learner_worker_manual_precision、
  test_multistep_rollout_worker::test_generate_rank_keyed_batch_sends_direct_batched_payload、
  test_repository_hygiene::test_files_live_under_their_architecture_domains)。**零 allowlist 之外新失败**;
  计数 5→4 因 `test_multistep_rollout_worker::test_generate_reads_channel_writes_results_and_stops`
  (Ray 启动 flaky)本次翻绿,与基线注记一致。
- git:提交后 `git status --short | grep -c '^R'` == 74(见提交步骤输出)。

## 结论
**DONE**。归档流水线端到端验证通过:compose 6/6 绿、套件零新增失败、ruff 干净、manifest+还原可追溯。

## 下一步
4b-2:继续 §3.1 轻耦合子集——`online_cotrain_ray_synthetic`(需 repoint 单测 `test_manual_resource_config_groups` + e2e test_s5 + routes.md 行)与 `online_cotrain_ray_dreamervla_tiny`(e2e test_s5 + configs/README 行);之后再进重耦合的 WM/classifier/VLA-SFT 注册表簇(整份 test_runner_public_api + docs 重写)。

## 残留风险
- 归档 synthetic 冒烟略降低主线 `ColdStartRayCollectRunner` 的无 GPU e2e 覆盖(SPEC §3.1 已授权;若用户认为该冒烟应保留,可用 restore 脚本一键回退)。
- 基线 4/5 失败含 Ray 启动 flaky,后续批次比对以「未引入 allowlist 之外新失败」为准,不苛求精确 5。
