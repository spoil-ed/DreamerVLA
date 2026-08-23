# Diagnostics 与 Runtime 清理设计

## 目标

删除已经失去生产入口的诊断代码和旧 non-Ray online cotrain 路线，并把仍在生产路径上的
runtime 代码按职责归类。清理后的仓库只保留当前主线：Ray collection/cotrain、独立 WM 与
classifier warmup、LIBERO evaluation，以及这些路径确实使用的诊断工具。

本次是删除和归类，不新增训练能力，不改变 replay buffer 的实现、schema、采样或恢复行为。

## 已确认的边界

- 正式废弃 `WorldModelTrainingRunner` 中由
  `online_rollout.total_env_steps > 0` 激活的旧 non-Ray online cotrain。
- `WorldModelTrainingRunner` 只负责离线 WM/classifier warmup；Ray `CotrainRunner` / 主线
  worker groups 继续负责 online cotrain。
- 已有 WM/classifier epoch checkpoint、optimizer/RNG resume、TensorBoard/W&B resume 和
  W&B offline sync 行为保持不变。
- `online_replay.py`、`offline_seed.py`、Ray replay worker 及其现有测试不属于本次清理范围。
- Hydra 继续是配置源；若旧配置尝试用 `WorldModelTrainingRunner` 启动非零 online env steps，
  配置验证应明确拒绝并提示使用 Ray cotrain，而不是静默忽略。

## 删除策略

删除分三类进行，每一类都先用引用搜索和测试确认依赖闭包。

### 1. 确定孤儿

直接删除没有生产调用者的文件和符号：

- `dreamervla/diagnostics/_common.py`；
- 已被 `dreamervla/launchers/wandb_sync.py` 取代的
  `dreamervla/diagnostics/wandb_relay_sync.py`；
- 只有自身单测、没有脚本/config/docs 入口的 raw/VLA single-trajectory overfit probes；
- runtime 中只有单测引用的旧 checkpoint mixin、world-model state loader 和未使用 helper；
- 与上述孤儿一一对应、只证明已删除接口存在的测试。

删除测试不等于降低现行行为覆盖率：凡是仍被生产代码调用的行为测试继续保留或迁移。

### 2. 旧 non-Ray online cotrain

从 `WorldModelTrainingRunner.run()` 删除 online phase dispatch、encoder 恢复和 online resume
准备逻辑；保留离线 replay seed、WM warmup、classifier warmup、checkpoint 与 resume。

从 `world_model_training_common.py` 删除只服务旧路线的环境构建、vectorized rollout、训练
burst、online checkpoint sidecar 和 rollout progress 逻辑。共享的 Hydra component 构造、
task-conditioning 校验、WM/classifier 构造和 warmup 所需方法继续保留。

对应的旧 online orchestration/render tests 删除；离线 warmup、checkpoint、resume、设备释放和
配置校验测试继续保留。旧 `experiment_stage_checks.py` 中没有现行脚本入口的 non-Ray stage
subcommands 一并删除，只保留实际被 classifier eval launcher 使用的 `cls-eval`。

### 3. 错位生产代码与低风险归类

- 把 `diagnostics/eval_cotrain_transaction.py` 移到 `runtime/cotrain_eval.py`，因为它是
  `LIBEROVLAEvaluationRunner` 的正式 observer，而不是人工诊断 CLI。
- 把只被 Ray rollout collection 消费的 collection config builder 合入
  `runtime/rollout_collection_ray.py`，避免一个单消费者转发模块。
- 把 `SuccessTracker` 归入统一 runtime metrics 模块；删除 `online_utils.py` 中失去用途的
  checkpoint loader。
- 把仍需使用的 visualization helper 移入其实际消费者；若没有生产消费者，则随旧 mixin
  文件整体删除。
- 删除确定未使用的 dataclass/helper；不为减少文件数而强行合并 evaluation mixins、Ray worker
  边界或 replay 模块。

这一步采用保守合并：只有职责相同且依赖方向清晰的模块才合并。`runtime/` 的目标不是变成
单个大文件，而是消除无入口代码和只做转发的碎片。

## Diagnostics 保留标准

`dreamervla/diagnostics/` 只保留满足至少一个条件的工具：

- 被现行 shell launcher 或 runner 调用；
- 在 install、EGL、Ray worker 或真实环境故障排查中有明确人工入口；
- 被现行文档教程引用并能独立运行。

因此 official eval、install verification、EGL pressure、manual worker benchmark、online env
smoke 和仍有文档入口的 WM overfit 工具保留。生产 runner 依赖的模块必须移出 diagnostics，
不能仅因当前 import 能工作就继续混放。

## 配置与错误处理

`validate_cfg` 对 `WorldModelTrainingRunner` 的 `online_rollout.total_env_steps > 0` 给出明确
错误，错误信息应指出该 runner 仅支持 offline warmup，并引导使用 Ray cotrain experiment。
已有 `total_env_steps: 0` 配置暂时可继续保留兼容；只有确认该字段不再被离线 warmup 借用后
才删除配置块。这样避免借清理之名改动 replay 或训练参数来源。

移动模块时不保留永久兼容 import shim：仓库内调用者和测试在同一提交更新。对已经对外发布
且仍有入口的 CLI 才保留命令兼容；本次删除的孤儿 CLI 没有兼容承诺。

## 验证

每个删除/迁移任务执行以下验证：

1. `rg` 证明旧模块、旧类名和旧 online 方法没有剩余生产引用；
2. 先调整或增加失败测试，再实施删除/迁移；
3. 运行对应 runner、config、diagnostics/runtime 单元测试；
4. 运行不涉及 replay 状态变更的完整 unit test suite；
5. 运行 Ruff、shell syntax、`git diff --check` 和 repository hygiene tests；
6. 确认 W&B sync launcher、WM/classifier resume 和 Ray cotrain resume 回归测试仍通过。

验收结果应满足：旧 non-Ray online cotrain 无可达入口；diagnostics 中不再放正式 runtime
依赖；删除的文件没有 import 残留；现行 Ray/离线 warmup/eval 路线行为不变。

