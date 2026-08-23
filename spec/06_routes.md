# 路由清单：主线与支线

本文件只回答一个问题：**当前唯一主线是什么**。其余路由都是支线——保留为可选项或测试夹具，
**不要把它们误当主线**。`experiment=<name>` 是 Hydra 入口（`python -m dreamervla.train
experiment=<name> task=<suite>`）。

## 主线（唯一）

OpenVLA-OFT 单轨迹 LIBERO 冷启动 → cotrain：

```
collect -> seed replay + warmup world model/classifier -> online cotrain -> eval
```

| 阶段 | experiment | runner |
| --- | --- | --- |
| 采集 | `collect_rollouts` | `RolloutCollectionRunner` |
| WM warmup | `dreamer-wm` / `dino-wm` | `WorldModelTrainingRunner` / `DinoTokenWorldModelTrainingRunner` |
| classifier warmup | `classifier_official_upper_bound` | `SuccessClassifierTrainingRunner` |
| cotrain | `openvla_libero` | `DreamerRunner` |
| 评估 | `eval_cotrain` | `LIBEROVLAEvaluationRunner` |

启动流程见 [`04_complete_loop.md`](04_complete_loop.md) 与 [`05_ray_runtime.md`](05_ray_runtime.md)；
操作配方见 [`../docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`](../docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md)。

`DreamerRunner` 是冻结 WM/CLS、只更新 actor 的 failure-conditioned imagined-RL
特化；它复用 `CotrainRunner` 的 Ray loop，不维护第二套 backend。

## 支线（非主线，config 保留为可选项 / 工具，勿当主线）

这些 config 仍可用，但**不在主线数据流里**：

| 路线 | experiment | runner |
| --- | --- | --- |
| 完整 staged cotrain | `openvla_onetraj_libero_cotrain` | `CotrainRunner` |

该 staged 路线包含 encoder SFT、WM/classifier learner update、imagined rollout 和
actor update；它与主线共享 Ray group 实现，但不是 `openvla_libero` 的别名。

**OpenVLA-OFT 阶段工具**——训练主线消费的 OFT VLA/WM/classifier，本身不是 cotrain 流程：

- full-replay world model：`wm_full_dataset_train`
- classifier：`wmpo_token_classifier_openvla_onetraj_libero_goal_h1`
- 这三条工具与主线共享 `hidden_token [256,4096]` 观测契约。

**其他 VLA 家族**——其模型代码和已有数据产物仍作为独立支线保留；它们自己的
hidden-token 语义不属于 OpenVLA-OFT 主线契约，不能据此改写为 hidden-token。

## 维护约定

新增 `experiment=` 时，回到本表归类：要么进主线（同时更新 `04_complete_loop.md` 与
`AGENTS.md` 的主线说明），要么明确写进“支线”或“测试夹具”。**任何时候都不能让仓库看起来
分不清主线。**
