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
| 采集 | `collect_rollouts_ray` / `collect_rollouts_onetraj` | `ColdStartRayCollectRunner` / `CollectRolloutsRunner` |
| cotrain（同步 pipeline，含 warmup） | `openvla_onetraj_libero_cotrain_noray` | `OnlineCotrainPipelineRunner` |
| cotrain（Ray 异步 manual，**主路**） | `openvla_onetraj_libero_cotrain_ray` | `ManualCotrainRayRunner` |
| 评估 | `eval_libero_vla` | `EmbodiedEvalRunner` |

启动流程见 [`04_complete_loop.md`](04_complete_loop.md) 与 [`05_ray_runtime.md`](05_ray_runtime.md)；
操作配方见 [`../docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`](../docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md)。

`openvla_onetraj_libero_cotrain_ray` 的 dreamervla config 用 `defaults` 复用基座
`openvla_onetraj_libero_cotrain_ray_base`（`OnlineCotrainRayRunner` target）——基座只是
共享实现，不是单独的主线入口。

## 预主线因果测试（不是主线）

`scripts/e2e_frozen_model_pre_mainline.sh` 执行一条独立门禁：

```
official LIBERO -> WM upper bound + classifier upper bound
                -> freeze WM/CLS -> policy-only imagined RL -> matched real eval
```

对应 experiments 是 `wm_official_upper_bound`、
`classifier_official_upper_bound` 和 `dreamervla_frozen_models_rl`；最后仍由
`eval_libero_vla` 对原始 VLA 与 RL checkpoint 做相同协议的真实环境评估。只有
success rate 严格提升、policy 确实更新且 WM/CLS 完全不变时通过。该测试用于证明
冻结上限模型提供的 imagined-RL 信号有效，不改变上面的唯一正式主线。完整契约见
[`09_frozen_model_pre_mainline.md`](09_frozen_model_pre_mainline.md)。

## 支线（非主线，config 保留为可选项 / 工具，勿当主线）

这些 config 仍可用，但**不在主线数据流里**：

**OpenVLA-OFT 阶段工具**——训练主线消费的 OFT VLA/WM/classifier，本身不是 cotrain 流程：

- full-replay world model：`wm_full_dataset_train`
- classifier：`latent_classifier_openvla_onetraj_libero_goal_h1`、
  `wmpo_token_classifier_openvla_onetraj_libero_goal_h1`
- 这三条工具与主线共享 `hidden_token [256,4096]` 观测契约。

**其他 VLA 家族**——其模型代码和已有数据产物仍作为独立支线保留；它们自己的
hidden-token 语义不属于 OpenVLA-OFT 主线契约，不能据此改写为 hidden-token。

## 维护约定

新增 `experiment=` 时，回到本表归类：要么进主线（同时更新 `04_complete_loop.md` 与
`AGENTS.md` 的主线说明），要么明确写进“支线”或“测试夹具”。**任何时候都不能让仓库看起来
分不清主线。**
