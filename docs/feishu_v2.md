# DreamerVLA 执行摘要

## 技术主张

DreamerVLA 以 OpenVLA-OFT 的投影视觉输入 token 为唯一世界模型观测：
`[T,256,4096]`。同一 token grid 贯穿真实采集、replay、world-model warm-up、
success-classifier training、imagined rollout 和 actor bridge。

## 已落地的训练闭环

```text
one-trajectory OpenVLA collection
  -> canonical reward/hidden-token shards
  -> replay seed
  -> WM + classifier warm-up
  -> Ray cotrain
  -> real LIBERO evaluation
```

Ray 路线按照 Actor、Rollout、Env 和 Learner 四组拆分；其中 Learner 额外负责
world model 与 classifier，Rollout 只做无梯度推理，Actor 负责 FSDP 策略更新。

## 研究问题

当前需要通过实验回答：

1. token world model 的多步 free-running 误差是否受动作条件显著影响；
2. success classifier 在真实失败、真实成功与 imagined trajectory 之间是否保持校准；
3. world-model rollout 是否能在相同真实交互预算下提高 OpenVLA-OFT 成功率；
4. Ray 各 worker group 在目标 batch geometry 下是否保持稳定吞吐与一致的 resume 轨迹。

## 必要对照

- one-trajectory OpenVLA-OFT，仅监督 checkpoint；
- 仅真实 rollout 的在线策略更新；
- world model 但不使用 classifier reward；
- classifier 但不使用 imagined rollout；
- 完整 DreamerVLA。

所有对照保持相同的 `[256,4096]` 外部观测契约，内部模型宽度、rollout horizon
和训练预算通过 Hydra 显式记录。

## 工程完成条件

- 四个 LIBERO suite 的主线配置可 compose；
- collection reuse 会扫描并验证每个 sidecar demo；
- classifier 和 world model 的 standalone train 各完成真实反向传播、optimizer step
  和 checkpoint save；
- sync 与 Ray learner 调用同一 classifier update；
- 完整测试通过后再提交并 push。
