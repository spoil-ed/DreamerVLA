# RynnVLA Encode DreamerVLA Scheme

## 定位

这条路线不是把 VLA 表征当成一个独立的低维 semantic bottleneck，也不是训练纯 pixel Dreamer。
它的核心是：观测仍然来自 LIBERO 的 RGB pixel，但 DreamerV3 RSSM 的 observation encoder 不再用
普通 CNN，而是使用 RynnVLA backbone 提取的 hidden observation。

```text
LIBERO pixel obs_t
  -> frozen / selected RynnVLA backbone encoder
  -> Rynn hidden observation e_t
  -> DreamerV3 RSSM posterior q(z_t | h_t, e_t)
  -> RSSM dynamics p(z_{t+1} | h_t, z_t, a_t)
  -> decoder / reward / continue
  -> imagined rollout
  -> actor / critic
```

这个方案保留 DreamerV3 的 RSSM 结构，只替换 `encode(images)` 这一层。也就是说，RSSM 仍然有
deterministic state `h_t` 和 stochastic state `z_t`，actor/critic 的训练仍然发生在 imagination 中。

## World Model

WM 阶段的输入和目标是：

```text
input:
  image_t, action_t, reward_t, done_t

encoder:
  image_t -> RynnVLA backbone -> e_t

RSSM:
  posterior: q(z_t | h_t, e_t)
  prior:     p(z_t | h_t)

loss:
  dynamics KL
  representation KL
  reward loss
  continue loss
  reconstruction loss
```

这里最关键的点是：`e_t` 不是最终 policy state，而是 RSSM posterior 的观测输入。真正用于 imagination
的 state 应该是 RSSM 的 `[h_t, z_t]`，因为只有它是由 dynamics 可以向前 rollout 的状态。

## Actor 路线

最初的 actor 设计是复用 RynnVLA 的 action head：

```text
RSSM imagined [h_t, z_t]
  -> actor adapter / decoder
  -> RynnVLA action-head-compatible hidden
  -> VLA action head
  -> continuous action chunk
```

这样做的动机是：RynnVLA action head 已经学会了从 backbone hidden context 到 action chunk 的映射，
所以 Dreamer actor 不必从零学一个全新的 action decoder。

## 卡住的问题：重建什么

这个方案真正卡住的位置不是 WM 的 RSSM，而是 actor 输入。

RynnVLA action head 原生吃的不是一个任意 4096-d 向量，而是经过 VLA backbone 后的 action-context
hidden。我们尝试过几种替代：

| 尝试 | 含义 | 问题 |
|------|------|------|
| pooled 4096 hidden | 把 full hidden sequence mean pool 成一个向量 | 接口简单，但丢失 token 位置和 action context |
| predicted 4096 hidden | 从 RSSM `[h,z]` 预测 pooled hidden | MSE 看起来可降，但 actor 对分布偏移很敏感 |
| full token hidden sequence | 从 RSSM 恢复 `[L,4096]` hidden states | 维度过大，恢复难度高，训练不稳定 |
| direct actor MLP | 不复用 VLA action head，直接从 `[h,z]` 输出 action | 稳定，但放弃了 VLA action head 的先验 |

因此，问题可以明确表述为：

```text
Dreamer RSSM 能产生可 rollout 的 [h,z]，
但 VLA action head 需要的是 VLA backbone 原生 hidden context。
二者之间缺少一个可靠、低维、action-relevant 的接口。
```

## 为什么 full hidden sequence 不合适

RynnVLA 的 full hidden states 通常是：

```text
hidden_states: [L, 4096]
```

其中 `L` 包括 prompt token、state token、image token、special token、action context 等。这个序列不是纯
物理状态，也不是全部都可由环境 dynamics 预测。强迫 RSSM 重建完整 `[L,4096]` 会让 WM 学很多与
动力学无关的语义和 prompt 信息，导致模型容量被错误使用。

这个也是之前实验中重建卡住的根本原因：不是 RSSM 不能预测未来，而是 full hidden reconstruction
目标本身过重，而且和 Dreamer 的 Markov latent 目标不一致。

## 可行修正：Action-Context Bottleneck

更合理的接口不是恢复全部 VLA hidden，而是只恢复 action head 真正需要的 action-context hidden。

推荐结构：

```text
RynnVLA full hidden states
  -> action-context selector / TokenLearner
  -> K 个 action-relevant tokens
  -> compact action context

Dreamer RSSM [h,z]
  -> action-context decoder
  -> K 个 action-relevant tokens
  -> RynnVLA action head
```

这里 TokenLearner 是合理的：它不应该用于无约束地“选重要区域然后恢复整段 hidden”，而应该用于从
VLA hidden sequence 中选出少量和 action head 直接相关的 token。这样 bottleneck 的目标从
`L * 4096` 降为 `K * 4096`，并且语义上更贴近 action head。

## 推荐最终版本

最终建议把这条路线定义为：

```text
pixel obs_t
  -> frozen RynnVLA backbone
  -> full hidden sequence
  -> TokenLearner / action-context selector
  -> compact action context c_t
  -> RSSM posterior q(z_t | h_t, c_t)

RSSM imagination:
  [h_t, z_t]
  -> action-context decoder
  -> predicted compact action context ĉ_t
  -> RynnVLA action head
  -> action_t
```

对应 loss：

```text
WM loss:
  dynamics KL
  representation KL
  reward loss
  continue loss

adapter loss:
  compact action-context reconstruction
  optional cosine alignment

actor/critic loss:
  DreamerV3 lambda-return actor loss
  twohot critic loss
```

注意这里不再要求重建 RGB，也不要求重建 full hidden sequence。重建目标只保留 action head 需要的
compact context。

## 当前实现状态

已经实现和验证过的部分：

- RynnVLA hidden sidecar 数据预处理。
- `DreamerV3PixelRynnBackboneWorldModel` 路线。
- pooled hidden / predicted hidden actor 路线。
- full sequence hidden 预处理和 compact reconstruction 的初步 probe。

尚未完成的关键部分：

- 明确 action head 实际依赖的 token span。
- 从 full hidden sequence 中构造 compact action context。
- TokenLearner / selector 的监督目标。
- RSSM `[h,z] -> compact action context` decoder。
- 用 compact context 接回 VLA action head 的 actor。

## 判断标准

这条路线是否成立，不应该只看 hidden MSE。更重要的是：

1. compact context reconstruction 的 cosine / scale 是否接近真实 VLA context。
2. VLA action head 在真实 compact context 上能否恢复正常 LIBERO 行为。
3. VLA action head 在 predicted compact context 上 action drift 有多大。
4. Dreamer imagination reward / value 是否随训练稳定提升。
5. LIBERO rollout 视频中是否出现接近任务目标的行为，而不是只在 loss 上下降。

## 与 semantic bottleneck 路线的区别

semantic bottleneck 路线把 VLA hidden 当作语义观测，然后训练一个小的 `z_phys`，actor/critic 直接吃
Dreamer latent。它更干净，但不直接复用 VLA action head。

RynnVLA encode 路线的目标更强：不仅用 VLA backbone 做 encoder，还希望继续复用 VLA action head。
因此它的核心难点不是 WM，而是如何从 Dreamer latent 恢复 action-head-compatible context。
