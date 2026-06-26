# 运行器控制平面说明

本文说明 DreamerVLA/RLinf 风格在线训练中，运行器（`Runner`）、工作器（`Worker`）、
真实环境、世界模型环境（`WorldModelEnv`）和学习器（`LearnerWorker`）之间的职责边界。

核心原则：

```text
Runner 是控制平面。
Worker 是执行平面。
Runner 编排阶段、版本、同步、日志和检查点。
Worker 执行模型前向、环境推进、数据缓存或优化器更新。
```

运行器不应该直接承载模型前向、环境 `step` 或优化器更新；这些工作应该留给具体工作器
或环境后端。

## 目标

真实环境和世界模型环境应走同一种控制流：

```text
真实环境:
  RolloutWorker / PolicyWorker -> EnvWorker(real env)

世界模型环境:
  RolloutWorker / PolicyWorker -> EnvWorker(world model env)

学习与同步:
  EnvWorker -> ReplayWorker / LearnerWorker
  LearnerWorker -> weight store -> RolloutWorker / WorldModelEnv
```

`hidden` / `obs_embedding` 不是策略工作器（`PolicyWorker`）的必需输出。它们只在冷启动
采集、世界模型数据构建或诊断阶段作为可选旁路字段打开。在线强化学习闭环的必要输出只有：

```text
action
```

训练路径可能额外需要：

```text
logprob
value
policy_version
```

## 工作器职责

### 推理工作器

推理工作器指 `RolloutWorker` / `PolicyWorker`。它的职责是根据观测产生动作。它不应该
知道环境是真实仿真器还是世界模型，也不应该强制输出世界模型训练用的 `hidden` 旁路字段。

必需契约：

```text
obs -> action
```

可选契约：

```text
obs -> action, logprob, value, policy_version, sidecars
```

其中：

```text
sidecars.hidden
sidecars.obs_embedding
```

只用于采集、旁路特征抽取或诊断，不是在线策略接口的硬要求。

### 环境工作器

环境工作器（`EnvWorker`）负责推进环境。它不关心背后是真实环境还是世界模型，只要求环境
后端提供一致接口：

```text
reset() -> obs, info
step(action) -> next_obs, reward, terminated, truncated, info
```

如果策略一次输出动作块，可以提供等价的动作块接口：

```text
chunk_step(action_chunk) -> obs_list, rewards, terminations, truncations, infos
```

真实环境后端负责 robosuite/LIBERO 的 `step`。世界模型环境后端负责：

```text
action + latent/condition -> next_obs, reward, done, info
```

分类器 / 成功验证器应先放在世界模型环境内部的推理快照中，因为 `reward` 和 `done` 通常
紧跟世界模型预测结果计算。只有当分类器很重，例如变成大视觉语言模型或视频奖励模型，并且
已经测量到它是瓶颈时，才值得拆成单独工作器。

### 学习器

学习器（`LearnerWorker`）负责持有训练副本并更新参数。它可以持有：

```text
policy
world_model
classifier / verifier / critic
optimizers
```

学习器从回放缓存或轨迹通道取数据，更新模型，并把新权重发布到权重存储。它是参数更新源；
推理工作器和世界模型环境只持有推理快照。

### 回放工作器

回放工作器（`ReplayWorker`）是可选的样本缓存层。on-policy PPO/GRPO 可以直接使用轨迹通道。
如果需要样本复用、陈旧度过滤、世界模型 / 分类器预热或联合训练混合数据，则使用回放工作器
更合适。

## 运行器职责

运行器拥有控制状态：

```text
global_step
policy_version
wm_version
classifier_version
rollout phase state
learner phase state
sync interval
checkpoint interval
metric namespace
```

运行器负责：

```text
1. 解析 Hydra 配置后启动 WorkerGroup
2. 初始化工作器和初始权重快照
3. 启动采样阶段
4. 启动学习阶段
5. 在 learner 更新后触发权重同步
6. 给轨迹记录 policy_version / wm_version / classifier_version
7. 聚合指标
8. 触发评测
9. 保存检查点 / resolved_config / manifest
10. 关闭工作器 / Ray runtime
```

运行器不负责：

```text
1. 直接调用模型 forward
2. 直接执行 env.step
3. 直接运行 optimizer.step
4. 直接拼写具体模型类
5. 用 assert 或代码常量决定训练维度
```

## 权重同步时序

权重同步必须发生在阶段边界，而不是每个环境步。

推荐同步粒度：

```text
LearnerWorker 完成一次或一组 update
  -> 发布 policy_version / wm_version / classifier_version
  -> Runner 在采样边界触发同步
  -> RolloutWorker 拉取 policy 快照
  -> WorldModelEnv 拉取 world_model + classifier 快照
  -> 新轨迹记录对应版本
```

不推荐：

```text
每个环境步同步 policy
每个环境步同步 world_model
classifier 每步单独远程调用
默认返回大 hidden / token 旁路字段
```

这些做法会把 Ray 进程通信、序列化、GPU/CPU 拷贝放进关键路径。

## 真实环境路径

真实环境路径的控制流是：

```text
Runner
  -> RolloutWorker.forward(obs)
  -> EnvWorker(real_env).step(action)
  -> trajectory:
       obs, action, reward, done, info, policy_version
  -> LearnerWorker.update(...)
  -> 同步 policy
```

真实环境奖励可以来自环境本身，也可以由外部奖励工作器或成功验证器补充。

## 世界模型环境路径

世界模型环境路径的控制流是：

```text
Runner
  -> RolloutWorker.forward(obs)
  -> EnvWorker(world_model_env).step(action)
  -> WorldModelEnv:
       world_model 预测 next_obs
       classifier / verifier 预测 reward 或 success
       done 由 success 或 truncation 得到
  -> trajectory:
       obs, action, reward, done, info
       policy_version, wm_version, classifier_version
  -> LearnerWorker.update(...)
  -> 同步 policy + world_model + classifier
```

世界模型环境可以运行在 latent/token 空间，不要求生成图像。它的默认返回值应保持最小：

```text
next_obs
reward
done
info
```

如果需要图像、`hidden` 或更多诊断字段，必须通过显式配置打开。

## 奖励、分类器、价值函数的边界

这三个概念不要混用：

```text
RewardModel:
  把状态、图像或 latent 映射成训练 reward。

SuccessVerifier / Classifier:
  输出 P(success) 或 terminal success score。

Critic:
  输出 V(s)，作为 advantage baseline。
```

在 outcome 路径中，`P(success)` 可以作为价值源：

```text
V(e_t) = P(success)
```

这时分类器在功能上类似价值函数，但工程命名仍建议保持 `SuccessVerifier` 或 `classifier`，
除非它确实承担通用 value baseline。

## 延迟判断

把世界模型拆到世界模型环境中，不会天然造成大幅延迟。延迟主要来自：

```text
1. 世界模型前向本身的成本
2. 是否跨进程传大 tensor
3. 是否每步同步权重
4. classifier 是否单独远程调用
5. obs 表示是否过大
```

低延迟设计：

```text
WorldModelEnv 常驻 world_model + classifier 推理快照
step/chunk_step 内部完成 next_obs/reward/done
只在采样边界同步权重
只返回最小 obs/reward/done/info
版本记录进轨迹
```

高延迟设计：

```text
PolicyWorker 每步远程调用 world_model
world_model 每步远程调用 classifier
每步 push/pull 权重
默认返回完整 hidden 旁路字段
```

因此，优先实现“世界模型环境作为环境工作器后端”。不要把世界模型前向拆成额外的
逐步 Ray 服务。

## 和 RLinf 的对应关系

RLinf 的 embodied 路径把世界模型伪环境包装成环境后端，例如 Wan/OpenSora：

```text
env_type: wan_wm
env_type: opensora_wm
```

推理工作器仍负责策略推理，环境工作器仍负责环境推进。区别只在于环境工作器内部的环境后端
是真实环境还是世界模型环境。

DreamerVLA 应采用同样的控制面：

```text
RolloutWorker 不知道 env 是否真实
EnvWorker 不知道 policy 是 VLA 还是 learned actor
Runner 通过配置选择后端并编排同步
LearnerWorker 是所有可训练权重的发布源
```
