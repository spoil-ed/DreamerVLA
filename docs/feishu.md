# DreamerVLA 主线说明

## 核心目标

DreamerVLA 当前只维护一条 OpenVLA-OFT one-trajectory 主线：

```text
collect rollouts
  -> seed replay
  -> warm up world model + success classifier
  -> online cotrain
  -> real LIBERO evaluation
```

这条路线把 OpenVLA-OFT 当前帧的投影视觉输入 token 当作 world-model
observation。外部数据契约固定为：

```text
obs_hidden_source = input_token_embedding
obs_embedding     = [T, 256, 4096]
token_count       = 256
token_dim         = 4096
wm_obs_dim        = 1,048,576
history           = 1
image_count       = 1
action_chunk      = 8
```

OpenVLA decoder 内部会产生离散动作位置，但这些位置只负责 action decoding，
不会写入 sidecar，也不是 world model 或 classifier 的观测。

## 为什么选择 input token

输入 token 保留了 OpenVLA 的空间 token 网格和 4096 维语义通道，同时位于动作
decoder 之前，因此能作为 world model、classifier 和 actor 共享的状态边界。相比给
不同组件提供不同中间表示，统一契约有三个直接收益：

1. collection、offline preprocess 和 online rollout 生成完全相同的 sidecar；
2. world model 与 classifier 的独立训练和 online update 复用同一更新函数；
3. 旧数据在进入 replay 或 optimizer 前即可通过元数据和 HDF5 形状明确拒绝。

## 组件

### World model

Chunk world model 接收 `[B,T,256,4096]` token 序列。动作、语言和本体信号在
模型内部各自投影后拼接到 token 通道。模型使用最近三帧历史，自回归预测未来
token grid；多步训练会把自己的预测放回历史，从而直接测量 free-running drift。

### Success classifier

Classifier 在多帧 `[256,4096]` token grid 上使用空间 Transformer head，输出二元
成功 logit。classifier warm-up、sync cotrain 和 Ray learner 使用同一个 update
实现，并共享 WMPO 平衡采样协议。

### Actor

Actor bridge 让内部离散动作位置 cross-attend 256 个预测输入 token，然后复用
OpenVLA LM head 产生动作 token。策略更新使用 token-level log probability、组相对
优势、clipped ratio、entropy 和 reference KL。

## 数据与校验

真实 rollout 写入：

```text
${DVLA_DATA_ROOT}/collected_rollouts/<suite>/reward/
${DVLA_DATA_ROOT}/collected_rollouts/<suite>/hidden/
${DVLA_DATA_ROOT}/collected_rollouts/<suite>/collection_manifest.json
```

`hidden/` 是通道角色名；其中的具体内容只能是 input-token sidecar。系统在以下边界
执行严格校验：

- collection resume；
- offline replay seed；
- standalone world-model train；
- standalone classifier train；
- sync/Ray warm-up。

校验覆盖 `preprocess_config.json`、全部 HDF5 shard、全部 demo、数据集键、rank、
尾部维度以及 reward/sidecar 对齐关系，不做自动转换或兼容回退。

## 同步与异步训练

同步路线由 `OnlineCotrainPipelineRunner` 完成 warm-up 和 online cotrain。Ray 手动
路线将职责拆成：

- `ActorGroup`：FSDP actor training；
- `RolloutGroup`：no-grad policy inference 和 actor weight pull；
- `EnvGroup`：real/imagined stepping 与 trajectory assembly；
- `LearnerGroup`：world model/classifier update 与权重发布。

两条路线使用相同的 replay、模型和 update contract。最终质量只用真实 LIBERO
success rate 判断；想象环境的 classifier score 只是训练信号。

## 当前验证标准

主线改动完成前必须同时满足：

- goal/object/spatial/10 四个 suite 的 collect、sync cotrain、Ray cotrain 配置均能
  完整 compose；
- classifier 与 world model 分别完成至少一次真实 optimizer update 并写出 checkpoint；
- synthetic 非主线 sidecar 在 resume 和 train 之前被拒绝；
- focused tests、完整 pytest、compileall 和 `git diff --check` 全部通过；
- 只有完成上述验证后才允许 push 到 `main`。
