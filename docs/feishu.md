本文档是 DreamerVLA 的说明文档，主要从现有成果和目标任务两方面入手。
[图片]
现有成果
Related Work
target的论文有：
- video-WM reference[https://arxiv.org/abs/2602.13977]
- WMPO[https://arxiv.org/abs/2511.09515]
- Dreamer[https://arxiv.org/abs/2301.04104]
- DayDreamer[https://arxiv.org/abs/2206.14176]
前两者都是在 pixel space 上训练了 worldmodel，并用一个二分类器预测 imagined trajectory 是否成功。
DayDreamer 单纯是把 Dreamer 迁移到了机器人身上
Core Idea
DreamerVLA 的核心思想是：将 VLA 的 action hidden state 作为 world model 的 latent state，并在同一个 latent space 中完成：
- imagined rollout
- success prediction
- actor optimization
与其他现有的 world model 不同，DreamerVLA 预测的是 future policy states，
现有架构
我们可以把整个架构分解为 worldmodel, VLA, RL policy 几部分来做。
world model
world model采用的架构主要有 RSSM, Transformer latent, Video prediction world model,  JEPA 等等；RSSM 由于latent 维度上限较低，不能得到过多有用的信息；Video prediction world model 实际上有最终 policy 不必要的recon信息，理论上可以优化；
Transformer latent 目前效果最好。DreamerVLA 使用的 latent state 来自 VLA 内部的 action hidden，而不是 pixel latent。VLA 可以近似理解为：
obs → VLA backbone → action hidden → action head → action
其中：
- backbone 输出是高维视觉语义 feature
- action hidden 是 action head 前的中间表示
DreamerVLA 选择 action hidden 作为 world model latent，主要有两个原因：
1. action hidden 已经参与 action generation，因此天然包含控制相关信息；
2. 相比 backbone feature，action hidden 维度更紧凑，可以理解为 VLA 自带的 bottleneck，更适合作为 predictive latent state。
Transformer latent 目前的效果：
暂时无法在飞书文档外展示此内容
VLA
原先复用的是 Rynnvla002 的 VLA，现在尝试在 openvla 上复现；

RL Policy
由于 VLA 架构可以理解为
obs - | backbone |-> action hidden -|action head|-> action
其中的 action hidden 就是 world model 的 latent state，于是我们可以基于预测的 action hidden 去得到之后的 action
obs - | backbone |-> action hidden -| world model |-> next action hidden -|action head|-> action

RL 采用的方案
1. reward：训练一个 classifier，这是多帧 action hidden 输入，判断是否实现了目标；这个 classifier 可以被称为 critic。
2. loop
  1. online rollout 收集轨迹
  2. trajectory 加入 replay buffer
  3. world model 学习 latent dynamics。因为 worldmodel 在训练过程中会比较依赖已有的数据集，导致脱离训练集的数据成为了 OOD 数据，这会导致 RL 的探索效果变差；
  4. imagined rollout 生成未来 latent trajectory。可以从已有的轨迹中任取一点作为起点，再从起点出发，closed-loop imagine 多条轨迹（即采用 VLA policy 输出的 action 作为 worldmodel 的输入 ）
  5. success critic 判断 imagined trajectory 是否成功
  6. PPO 更新 VLA policy
目标任务
之前的方案试图在 Rynnvla002 的基础上训练，并且是基于全训练集 SFT 训练的；这通常已经达到较高成功率，导致：
- PPO improvement signal过弱
- imagined exploration空间有限
- RL难以展现价值
于是，对比了目前的 related work 后，目前计划在 openvla-oft 上训练，且训练集改为 one-trajectory SFT（即每个 task 只有一条训练集轨迹）；这种方案提供：
- 极低数据 regime
- 更强的 recovery 需求
- 更清晰的 RL improvement signal
当前已实现
目前很多内容已经不是 idea，而是有了可运行的主线实现：
- OpenVLA-OFT one-trajectory cold-start workflow 已经作为当前主线；
- 已经支持 rollout collection，并保存 reward trajectory 和 action hidden sidecar；
- 已经支持基于 action hidden 的 world model warmup；
- 已经支持 success classifier warmup；
- 已经有 online replay buffer；
- 已经有 DreamerVLA manual cotrain route；
- Ray async cotrain 已经拆成 LearnerGroup / ActorGroup / RolloutGroup / EnvGroup；
- ActorGroup 负责 VLA FSDP training；
- RolloutGroup 负责 no-grad VLA inference；
- EnvGroup 同时支持 real env rollout 和 WMEnv imagined rollout；
- LearnerGroup 负责 world model / classifier update；
- LIBERO 作为当前最主要的 benchmark 路线已经接入。

当前还没完成的是完整科研验证，而不是从零实现 pipeline。

大框架实现进展
1. 数据闭环框架
我们已经把真实环境 rollout、replay buffer、reward trajectory 和 VLA action hidden sidecar 之间的 API 接口设计好了。
后续实验可以直接基于这些接口组织 one-trajectory baseline、world model warmup 和 online cotrain 的对照。

2. Latent world model 框架
已经实现以 VLA action hidden 为状态的 world model 路线，并且支持基于这些 hidden states 做 warmup 和 imagined rollout。
这部分对应 DreamerVLA 的核心假设：world model 预测的不是 pixel future，而是 future policy states。

3. Success critic 框架
已经实现多帧 action hidden 输入的 success classifier，用来判断 imagined trajectory 是否可能完成任务。
这部分目前还需要实验证明它的 score 和真实环境 success 有稳定相关性。

4. Online cotrain 框架
已经实现 manual cotrain 的整体结构：
LearnerGroup 训练 world model / classifier；
ActorGroup 训练 VLA；
RolloutGroup 做 no-grad policy inference；
EnvGroup 负责 real env 和 world model env 的 rollout。
这个结构已经接近论文方法图里的系统形态。

5. Benchmark 接入框架
LIBERO 已经作为当前主 benchmark 接入。后续 RoboTwin / CALVIN / RoboCasa 更像是 benchmark 扩展，而不是方法本身从零开始。

预计产出节奏
以 2026-07-06 为起点，比较现实的科研产出节奏是：

1. 一周内（到 2026-07-13）
预计可以得到 LIBERO goal 上的第一版完整实验结果，包括：
- OpenVLA-OFT one-trajectory baseline；
- world model / success critic 的基本质量指标；
- DreamerVLA full loop 是否有初步提升信号。

这一步的目标不是追求最终最好结果，而是判断路线是否成立。

2. 两到三周内（到 2026-07-27）
如果 goal suite 上有正向信号，可以扩展到 LIBERO object / spatial / 10，并形成第一版主实验表格。
同时需要补充 critic correlation、world model rollout drift、real-only baseline 等关键对照。

3. 四到六周内（到 2026-08-17）
预计可以形成较完整的 paper evidence：
- LIBERO 多 suite 结果；
- latent choice ablation；
- real-only vs imagined PPO 对比；
- failure analysis；
- 初步 model family / benchmark 扩展计划。

4. 后续扩展
在 OpenVLA-OFT + LIBERO 结果稳定后，再扩展到 StarVLA / π 系列 / OpenVLA 系列 / GR00T 系列，以及 RoboTwin / CALVIN / RoboCasa。
这部分更适合作为第二阶段实验，用来证明 DreamerVLA 不是只针对一个 backbone 或一个 benchmark 的 special case。

TODO
Baseline
[] 系统报告 OpenVLA-OFT one-trajectory SFT baseline
[] 在 LIBERO 4 suites 上复现 one-trajectory setting
[] 建立 low-data RL benchmark，并报告 per-task success 和 failure mode

DreamerVLA
[] 验证 VLA action hidden latent rollout 的长期稳定性
[] 验证 success critic score 和真实环境 success 的相关性
[] 验证 DreamerVLA imagined PPO training 是否带来真实环境提升
[] 对比 real-only online finetune，证明提升不是来自更多训练步数

Experiments
[] 在 LIBERO 4 suites 上训练
[] 在 RoboTwin 上测试
[] 在 CALVIN 上测试
[] 在 RoboCasa 上测试
[] 对比 planning-based world model 方法
[] 对比不同 latent state 方案

后续实验扩展
模型方向：
[] StarVLA 系列
[] π 系列
[] OpenVLA 系列
[] GR00T 系列
[] 其他通用 VLA / robot foundation model

benchmark 方向：
[] LIBERO
[] RoboTwin
[] CALVIN
[] RoboCasa

扩展目标：
先在 OpenVLA-OFT + LIBERO goal 上完成 clean result，再扩展成 model family × benchmark 的系统比较。
最终需要回答：DreamerVLA 是只适用于 OpenVLA-OFT，还是可以作为更通用的 VLA world-model RL 方法。

Infrastructure

科研进展 OKR
当前判断
DreamerVLA 目前的科研状态是：核心假设清楚，方法路径基本成型，但还处在等待关键实验证明的阶段。
它要证明的不是一个新的训练框架能跑起来，而是：
VLA action hidden state 可以作为一种 policy-aware latent world model state，并在低数据条件下产生有用的 imagined improvement signal。

Objective 1
证明 DreamerVLA 在 low-data VLA setting 下有真实科研价值。

KR：
- 建立 OpenVLA-OFT one-trajectory benchmark，确认 baseline 成功率不饱和，且仍有可学习空间。
- 在 LIBERO goal 上证明 DreamerVLA full loop 相比 SFT baseline 有真实环境提升。
- 控制 real env interaction、gradient steps 和 replay size，证明提升不是来自更多 finetune。

当前卡点：
还没有真实 long-run 结果证明 full loop 能稳定提升 one-trajectory baseline。

Objective 2
证明 action hidden latent 是适合 VLA world model 的中间表示。

KR：
- 证明 world model 能在 action hidden space 中预测 future policy states，而不是快速 drift。
- 证明 success critic 对 imagined trajectory 的评分和真实环境 success 有相关性。
- 完成 action hidden vs backbone feature / pixel latent / no-WM 的关键 ablation。

当前卡点：
action hidden 的优势还缺实验证据；success critic 作为 imagined reward 的可信度还没有被证明。

Objective 3
形成一条可以支撑 paper 的完整证据链。

KR：
- 在 LIBERO goal 上先得到 clean result，再扩展到 LIBERO 4 suites。
- 后续覆盖 StarVLA / π 系列 / OpenVLA / GR00T 等模型方向。
- benchmark 扩展到 LIBERO / RoboTwin / CALVIN / RoboCasa。
- 给出 failure analysis，说明方法在哪些任务上有效、在哪些情况下会失败。
- 与 planning-based world model 或 pixel-space world model 方法做对比，说明 DreamerVLA 的差异和优势。

当前卡点：
目前工程组件已经基本齐备，缺少从 baseline、WM/critic quality 到 imagined PPO improvement 的闭环实验结果。

一句话总结
DreamerVLA 当前最重要的科研任务，是把“action hidden 可以作为 VLA world model latent”这件事从方法假设变成实验证据。
