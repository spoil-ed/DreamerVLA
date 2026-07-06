本文档是 DreamerVLA 的阶段性进展报告，主要从已交付成果、实测性能、目标与预期时间表三方面入手。
[图片]
项目定位与核心思想
DreamerVLA 提出并已经实现了一条与现有 world-model RL 路线正交的技术主张：不在 pixel space 重建世界，而是直接把 VLA 的 action hidden state 作为 world model 的 latent state，在同一个 latent space 中完成：
- imagined rollout
- success prediction
- actor optimization
与现有 world model 不同，DreamerVLA 预测的是 future policy states：world model 学的不是"世界接下来长什么样"，而是"策略接下来会处于什么状态"——这是一个天然贴合 policy optimization 的预测目标。
Related Work
target 的论文有：
- WoVR[https://arxiv.org/abs/2602.13977]
- WMPO[https://arxiv.org/abs/2511.09515]
- Dreamer[https://arxiv.org/abs/2301.04104]
- DayDreamer[https://arxiv.org/abs/2206.14176]
前两者都在 pixel space 上训练 world model，并用二分类器/reward model 判断 imagined trajectory 是否成功。WoVR 用的是 5B 视频扩散模型、每个 action chunk 要跑 5 步去噪推理；DreamerVLA 的 latent-only world model 每步只是一次轻量 latent forward，推理开销低一个数量级。
pixel-space 路线已经替我们验证了命题的前半句（WoVR 论文自报，LIBERO one-trajectory SFT 设定）：
- Spatial：63.6% → 84.2%（+20.6）
- Object：36.4% → 80.8%（+44.4）
- Goal：48.2% → 77.4%（+29.2）
- Long：13.8% → 35.8%（+22.0）
- 平均：40.5% → 69.5%（+29.0）；同设定下 online GRPO 只有 44.6%、WMPO 只有 50.9%
即"world model RL 能在极低数据 regime 带来 ~+29 点成功率"已被证明。DreamerVLA 的命题是后半句：用便宜一个数量级的 world model，拿到同一级别的提升。
已交付成果
1. 完整训练闭环已建成，一条命令跑通全程
collect rollouts → seed replay → warmup world model + success classifier → online cotrain → eval，全流程由统一 launcher 驱动：
python -m dreamervla.launchers.coldstart_warmup_cotrain mode=ray task=goal
支持断点续跑（cotrain_phase=warmup/online 分段恢复）、多 suite（goal/object/spatial/10）、多 GPU profile 一键切换。
2. RLinf 级别的分布式训练工程
主线 Ray 异步 cotrain 采用与 RLinf embodiment 对齐的四组拓扑：
- LearnerGroup：world model + success classifier 在线更新
- ActorGroup：VLA FSDP 训练（group-relative PPO，group_size=8）
- RolloutGroup：no-grad 策略推理副本，增量 patch 权重同步
- EnvGroup：真实 LIBERO env 与 latent world-model env 统一走同一条轨迹管线
对照 RLinf 的 9 个推理优化点：7 项已对齐或本质等价，1 项机制不同但语义等价，1 项在 roadmap 上；另有 1 项（world model env 的 stepping 成本）DreamerVLA 结构性领先——latent-only，从不解码像素。
imagination 已实现为 WM-as-env：world model 被包装成 gym 式环境，imagined rollout 和真实 rollout 走完全相同的采样、组装与 replay 管线，chunk 级批量前向。
3. 实测性能：imagine 吞吐已反超 pixel-space 参考实现
与 WoVR 参考实现同机同卡数（2 卡对 2 卡）计时对照：
- WoVR（5B 视频扩散 WM + reward model，32 envs，GPU 独占）：~94 imagined env-steps/s
- DreamerVLA（latent WM，且同卡还被其他任务抢占 ~20GB 显存 + 间歇 90% 利用率）：~99 imagined env-steps/s
即：用轻一个数量级的 world model、在不利的资源条件下，imagine 吞吐已经略优于对照方案。瓶颈定位（env-worker CPU 序列化，payload 线性）经三轮 A/B 实测闭环验证，优化方向明确且已兑现（消除 hidden 回传后 +36%）。
wall-time 口径：WoVR 2 卡单个 global step ≥ 45-50 分钟；DreamerVLA 6 卡预估 ~25-30 分钟/global step，且每步消耗 4 倍的 imagined env-steps（1024 条 × 512 步 vs 512 条 × 256 步）。
4. 基线锚点已互相印证
我们独立复测的 OpenVLA-OFT one-trajectory SFT baseline（libero-goal traj1 ≈ 0.50 success_once）与 WoVR 论文自报的 48.2% 几乎一致——起点水位对齐，后续对比完全同口径、公平可信。
5. 工程质量护栏
- 全量单测 1328 passed / 0 failed，覆盖 config 校验、消息契约、collective 对齐、回退路径
- 统一 run root 产物：resolved_config / run_manifest / checkpoints / tensorboard+wandb / videos / diagnostics
- 每个 global step 用真实 LIBERO episode 评估成功率（eval/* 与 classifier 打分严格分离），训练曲线即评估曲线，无需事后补测
目标与预期时间表
靶子非常明确：LIBERO goal suite，one-trajectory SFT 起点 ~48-50%，WoVR 立下的标杆是 77.4%（+29.2 点）。
预算换算：WoVR 全程消耗 2500 条真实环境轨迹/suite；DreamerVLA 每个 global step 消耗 8 条真实轨迹 + 1024 条 imagined 轨迹，即 ~310 个 global step 达到 WoVR 同预算点位，6 卡 ≈ 6 天连续训练。
里程碑（以 2026-07-03 为 T0）：
1. T+1 周（7 月中旬）：goal suite 长训启动并出首条收敛曲线。由于每个 global step 都有真实 SR 读数，第一周内即可判断 RL improvement signal 是否出现（SR 脱离 0.50 基线上行）。当前 in-flight 的多卡 actor collective 对齐改动是长训前最后一块拼图，本周落地。
2. T+2-3 周（7 月下旬）：跑满 WoVR 同预算点位（~310 global steps），产出与 WoVR Table 2 严格同口径的 goal suite 对比数字。这是第一个可对外报告的核心结果。
3. T+4-6 周（8 月）：扩展到 4 suites（goal/object/spatial/10）全量，补 WMPO / online GRPO 基线对比与 latent 方案消融（action hidden vs backbone feature vs pixel latent）。
4. T+8 周（8 月底-9 月初）：整理论文级实验材料（仓库已建 CoRL/NeurIPS 论文目录），完成方法、系统、实验三条线的成稿素材。
风险与依赖：
- GPU 可用性是唯一的硬性 gate（8×H100 间歇可用）；训练本身支持分段恢复，可利用碎片化 GPU 窗口
- 大 batch RL 更新的显存风险已有 micro-batch 预案（实测 82GB → ~11GB）
- 若 goal suite 首轮涨幅不及预期，调参面已收敛到 algorithm.*（clip/entropy/lr）与 rollout 预算配比，均为 config 级改动
Roadmap
Baseline
[x] 完成 OpenVLA-OFT one-trajectory SFT baseline（goal ≈ 0.50，与 WoVR 自报 48.2% 互证）
[x] 在 LIBERO 上复现 one-trajectory setting
[x] 建立 low-data RL benchmark（RLinf 并行评估 + 同机双方案计时对照）
DreamerVLA
[x] 完成 VLA action hidden latent rollout（LatentWorldModelEnv，latent-only）
[x] 完成 success critic（classifier warmup + 在线更新，latent 窗口打分）
[x] 构建 online rollout + replay loop（向量化 EGL rollout + Ray 异步四组拓扑）
[x] 完成 imagined PPO training 代码路径（group-relative PPO + micro-batch 显存控制）
[] 真实 OFT/LIBERO 长训收敛验证（对照 0.50 基线，目标 77.4%）← 当前唯一关键路径
Experiments
[] goal suite 同预算对比 WoVR（T+2-3 周）
[] 4 suites 全量 + WMPO/GRPO 基线对比（T+4-6 周）
[] 对比不同 latent state 方案（action hidden vs backbone feature vs pixel latent）
[] CALVIN 迁移（远期：需要新 env 接入，当前 LIBERO 是唯一稳定 env 面）
Infrastructure
[x] Hydra 分组配置 + Runner 模式统一入口（experiment=<name> task=<suite>）
[x] 单机 Ray 异步后端（scheduler / placement / channel / weight-syncer 全栈自研）
[x] FSDP1/FSDP2 训练栈 + 增量 patch 权重同步
[x] 双格式 checkpoint（torch/HF）+ 数据分片轮转 + 分段恢复
[x] 统一可观测性：train/ eval/ env/ rollout/ time/ 指标命名空间 + 每步真实 SR 曲线
[] env bootstrap overlap（对照参考实现的最后一个推理优化点，收益预估为每步头部串行 reset 时间）
