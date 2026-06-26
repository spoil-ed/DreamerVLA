# Online Cotrain Bracket Resolution Checklist

本文档逐条映射原始审计窗口中的方括号批注到已确定结论；其中 task-prompt
conditioning 与 WM boundary 被拆成两条代码事实，便于核验。

1. OpenVLA-OFT 是教程默认离散 VLA 配置，不是硬编码依赖；模型选择继续由 Hydra 决定。
2. Replay 可以封装为 Ray layer/worker；真实 replay 与 imagined rollout memory 语义分离。
   Online real episodes 写入 replay，imagined trajectories 不写入 replay。
3. Warmup 使用 `replay_epoch` coverage 语义。Full learning claims 需要 full warmup，
   smoke warmup 只验证链路。
4. Episode 完成条件是 env success/done 或 env-step horizon/truncation。
5. 每个 completed real episode 后更新累计 success rate 和 recent-window success rate。
6. 一个 episode 可让 replay 非空，但 full cotrain ready 需要配置的 transitions、task coverage、
   每任务 episode 数和 classifier 正负证据。
7. WM warmup 在 cold-start collection/replay initialization 产生目标数据后开始。
8. Classifier warmup 与 WM warmup 使用同一 replay pool 的高效交替 datastream，
   不要求同一 tensor batch。
9. PPO outcome signal 使用 bounded adaptive imagined rollout：`K_min=4`、`K_max=16`；
   到 max 仍无方差则跳过该起点。
10. Learned actor 真实提升应看真实 rollout/eval completed episodes。累计 success rate
    可汇总，recent-window success rate 更适合在线趋势。
11. Training artifacts 必须支持 save/resume/load；默认 torch checkpoint 可用，
    HF 兼容性按组件边界审计。
12. Debug/full 的区别是 coverage 与 update budget，不改变训练语义；full 必须覆盖 collect、
    warmup、online update 和 post-update rollout。
13. Cold-start 数据包括原始 reward/rollout HDF5 和 hidden sidecar HDF5。原始 HDF5 是真实轨迹；
    hidden sidecar 用于 warmup/learner 加速。Online replay metadata 保持 episode-level；
    offline metadata 通常在 HDF5 attrs、manifest JSON、preprocess config YAML/JSON。
14. Ray online cotrain 是主要 async 实现；no-Ray 可保持同步，除非后续明确添加 async no-Ray。
15. “learner 不更新 actor”修正为：learner 会在 imagined PPO/GRPO 有方差时更新 actor；
    它不通过直接模仿成功真实轨迹更新 actor。
16. Imagined rollout 条数由 bounded adaptive rule 决定，最少 4，最多 16。
17. 更新后的 actor 自然进入后续 online rollout 或同步到 Ray inference worker，不只是单独 eval。
18. `obs_embedding` / `actor_hidden_states` 已按代码确认经过 VLA task prompt 路径：
    online OFT extractor 由 task description 构造 prompt 并输入 `input_ids` /
    `attention_mask`；offline sidecar 保存 task prompt 或 actor token 序列。因此显式
    task conditioning 是多任务消融/稳健性增强，不是唯一 task 信息来源。
19. WM state representation boundary 已按代码确认由 config/sidecar 的
    `obs_hidden_source` 决定：`action_query` 是 action-query/action-hidden，
    `input_token_embedding` 是 projected input-token/backbone-token hidden；
    不使用未经确认的“双向注意力后的 token 层”描述。
20. Classifier imagined outcome 是 sparse success-style scoring；trajectory/window success
    形成 PPO outcome，概率/score 可作为阈值前诊断。
21. Imagined rollout 和相关学习在外部语义上保持 chunk-level。
22. RL 判断真实性能通常依赖真实环境 eval/rollout 的回报和成功率；loss 与 imagined score
    只能说明训练信号。
23. VLA action chunk 必须按顺序执行所有 low-level actions，不能只执行第一个。
24. Replay loading 不要求 DataLoader 作为语义层；只有在 prefetch/pinning 等工程收益明确时使用。
25. WM 保持当前 chunk-aware end-to-end 外部语义；内部可 step-recursive。
26. Classifier N-frame window 已按代码事实定义为 env-step frames 加可选 chunk pooling：
    replay 先取 `window * chunk_size` 个 env-step `obs_embedding`，再按
    `chunk_pool=last|first|mean` 聚合为 W 个 classifier frames。
27. Imagined trajectory 成功规则：任一 scored window 成功即可作为 sparse trajectory success；
    概率/score 诊断另存。
28. Warmup `replay_epoch` 是对 sampleable replay windows 的一次完整覆盖，内部以 batch steps 实现。
