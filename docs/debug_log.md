## 2026.6.27

**问题**

存储逻辑出现问题：没有保存全局步数，没有保存 cotrain 阶段 online collect 而 rebuff 的内容。这部分重要是因为需要 resume 训练，且训练 WM 和 cls 需要足够充分的邻近历史数据。

**解决方案**

按照
```
pace_replay/
  manifest.jsonl
  episodes/
    task_07/
      global_step000120_success_True/
        ep_000001.h5
        ep_000002.h5
```
落盘最近 K 个 global_step 的数据，且标注成功失败。逻辑存储所有全局时间步，包括

**问题**

表达时间步的名称过多，global_step / update_step / learner_updates:等等

**解决方案**

统一字段
1. 时间步
- global_step
- env_step
2. Episode 身份
- task id: 0
- episode_id: 6
- init_state_index: 6
3. Episode 结果
- success: ture
- complete: ture

**疑问**

PPO 收集多少条数据？manual cotrain 的配置如何？
  manual cotrain 这里严格说是 GRPO，不是带 critic 的 PPO：

  adv_type: grpo
  loss_type: actor
  critic.use_critic_model: False

  但 loss 是 PPO-style clipped actor loss，关键参数：

  - clip_ratio_low: 0.2
  - clip_ratio_high: 0.28
  - clip_ratio_c: 3.0
  - normalize_advantages: True
  - kl_beta: 0.0
  - entropy_bonus: 0
  - gamma: 0.99
  - gae_lambda: 0.95
  - reward_type: action_level
  - logprob_type: token_level
  - entropy_type: token_level
  - filter_rewards: True
  - rewards_lower_bound: 0.5
  - rewards_upper_bound: 4.5

  模型和采样
  OpenVLA-OFT：

  - model_type: openvla_oft
  - precision: bf16
  - action_dim: 7
  - num_action_chunks: 8
  - max_prompt_length: 128
  - unnorm_key: libero_goal_no_noops 等按 suite 切换

  采样：

  - do_sample: True
  - temperature_train: 1.6
  - temperature_eval: 1.6
  - top_k: -1
  - top_p: 1.0
  - max_length: 1024

  Wan Env 参数
  训练 env：

  - env_type: wan_wm
  - total_num_envs: 64
  - max_episode_steps: 256
  - max_steps_per_rollout_epoch: 256
  - group_size: 8
  - num_inference_steps: 5
  - enable_offload: True
  - reset_gripper_open: True

  Wan 本身：

  - chunk: 8
  - condition_frame_length: 5
  - num_frames: 13
  - image_size: [256, 256]
  - enable_kir: True
  - reward model: TaskEmbedResnetRewModel

**问题**

进度条可视化。

**解决方案**

进度条需要输出：
当前阶段，global_step，当前rollout的局部step，learner学习的step。
