# Dreamer-VLA 内部技术文档

## 一句话总结

用 VLA 提供语义表征，用 Dreamer 风格的 world model 学物理动力学，在 imagination 中训练 RL policy。

---

## 1. 动机

| 方法 | 优点 | 缺点 |
|------|------|------|
| **纯 VLA** | 语义理解强，行为合理 | 只能模仿，无法 imagination，难做 long-horizon |
| **纯 Dreamer** | world model + imagination + RL 闭环 | 感知弱，语义能力差，sample inefficient |

**核心判断**："理解世界"和"预测世界"是两种能力，不应强行让一个模型承担。

---

## 2. 整体架构

```
obs_t (image, proprio, text)
        │
        ▼
┌─────────────────────────┐
│   VLA Encoder (frozen)  │  ← 提供语义表征
└─────────────────────────┘
        │
        │ z_sem (4096-dim, 非 Markov)
        ▼
┌─────────────────────────┐
│  Bottleneck Projection  │  ← 筛选可预测信息
└─────────────────────────┘
        │
        │ z_phys (32-dim, stochastic, Markov)
        ▼
┌─────────────────────────┐
│     World Model         │  ← 学习动力学
│  ├─ dynamics            │
│  ├─ reward head         │
│  └─ continue head       │
└─────────────────────────┘
        │
        │ imagined rollouts
        ▼
┌─────────────────────────┐
│    Actor / Critic       │  ← 在 imagination 中训练
└─────────────────────────┘
```

**梯度规则**：

| 模块 | 梯度 |
|------|------|
| VLA Encoder | ❌ freeze |
| Bottleneck | ✅ train |
| World Model | ✅ train |
| Actor/Critic | ✅ train |

---

## 3. 两种实现方案

### 方案 A：带 RSSM（推荐第一版）

适用于：partial observability 较强、需要历史信息的场景

```
           z_sem_t
              │
              ▼
        ┌───────────┐
        │ Bottleneck│ → μ, σ
        └───────────┘
              │
              ▼ sample
           z_phys_t ─────────────┐
              │                  │
              ▼                  ▼
        ┌───────────┐      ┌──────────┐
        │   RSSM    │      │ Decoder  │
        │  h_t, ẑ_t │      │ (可选)   │
        └───────────┘      └──────────┘
              │
              ├── reward_head(h_t, z_t) → r̂
              ├── continue_head(h_t, z_t) → ĉ
              └── dynamics: h_{t+1} = f(h_t, z_t, a_t)
```

**RSSM 结构**：
```python
# Deterministic path
h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])

# Stochastic path
prior:     ẑ_t ~ p(z | h_t)           # imagination 时用
posterior: z_t ~ q(z | h_t, z_sem_t)  # 训练时用

# KL loss 对齐 prior 和 posterior
```

**Loss**：
```
L = L_reward + L_continue + β * KL_dynamics + γ * KL_bottleneck
```

---

### 方案 B：不带 RSSM（简化版）

适用于：任务较简单、VLA 表征足够 Markov 的场景

```
           z_sem_t
              │
              ▼
        ┌───────────┐
        │ Bottleneck│ → μ, σ
        └───────────┘
              │
              ▼ sample
           z_phys_t
              │
              ├── dynamics_head(z_t, a_t) → z_{t+1}
              ├── reward_head(z_t) → r̂
              └── value_head(z_t) → V̂
```

**核心假设**：z_sem 已经足够 Markov，不需要额外的 recurrent state。

**Loss**：
```
L = MSE(z_pred, z_true) + MSE(r_pred, r_true) + γ * KL_bottleneck
```

---

### 方案对比

| | 方案 A (RSSM) | 方案 B (无 RSSM) |
|--|---------------|------------------|
| **复杂度** | 中 | 低 |
| **处理 POMDP** | 强 | 弱 |
| **Imagination 稳定性** | 高 | 需验证 |
| **实现难度** | 中 | 低 |
| **推荐场景** | 第一版、长 horizon | 快速验证、短 horizon |

---

## 4. 关键模块细节

### 4.1 Bottleneck（核心）

**不是简单降维，是信息筛选**：只保留"可被 dynamics 预测"的信息。

```python
class StochasticBottleneck(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=32):
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, z_sem):
        mu = self.fc_mu(z_sem)
        logvar = self.fc_logvar(z_sem)
        z_phys = mu + torch.randn_like(mu) * (0.5 * logvar).exp()

        # KL regularization
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
        return z_phys, kl
```

**维度建议**：
- Manipulation: 16-32
- Locomotion: 32-64
- 原则：**宁小勿大**

### 4.2 World Model

**方案 A 的 RSSM**：
```python
class RSSM(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, action_dim=7):
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.prior = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ELU(),
                                   nn.Linear(64, 2 * latent_dim))
        self.posterior = nn.Sequential(nn.Linear(hidden_dim + latent_dim, 64), nn.ELU(),
                                       nn.Linear(64, 2 * latent_dim))

    def forward(self, h_prev, z_prev, a_prev, z_sem=None):
        # Deterministic
        h = self.gru(torch.cat([z_prev, a_prev], -1), h_prev)

        # Prior (for imagination)
        prior_mu, prior_logvar = self.prior(h).chunk(2, -1)

        # Posterior (for training, needs z_sem)
        if z_sem is not None:
            post_mu, post_logvar = self.posterior(torch.cat([h, z_sem], -1)).chunk(2, -1)
            z = post_mu + torch.randn_like(post_mu) * (0.5 * post_logvar).exp()
            kl = kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar)
        else:
            z = prior_mu + torch.randn_like(prior_mu) * (0.5 * prior_logvar).exp()
            kl = 0

        return h, z, kl
```

**方案 B 的简化 Dynamics**：
```python
class SimpleDynamics(nn.Module):
    def __init__(self, latent_dim=32, action_dim=7, hidden_dim=256):
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], -1))  # 残差
```

### 4.3 Reward / Value Heads

```python
class RewardHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        # state = h_t (RSSM) 或 z_t (简化版)
        return self.net(state)
```

---

## 5. 训练流程

### Phase 1：训练 World Model

```python
for batch in replay_buffer:
    # 1. VLA 编码 (frozen)
    with torch.no_grad():
        z_sem = vla_encoder(obs)

    # 2. Bottleneck
    z_phys, kl_bn = bottleneck(z_sem)

    # 3. World model forward
    # 方案 A: h, z, kl_dyn = rssm(h_prev, z_prev, a_prev, z_phys)
    # 方案 B: z_next = dynamics(z_phys, action)

    # 4. Prediction
    r_pred = reward_head(state)

    # 5. Loss
    loss = F.mse_loss(r_pred, reward) + β * kl_dyn + γ * kl_bn
```

### Phase 2：Imagination 训练 Actor-Critic

```python
# 从真实状态出发
state = get_initial_state(batch)

imagined_states, rewards, values = [], [], []

for t in range(horizon):
    action = actor(state)

    # Imagine next state
    # 方案 A: h, z, _ = rssm.imagine(h, z, action)
    # 方案 B: z = dynamics(z, action)

    r = reward_head(state)
    v = value_head(state)

    imagined_states.append(state)
    rewards.append(r)
    values.append(v)

# λ-returns
returns = compute_lambda_returns(rewards, values, gamma=0.99, lambda_=0.95)

# Actor loss: maximize returns
actor_loss = -returns.mean()

# Critic loss: predict returns
critic_loss = F.mse_loss(torch.stack(values), returns.detach())
```

---

## 6. 数据准备

**Replay Buffer 存储**：
```python
{
    'obs': ...,        # 原始观察（用于 VLA 编码）
    'action': ...,     # (T, 7)
    'reward': ...,     # (T,)
    'done': ...,       # (T,)
}
```

**不要只存 z_phys**：因为 bottleneck 也需要训练。

---

## 7. 实验验证

**Success Criteria**：

1. Imagination rollout 稳定（不发散）
2. 预测 reward ≈ 真实 reward
3. RL policy 超越纯 VLA imitation
4. 消融实验：移除 VLA / imagination 性能下降

**第一版配置**：

| 组件 | 选择 |
|------|------|
| VLA | RynnVLA-002, frozen |
| Bottleneck | Gaussian VIB, 32-dim |
| World Model | RSSM (方案 A) |
| Decoder | 预测 reward，**不重建 image** |
| Actor/Critic | 标准 Dreamer |

**不要做**：
- ❌ Joint train VLA
- ❌ Reconstruct pixels
- ❌ Discrete latent
- ❌ Multi-task language conditioning

---

## 8. 参考

- RynnVLA: https://github.com/alibaba-damo-academy/RynnVLA-002
- DreamerV3: https://arxiv.org/abs/2301.04104
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
