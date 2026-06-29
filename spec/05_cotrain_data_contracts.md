# Cotrain Data Contracts

状态：current

本文固定 manual cotrain route 中 `EnvGroup`、`RolloutGroup`、`ActorGroup`、`LearnerGroup`、replay
和 WMEnv 共享的数据契约。dataclass 定义在 `dreamervla/workers/cotrain/messages.py`。

## Channel Messages

```text
ObservationMsg(
  env_rank: int,              # EnvWorker rank（含 real/wm offset）
  slot_id: int,               # 该 EnvWorker 内的 slot
  task_id: int,
  episode_id: int,
  step: int,
  obs: dict[str, Any],        # 真实 image/state obs，或 WM latent obs
  versions: dict[str, int],   # 该 env 可见的模型版本
)
```

```text
RolloutResultMsg(
  env_rank: int,
  slot_id: int,
  task_id: int,
  episode_id: int,
  step: int,
  actions: Any,                       # [chunk, action_dim]
  prev_logprobs: Any,                 # [1] 或 scalar-like chunk logprob
  prev_values: Any | None,            # 可选 [1]
  forward_inputs: dict[str, Any],
  versions: dict[str, int],
)
```

```text
TrajectoryShard(
  env_rank: int,
  slot_id: int,
  task_id: int,
  episode_ids: list[int],
  actions:        torch.Tensor,       # [1, 1, chunk, action_dim]
  rewards:        torch.Tensor,       # [1, 1, chunk]
  dones:          torch.Tensor,       # [1, 1, chunk]
  prev_logprobs:  torch.Tensor,       # [1, 1]
  prev_values:    torch.Tensor | None,# [1, 1] when present
  forward_inputs: dict[str, torch.Tensor],  # 每项 [1, 1, ...]
  versions:       dict[str, torch.Tensor],  # 每项 [1, 1]
)
```

```text
TrajectoryBatch(
  actions, rewards, dones, prev_logprobs, prev_values,
  forward_inputs: dict[str, torch.Tensor],
  versions:       dict[str, torch.Tensor],
  loss_mask:   torch.Tensor,
  task_ids:    torch.Tensor,
  episode_ids: torch.Tensor,
)
```

```text
StopMsg(reason: str)
```

EnvWorker 当前每个 slot/chunk 产出 leading `[1, 1, ...]` 的 shard。`collate_trajectory_shards`
沿 batch 维（`dim=1`）拼接，pad 到 max time steps，要求所有 shard 的 `forward_inputs`/`versions`
key 一致，并保留 chunk trailing 维，得到统一 `[T, B, ...]` 的 `TrajectoryBatch`。

## Required Forward Inputs

`forward_inputs` 必须完整到能让 ActorGroup 对**同一个 sampled action** 重算当前 policy logprob。

必需：

- `hidden`：rollout observation embedding 或 latent 输入。
- `action`：sampled policy action chunk。

存在时保留：

- `lang_emb`
- `action_token_ids`
- `input_ids`
- `attention_mask`
- `hidden_states`

ActorGroup 用 `mode="evaluate"` + `hidden` + `action`（外加上面存在的 extra key）重跑 policy 得到
`new_logprobs`。如果未来 VLA encoder 变成 trainable，仅存 detached embedding 不够：那条 route 必须
把保留训练路径所需的非 detached 输入也加进 `forward_inputs`。

## Real Env Sidecars

真实 LIBERO obs 可能不带 `obs_embedding`。OpenVLA-OFT manual route 下：

```text
real image/state obs
-> RolloutGroup OFTRolloutBundle
-> obs_embedding / lang_emb / action / logprob / forward_inputs
-> EnvWorker replay transition sidecars
```

EnvWorker 把 `forward_inputs["hidden"]` 写成 transition 的 `obs_embedding`，并带上 `lang_emb`
和各模型 `*_version`。它把 policy action 与 environment action 分开存：当
`env.real.cfg.action_postprocess=openvla_oft` 时，只有 env step 收到 postprocess 后的 gripper
动作；replay 和 ActorGroup 仍使用原始 policy action chunk。

## WMEnv Bootstrap

WMEnv 可从 replay 初始化 slot 状态：`obs_embedding`、`lang_emb`、`proprio`。bootstrap 是
best-effort——replay 为空或缺 key 时回落到 env 配置的 reset 行为。WMEnv 状态同步保持显式：

```text
WMEnvWorker.load_world_model_state(state_dict, version)
WMEnvWorker.load_classifier_state(state_dict, version)
WMEnvWorker.load_component_states({"world_model": ..., "classifier": ...}, version)   # runner 主路径
```

## Actor PPO Contract

ActorGroup 期望 chunk-level actions：

```text
batch.actions.ndim == 4
batch.actions: [time, batch, chunk, action_dim]
```

return 是 trajectory 级别：对 time 维和任意 chunk trailing 维求和。

```text
rewards [T, B]          -> returns [B]
rewards [T, B, chunk]   -> returns [B]
```

manual route 当前用 group-relative/GRPO advantage（按 `algorithm.group_size`）。value bootstrap
字段（`prev_values`、`returns`、final bootstrap）是可选的，只有未来启用 actor-critic/GAE 分支时
才变成必需。
