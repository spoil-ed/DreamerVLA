# TODO

## 已完成依据

- LIBERO VLA 主 ckpt 已有，移出待办：
  - `data/ckpts/VLA_model_256/libero_goal`
  - `data/ckpts/pi0_query_vla_libero_goal/epoch003_train_vla_loss1.255_success8of10.ckpt`
  - `data/ckpts/frozen_backbones/openvla_oft_libero_goal_hdf5_latest_6650`
- LIBERO WM 主训练已有可用 ckpt，移出“从零训练 WM”待办：
  - `data/ckpts/Action_World_model_512/libero_goal`
  - `data/outputs/worldmodel/rynn_dino_wm_action_hidden/m1024_d6/resume_gpu7_20260524_213615/ckpt/step_00010000.ckpt`
  - `data/outputs/worldmodel/rynn_dino_wm_action_hidden/m1024_d6/resume_gpu7_20260524_213615/ckpt/latest.ckpt`

## 当前待办

### 1. 收口 LIBERO WM ckpt

- [ ] 从现有 WM ckpt 中选定主线版本：优先检查 `m1024_d6/resume_gpu7_20260524_213615/ckpt/step_00010000.ckpt`，并和 `rynn_dino_wm_fullhidden/*/step_00002000.ckpt` 做必要对比。
- [ ] 汇总主 WM 的关键指标：`hidden_cosine_loss`、`rollout_mse`、`rollout_cosine_loss`、`reward_binary_acc`。
- [ ] 跑或整理 closed-loop / imagine diagnostics，确认 10k-step WM 不是只在训练 loss 上好看。
- [ ] 将选定 WM ckpt 归档到稳定路径，例如 `data/ckpts/recovered_best/<wm_run_tag>/`，并记录来源日志路径。

### 2. latent 表达效果消融

- [ ] 汇总已有实验结果，至少覆盖：pi0-query hidden、legacy full action hidden、time-broadcast、ResNet/TSSM/token variants。
- [ ] 把已有日志整理成对比表：训练 loss、hidden cosine、action diff、rollout 指标、下游 LIBERO success。
- [ ] 补齐还没完成的轻量消融：top-5 PC loading 可视化、shared-vs-independent decoder 对比。
- [ ] 如资源允许，再做 latent 压缩实验：例如 `stoch=8x16, deter=256`，确认是否损伤 actor 可用性。
- [ ] 根据消融结果定一个最终 latent 接口，并写清楚选择理由和对应 ckpt。

### 3. 下游验证

- [ ] 用选定 WM 跑最小 actor / PPO / WMPO smoke，确认 hidden 维度、action head type、prompt、history、action horizon 都匹配。
- [ ] 跑 LIBERO quick eval：先 1 episode/task，再 quick10。
- [ ] 如果 quick10 接近历史强基线，再跑 official50，并和历史指标对齐记录。

### 4. xllmx import 路径修复（2026-05-27 清理时发现）

- [ ] `src/preprocess/item_processor.py:19` 写的是 `from src.xllmx.data.data_reader import read_general`，但模块实际在 `src/models/xllmx/data/data_reader.py`。这是断的 import 路径，会导致 `item_processor.py` 直接执行时 ImportError。
- [ ] 决策：要么把 import 改为 `from src.models.xllmx.data.data_reader import read_general`；要么如果 `item_processor.py` 本身已经不再使用，把它和 `data_reader.py` 一起归档到 `graveyard/`。

### 5. LatentSuccessClassifier 修正（详见 [classifier_revision_plan.md](classifier_revision_plan.md)）

现状：v1/v2/v3 四版 window-level F1 ≤ 0.33，但 real hidden 上 sklearn LR 可达 0.87。问题在训练管线，不在表征也不在容量。

- [ ] **阶段 0** 基线校准（先做）
  - [ ] 写 `scripts/eval_latent_classifier_episode.py`，对 5 个旧 ckpt 重跑 episode-level F1（WMPO 协议 threshold=0.93/stride=1）
  - [ ] 落盘 LR ceiling（real hidden W=8 + class_weight=balanced），同口径 episode F1
  - [ ] **决策点**：若某个旧 ckpt 在 episode-level 上其实够用，直接归档，不进入阶段 1
- [ ] **阶段 1** 修输入分布：`WMReplayClassifierDataset` 加 `use_real_hidden=True`，新 config `v4_real_hidden.yaml`
- [ ] **阶段 2** 修容量：`LatentSuccessClassifier` 加 `head_type ∈ {linear, mlp2, transformer2}` + 强正则（wd=1e-3, dropout=0.3, label_smoothing=0.1）
- [ ] **阶段 3** WMPO-parity 评估 + 阈值 sweep + 1 epoch PPO smoke 验证 reward 信号上升、actor 不崩
