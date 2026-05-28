# LatentSuccessClassifier 修正版训练方案

更新时间: 2026-05-25

## 0. 现状一句话

四版 `LatentSuccessClassifier` 全部 plateau 在 window-level F1 ≤ 0.33;同时已验证 **real hidden 上 sklearn LogisticRegression 可达 F1 ≈ 0.87**。差距来自训练管线,不是表征也不是容量。需要从评估口径 → 输入分布 → 模型容量 三处一起修。

## 1. 已有 ckpt 实测一览

源数据: `data/outputs/dreamervla/outcome_classifier/libero_goal/*/train_log.jsonl` 的 `best_val` 字段(window-level,sweep 阈值取最大 F1)。

| run | dataset | hidden 来源 | pos:neg | best F1 | best thresh | 备注 |
|---|---|---|---|---|---|---|
| `v1_demo_only` | shards | **real** | demo end + earlier | 0.151 | 0.96 | 9k step plateau |
| `v2_wm_replay` | replay | **imagined** (`imagine_all`) | swap-neg only | **0.328** | 0.48 | 13500 step 峰值,后回落到 0.197 |
| `v3b_finish_step` | replay+failure | imagined | swap+failure ~1:3 | 0.111 | 0.93 | 20k step |
| `v3b_swap_plus_failures` | replay+failure | imagined | swap+failure ~1:3 | 0.103 | 0.56 | 20k step |
| **sklearn LR ceiling** | demo only | **real** | balanced W=8 | **~0.87** | — | CLAUDE.md 已记录 |
| WMPO VideoMAE (参考) | RGB video | — | — | ~0.20 | 0.93 | `best_videomae_f10.1989_th0.93.pth` |

**唯三的现象**:

1. v1 (real hidden, demo only) F1 0.15 vs LR 0.87 — **同样的 real hidden,容量更大反而差 5×** → 过拟合。
2. v2 (imagined) F1 0.33 比 v1 高 — 因为训练分布更贴 PPO 推理分布;但仍远低于 LR 上限,且与训练步数非单调(13500 峰 → 14500 跌一半)。
3. v3b (replay + failure) F1 比 v2 还低 — class balance 改了,但训练目标本身就有问题,任何配方调整都救不回来。

## 2. 三个根因

按可观测证据排序,从最确定到最推测:

### R1. window-level F1 是错的主指标(确定)

WMPO 自己的 VideoMAE 在 RGB 视频上 window F1 也只有 0.20。WMPO 通过 `predict_success(threshold=0.93, stride=1)` + episode-level "any-positive" 聚合把它兑成可用信号。我们一直在跟一个内禀偏低的指标卷,得到的"best ckpt"未必对下游 PPO 最好。

→ **必须切到 episode-level 评估**才能跟 WMPO 同口径比较、才能用到 actor-critic 上。

### R2. v2/v3 训练输入是 WM 想象,不是真 hidden(确定)

`WMReplayClassifierDataset.__getitem__` 走的是 `ChunkAwareRynnDinoWMWorldModel.imagine_all()` 输出。CLAUDE.md 已实证 `cos(real, imagined)` 从 episode 开头 0.987 漂到结尾 0.80 —— **正样本窗口在 episode 末端,正是漂移最大处**。

→ 这等于在用一个分布偏移的输入训练分类器再去判定 real hidden。线性可分性丢了。

### R3. 130 M Transformer 容量过剩(高置信度推测)

LR (~36k 参数)上 F1 0.87,意味着 35840-d 隐空间在 W=8 窗口上几乎线性可分。8 层 16 头 Transformer 的归纳偏置不需要;在 small dataset(401 正 / 39142 负 in v1, ~400+67 per epoch in v3)上反而过拟合 — v3b 训练 loss 在 20k 步降到 0.005,验证 F1 仍 ~0.1。

→ 减容量 / 强正则 / 走 LR-style head。

## 3. 修正方案

### 阶段 0: 基线校准(< 1 day, **必须先做**)

目的: 把所有比较拉到同一口径下,避免下面任何改动被 "window-level 上下抖动" 噪声盖住。

- [ ] **0.1 写 episode-level 评估脚本** `scripts/eval_latent_classifier_episode.py`
  - 输入: 任一 `LatentSuccessClassifier` ckpt + LIBERO HDF5 + real action-hidden sidecar
  - 协议: 完全复制 `LatentSuccessClassifier.predict_success(threshold=0.93, stride=1)` → 取 episode 是否 `complete` + `finish_step` → 算 episode-level F1 / precision / recall / mean abs error of finish_step
  - 跑五个现有 ckpt(v1, v2, v3b_finish_step, v3b_swap_plus_failures, v3_with_failures step 6200)+ 一条 LR baseline(real hidden W=8) ckpt
  - 产出 `data/outputs/dreamervla/outcome_classifier/_compare_v0/episode_eval.json` + markdown 表
  - **决策点**: 如果某个现有 ckpt 在 episode-level 上其实够用,直接归档,不进入阶段 1。

- [ ] **0.2 LR ceiling 落盘** `scripts/train_logreg_classifier.py`
  - 训练 sklearn `LogisticRegression(C=0.01, class_weight="balanced")` on real hidden flatten W=8
  - 包成一个能被 `predict_success` 接口替换的 PyTorch 模块(只是 linear head),走同一套 episode eval
  - 这成为 **阶段 1 之后所有 v4 模型必须超过的下限**

### 阶段 1: 修输入分布 — v4_real_hidden

- [ ] **1.1 新数据集** `dreamer_vla/dataset/wm_replay_classifier_dataset.py::WMReplayClassifierDataset`
  增加 flag `use_real_hidden: bool = True`(默认开),走到 `imagine_all` 那条分支改成直接从 sidecar hidden 取窗口。保留 `use_real_hidden=False` 走 imagine_all 作为消融。
- [ ] **1.2 新 config** `configs/wmpo_classifier_libero_goal_v4_real_hidden.yaml`
  - 复用 v3b 的 swap+failure 1:3 配比
  - `wm_replay.use_real_hidden: true`
  - 其余照搬 v3b
- [ ] **1.3 训练入口** 复用 `scripts/train_latent_success_classifier_v3.py`,加一个 `--config configs/wmpo_classifier_libero_goal_v4_real_hidden.yaml` 命令
- [ ] **1.4 评估**: 同时记 window-level(老指标,sanity)和 episode-level(新主指标)
- [ ] **1.5 决策门**: episode F1 应 ≥ LR baseline ×0.9。若没达到 → 不是输入分布问题,跳 R3。

### 阶段 2: 修容量 + 正则 — v5_linear_head / v5_mlp2

- [ ] **2.1 模型选项** 在 [latent_success_classifier.py](../dreamer_vla/models/reward/latent_success_classifier.py) 加 `head_type: Literal["transformer8", "mlp2", "linear"]`,linear 走 `nn.Linear(latent_dim*W, 2)`,mlp2 走 `[Linear(L*W, 1024), GELU, Linear(1024, 2)]`
- [ ] **2.2 跑三档容量** v5_linear、v5_mlp2、v5_transformer2(num_layers=2 而非 8) 在阶段 1 已通过的数据集上各训一遍
- [ ] **2.3 强正则** 全部开 `weight_decay=1e-3`(当前 1e-4)+ `dropout=0.3`(当前 0.1) + `label_smoothing=0.1`
- [ ] **2.4 决策门**: pick 最高 episode F1,要求达到 / 超过 LR ceiling 才算修通

### 阶段 3: 修评估口径 + 下游验证

- [ ] **3.1 WMPO-parity 评估** `scripts/eval_latent_classifier_episode.py` 加 `--protocol wmpo`,严格复制 WMPO `robwm_rollout.py::predict_success`:`threshold=0.93`, `stride=1`, `min_steps>=W`, `any-positive` 聚合
- [ ] **3.2 阈值 sweep** 对最终 ckpt 在 [0.5, 0.99] 上 sweep,记录 (precision, recall, finish_step MAE) 曲线
- [ ] **3.3 PPO smoke** 把 v5 winner 接到 `dreamer_vla_libero_goal_*_actor*.yaml` 作为 outcome reward provider,跑 1 epoch PPO,确认 reward 信号上升、actor 不崩。**这是终极指标**,不是 F1。

## 4. 不做的事

显式声明几条**不在本方案内**的,避免被自己说服去做:

- **不重训 WM**。当前 chunk-aware WM(在 [chunkaware_m1024_d6_resume10k_bs80_20k_metrics/20260525_135027](../data/outputs/worldmodel/rynn_dino_wm_action_hidden/chunkaware_m1024_d6_resume10k_bs80_20k_metrics/20260525_135027) 跑到 20k)已是分类器输入,classifier 自己出了问题,跟 WM 无关。
- **不上 contrastive / triplet / temporal smoothness loss**。LR 已达 0.87,问题在管线不在 loss。引入新 loss 是 R3 之外的复杂度。
- **不扩 dataset**。401 success + 67 failure 已经够 LR 达 0.87。先把现有数据用对。
- **不动 WM-内 `reward_head`**。那是 dense per-step,跟 outcome classifier 是两套(见 2026-05-25 与用户的对话)。

## 5. 时间预算

| 阶段 | 单 GPU 估计 | 关键 ckpt 产出 |
|---|---|---|
| 0.1 episode eval 脚本 | 2 h | `episode_eval.json`,5 个旧 ckpt 重判分 |
| 0.2 LR baseline | 1 h | `lr_ceiling.pkl` + episode F1 |
| 1 v4_real_hidden | 4 h(20k step) | `v4_real_hidden/best.ckpt` |
| 2 v5 三档容量 | 3×3 h | `v5_{linear,mlp2,transformer2}/best.ckpt` |
| 3.1-3.2 协议+sweep | 1 h | 决定最终 (ckpt, threshold) |
| 3.3 PPO smoke | 4 h | actor 1 epoch curve |

**总计 ~24 GPU-hour,GPU 4-7 在跑 chunk-WM 时可用 GPU 0-3**。

## 6. 提交结构

按阶段开 PR / commit:

1. `feat(classifier-eval): episode-level eval harness` ← 阶段 0.1 + 0.2
2. `feat(classifier-data): real-hidden mode for WMReplayClassifierDataset` ← 阶段 1
3. `feat(classifier-model): linear / mlp2 head variants` ← 阶段 2
4. `feat(classifier-protocol): WMPO-parity evaluation` ← 阶段 3.1-3.2
5. `docs(classifier): final ckpt + threshold selection` ← 3.3 后归档
