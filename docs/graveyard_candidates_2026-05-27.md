# Graveyard 候选清单 — 2026-05-27

**用途：** 在执行 `mv → docs/archive/graveyard/` 之前，列出全部候选项并附证据，供人工 review。
未列入此清单的文件 / 模块均判定为活跃，保持原状。

**孤儿的判定标准（5 条同时成立）：**
1. `scripts/*.sh` 没有 `--config-name=<name>` 或 `bash scripts/...` 引用
2. `configs/README.md` / `scripts/README.md` 注册表未列出
3. 其它 YAML `defaults:` / Python `import` / config `_target_` 不指向
4. `tests/` 没有用它
5. `CLAUDE.md` / `AGENTS.md` / `README.md` / `docs/**` 没有提到

**保护的主线（即使 grep 失误也不动）：**
- 训练路由 YAML：`vla_pi0_query`、`vla_sft_one_trajectory`、`openvla_oft_hdf5`、`openvla_oft_hdf5_one_trajectory`、`world_model_dinowm_step`、`world_model_dinowm_chunk`、`dreamervla_rynn_dino_wm_actor_critic`、`dreamervla_rynn_dino_wm_wmpo_outcome`、`eval_libero_vla`
- `task/libero_{goal,object,spatial,10}.yaml`
- `scripts/README.md` 形式入口：`train_vla.sh`、`train_vla_nongoal_45.sh`、`train_wm.sh`、`train_dreamervla.sh`，以及 Diagnostics 表里的 4 个脚本

---

## 1. configs/ — **0 个孤儿**

13 个非归档 YAML 全部有引用。仅一条记录性发现：
- **docs drift**：`configs/README.md` 第 25、28 行登记的 `world_model_rssm_step.yaml` 和 `dreamervla_pi0_action_hidden_head_actor.yaml` 在磁盘上不存在。属于文档漂移，不属于"孤儿"（孤儿是文件存在但无人用）。**不处理**，留待文档维护方决定补 YAML 还是改 README。

---

## 2. docs/ — **2 个候选 → docs/archive/graveyard/docs/**

| 文件 | 证据 | 候选理由 |
|---|---|---|
| `docs/HANDOFF_2026-05-20.md` | `grep -rn "HANDOFF" CLAUDE.md AGENTS.md README.md` → 0 命中 | 日期化一次性交接文档（2026-05-20，距今 7 天），不在 CLAUDE.md "Further reading" 列表 |
| `docs/progress.md` | `grep "progress.md" CLAUDE.md AGENTS.md README.md docs/*.md`（排除自身）→ 0 命中 | 不在 CLAUDE.md/AGENTS.md "Further reading"；内容是按里程碑完成的状态流水账，最后一条为 2026-05-27 已完成项 |

**保留的 docs**（已验证有引用，全部出现在 CLAUDE.md/AGENTS.md "Further reading"）：
`dreamer_vla_writeup.md`、`findings.md`、`history.md`、`install.md`、`task_plan.md`、`TODO.md`、`classifier_revision_plan.md`、`multicollector_batched_encoder_plan.md`

---

## 3. dreamer_vla/ — **1 个候选 + 1 个待定**

| 模块 | 状态 | 证据 |
|---|---|---|
| `dreamer_vla/models/xllmx/model/tokenizer.py` | **orphan-candidate** | `dreamer_vla/models/xllmx/model/__init__.py` 是空文件无再导出；`grep "xllmx.model.tokenizer\|xllmx.model import"` → 0 命中 |
| `dreamer_vla/models/xllmx/data/data_reader.py` | **待定 — 关联到一个 bug** | `dreamer_vla/preprocess/item_processor.py:19` 写的是 `from dreamer_vla.xllmx.data.data_reader import ...`，但模块实际在 `dreamer_vla.models.xllmx.data.data_reader` — **import 路径是断的**。如果 `item_processor.py` 本身就跑不通，那 `data_reader` 也算孤儿；如果 `item_processor.py` 是活跃的需要修 import — 这超出本次清理范围（CLAUDE.local.md "外科手术式改动"）|

**建议：** 仅移 `tokenizer.py` 到 graveyard；`data_reader.py` 留原地，把 `item_processor.py` 的 import bug 加到 `docs/TODO.md` 让你后续单独决定（要么修 import 路径，要么连 `data_reader.py` 一起归档）。

---

## 4. scripts/ — **候选分两档**

### 4a. 明确孤儿（建议直接归档，零引用 + 无 manual-use 暗示）

| 脚本 | 证据 |
|---|---|
| `scripts/chain_libero_object_after_10.sh` | 仅自引用注释；无其它 shell/py/md 引用 |
| `scripts/run_libero_missing_data_45.sh` | 0 引用；包装 `process_all_libero_data.sh` 但自身没人调 |
| `scripts/run_vla_nongoal_after_data_45.sh` | 0 引用；包装 `train_vla_nongoal_45.sh` 但自身没人调 |
| `scripts/eval/eval_dreamerv3_token_action.py` | 0 外部引用（含此次刚从 `dreamer_vla/cli/` 搬过来的脚本 — 见 memory obs 4627）|
| `scripts/eval/eval_action_diff_wm_vs_sft.py` | 仅 `configs/archive/**` 间接相关；活跃区零引用 |
| `scripts/eval/run_eval_ppo_alltasks_g67_8proc.sh` | 仅自身 header 注释；无其它引用 |
| `scripts/training/train_chameleon_latent_flow_wm.py` | 0 引用 |
| `scripts/smoke/smoke_native_actor_training.py` | 0 引用 |
| `scripts/preprocess/preprocess_progress_delta_reward.py` | 0 引用 |

### 4b. 借界诊断脚本（标"manual-use"但无 caller — 等你确认是否归档）

这一档值得停下来征求你的意见 — 它们都是带 argparse 的独立 CLI 诊断工具，可能曾经用于一次性研究，未来或许还会手工跑。`scripts/README.md` Diagnostics 表只显式收录了 4 个（`analyze_rynn_hidden_action_metrics.py` / `monitor_dreamer_vla_metrics.py` / `visualize_dreamervla_reward.py` / `smoke_libero_online_env.py`），下列均不在表内：

| 脚本 | 说明 |
|---|---|
| `scripts/diagnostics/analyze_compact_token_z_reconstruction.py` | 独立 CLI；0 引用 |
| `scripts/diagnostics/compare_action_chunks.py` | 独立 CLI；0 引用 |
| `scripts/diagnostics/compare_policy_trace_runs.py` | 独立 CLI；0 引用 |
| `scripts/diagnostics/diagnose_dreamervla_latent_distribution.py` | 仅 `docs/archive/**` 提到 |
| `scripts/diagnostics/diagnose_hidden_token_structure.py` | 仅 `docs/archive/**` 提到 |
| `scripts/diagnostics/diagnose_residual_cosine.py` | "Section 5.1 follow-up" — 是写论文 5.1 节时用的诊断；0 引用 |
| `scripts/diagnostics/estimate_classifier_ceiling.py` | 0 引用；与 `docs/findings.md` 里"分类器天花板"主题相关但未直接被引 |
| `scripts/diagnostics/finetune_reward_head_sparse.py` | 0 引用 |
| `scripts/diagnostics/measure_recon_and_action_delta.py` | 0 引用 |
| `scripts/diagnostics/measure_reward_and_drift.py` | 0 引用 |
| `scripts/diagnostics/measure_wm_imagine_actor.py` | 0 引用 |
| `scripts/diagnostics/measure_wm_imagine_fidelity.py` | 0 引用 |
| `scripts/diagnostics/validate_oft_rynn_style_sidecar.py` | 0 引用 |
| `scripts/eval/eval_frozen_wm_actor.py` | 0 引用 |
| `scripts/training/train_frozen_wm_actor_critic.py` | 0 引用（虽 import 了活跃训练脚本，但自己没人调）|
| `scripts/preprocess/build_classifier_shards_from_demos.py` | 仅 `dreamer_vla/dataset/libero_sim_rollout_shards.py` 的 docstring 提到 |

**两种处理方式可选：**
- 保守：4b 全部留原地，只移 4a（共 9 项）
- 激进：4b 全部一起移走，方便日后清晰区分"形式诊断 vs 一次性研究脚本"

---

## 5. 代码内 deprecated/legacy 标记 — **不建议移动**

共找到 9 处 `# legacy` / `# Legacy aliases` / `# DO NOT USE in new code` 标记，分布在：

- `dreamer_vla/algorithms/__init__.py`、`dreamer_vla/algorithms/ppo/__init__.py` — 4 个 `dino_wmpo_*_step` 函数别名，**仍在 `__all__` 中导出且被外部使用**
- `dreamer_vla/algorithms/dino_wmpo.py`、`dreamer_vla/algorithms/dino_wmpo_chunk.py` — 整个文件是 legacy alias shim，**仍在 `__init__` 中导出**
- `dreamer_vla/dataset/wm_replay_classifier_dataset.py:237` — 老数据格式兼容分支
- `dreamer_vla/models/world_model/tssm_rynn_backbone_world_model.py:58` — `tssm_window` 参数 "legacy / unused but kept for cfg-compat"
- `dreamer_vla/models/chameleon_model/chameleon/modeling_chameleon.py:473` — flash-attn 版本 workaround `TODO`

这些都是**活跃维护中的 back-compat shim**，移走会破坏现有 import。CLAUDE.local.md 第 3 条"don't remove pre-existing dead code unless asked"明确说不要碰这种已存在但仍用的代码。**保持原状。**

无 `TODO(remove)` 模式被找到。

---

## 汇总

- **configs/**: 0 项
- **docs/**: 2 项 → `docs/archive/graveyard/docs/`
- **dreamer_vla/**: 1 项 → `docs/archive/graveyard/src/models/xllmx/model/`
- **scripts/ (4a 明确孤儿)**: 9 项 → `docs/archive/graveyard/scripts/...`
- **scripts/ (4b 借界诊断)**: 16 项 — **等你决定**
- **代码块标记**: 0 项移动（按 CLAUDE.local.md 保留）

**目录布局**：在 `DreamerVLA/docs/archive/graveyard/` 下按原始路径镜像存放，便于追溯。
