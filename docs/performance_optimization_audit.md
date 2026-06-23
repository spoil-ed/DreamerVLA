---
title: DreamerVLA 性能优化审查
date: 2026-06-23
tags:
  - performance
  - optimization
  - audit
  - gpu-memory
  - dataloader
  - async
aliases:
  - 性能审查
  - Perf Audit
  - Optimization Audit
status: draft
---

# DreamerVLA 性能优化审查

> [!abstract] 范围与方法
> 对 `dreamervla/` 全包（约 74k 行）做了一次**只读**性能审查，覆盖 7 个子系统：在线 RL 采集热路径、离线训练 runner / checkpoint、RL/WM 算法更新、世界模型 / 编码器、VLA embodiment 模型、数据 / 预处理流水线、env / utils / workers。审查只针对"基于朴素实现、可优化"的点：**GPU 显存、内存/RAM、异步并发、cache 复用、向量化批处理、冗余计算、IO、数据加载、数值精度**。
>
> 每条发现都落到真实 `文件:行号`。本文中标注 ==已核验== 的条目是审查者本人重新读源码确认过行号与逻辑的（共 10 处关键项）；其余条目来自子系统审查，行号准确但未逐条复读。
>
> ⚠️ 本审查未修改任何代码。下面是发现清单与建议，不是已落地的改动。

> [!tip] 怎么读这份文档
> - 只想快速行动 → 看 [[#1 速查 优先级清单]]。
> - 想理解"为什么到处都慢" → 看 [[#2 跨子系统的共性模式]]（这是本审查最有价值的部分：~80 条发现可归约为 10 个根因模式）。
> - 想要完整清单 → 看 [[#3 按子系统详单]]。
> - 改之前先看 [[#4 改动前必读的约束]]。

---

## 1. 速查 优先级清单

### 1.1 立即可做的低风险高收益（quick wins）

> [!success] 这些改动数值等价或近似等价、改动局部、风险低，建议优先做。

| # | 位置 | 类别 | 一句话 | 影响 | 置信 |
|---|------|------|--------|:----:|:----:|
| Q1 | `utils/ema.py:41-45` | 向量化 | 逐参数 `mul_().add_()` 循环 → `torch._foreach_*` 融合多张量核 | Med | ==已核验== |
| Q2 | `hybrid_engines/weight_syncer/objectstore.py:70-71` 等 5 处 | 内存 | `.detach().cpu().clone()` 的 `.clone()` 是多余的整模型拷贝，删掉 | High | ==已核验== |
| Q3 | `dataset/vla_sft_hdf5_dataset.py:210` | IO | 每个 `__getitem__` 读整段 `demo["actions"]`（O(T²)）→ 只切 `[index:index+H]` | High | ==已核验== |
| Q4 | `dataset/pixel_sequence_dataset.py:176,183` | IO | 同上整段读 + 多余 `.copy()` | Med | ==已核验== |
| Q5 | `algorithms/ppo/outcome.py:117-125` | 向量化 | sparse reward 用 Python 循环 + 逐元素 `.item()` → 一次 `scatter_` | Med | ==已核验== |
| Q6 | `workers/inference/inference_worker.py:115-122` | 异步 | 逐行 `.cpu().numpy()`（每 env 一次 D2H）→ 整批一次转移 | Med | High |
| Q7 | `runners/dreamerv3_pixel_runner.py:394` / `dreamerv3_token_runner.py:325` | 冗余 | 每步无条件 `{k:.cpu()}` 物化指标，再判断是否 `log_every` → 把物化移进 log 分支 | Med | High |
| Q8 | `runners/distributed.py:185-213` | 异步 | `reduce_mean_dict` 每个 key 一次 `all_reduce`+`.item()` → 堆叠成一个张量一次 all_reduce（同仓 `dreamerv3_pixel_runner.py:262` 已有正确写法） | Med | High |
| Q9 | `models/embodiment/.../modeling_chameleon.py:1261` | 异步 | `img2bpe_mapping_tensor` 常驻 CPU → 每次图像 token 做 GPU→CPU→GPU 往返；注册为 buffer 留在 GPU | Med | High |
| Q10 | `models/embodiment/openvla/openvla_action_model.py:699` | 向量化 | `np.asarray([bin_centers[da] for da ...])` → `bin_centers[discretized_actions]`（official 变体已是对的） | Low | High |
| Q11 | `hybrid_engines/weight_syncer/bucket.py:80,95` | 异步 | 分桶权重同步用循环内 `ray.get` 完全串行化 → 先发全部 ref 再一次 `ray.get` | Med | High |

### 1.2 高收益但需谨慎 / 需验证

> [!warning] 这些改动收益大，但触及数值、采样语义或 vendored 上游，必须配等价性测试 / smoke run（参考已有的 `tests/unit_tests/test_wmpo_microbatch_equivalence.py`）。

| # | 位置 | 类别 | 一句话 | 影响 | 风险 |
|---|------|------|--------|:----:|:----:|
| H1 | `runners/online_cotrain_runner.py:555` + `collect_parallel_rollouts.py:475` | 异步 | 单 env 采集 / cotrain 仍逐 env 串行（env.step ↔ GPU forward），而批量路径 `VecRolloutEnv`+`OFTBatchedDecoder` 已存在却只在 `envs_per_gpu>1` 用 | High | RL 语义依赖单流 latent 递归 |
| H2 | `runners/online_replay.py:237-322` | 数据加载 | replay 存成 list-of-dict，每次 `sample()` 用嵌套列表推导对 22 万维 `obs_embedding` 做 `np.stack` + 双重 Python 循环填 actions | High | 改 replay 内部布局（有测试覆盖） |
| H3 | `runners/online_dreamervla.py:909`（每步） | 冗余 | 每个 env step 全量重扫 `_valid_records()`（O(episodes)）算 readiness，并触发一次 DDP `all_reduce` | Med | 需增量维护计数 |
| H4 | `algorithms/dreamervla.py:1009-1066` | 冗余 | 诊断用 `_flat_grad` 每步最多 4 次额外 `autograd.grad(retain_graph=True)` + 最终 `backward()`，`retain_graph` 全程占激活显存 | High | 纯诊断，gate 后默认关 |
| H5 | `models/world_model/dino_wm.py:667` / `tssm_torch.py:490` | 冗余/显存 | 自回归 imagination/rollout 每步对整段历史重算 self-attention（无 KV-cache），掩码/位置编码每步重建 | High | 1:1 复刻上游，需数值对齐 |
| H6 | `models/world_model/tssm_torch.py:100-114` | 显存 | 注意力用 einsum 显式物化 `[Tq,Tk,B,H]` score + `.float()` 往返，未走 `F.scaled_dot_product_attention`(Flash) | High | mask 语义需核对 |
| H7 | `runners/dreamervla_runner.py:1364,1678` | 精度/显存 | 整个 WM 静态 `.to(bf16/fp16)` 且无 `autocast`/`GradScaler`；policy/critic 又被 `.float()` 升回 fp32 | High | 数值，需 smoke run |
| H8 | `runners/base_runner.py:800-821,1054` | IO/异步 | checkpoint 同步深拷贝整模型+优化器到 CPU 再阻塞 `torch.save`，且 latest 与 top-k 重复序列化两份 | Med | 落盘原子性 |
| H9 | `models/embodiment/chameleon_model/modeling_xllmx_chameleon_ck_action_head.py:497-519,640` | 冗余 | action-head 解码每步重建 O(L²) `tril` 掩码，内部 Python 循环 + 逐步 `.cpu().numpy()` 同步（O(L³)/rollout） | High | vendored，保留双向 block 语义 |
| H10 | `models/reward/latent_success_classifier.py:241-250` | 向量化 | 成功分类器滑窗逐窗跑 8 层 Transformer（O(T) 次独立前向） → `unfold` 堆批一次前向 | Med | early-break 失效，吞吐净赚 |

> [!note] H1 已有在进行的工作
> cotrain 在线采集向量化（单 env osmesa → 多 env `VecRolloutEnv`+egl，~4×）已立项，计划见 `docs/plans/2026-06-23-cotrain-vec-egl-rollout.md`，正在 worktree 中实现（Option 1）。本条列入是为完整性，落地以该计划为准。

---

## 2. 跨子系统的共性模式

> [!note] 这一节是审查的核心结论
> 约 80 条单点发现，其实反复落在下面 **10 个根因模式**上。按模式去改，比逐条改更省力、更不容易漏。

### A. 热路径里的 GPU 同步点（`.item()/.cpu()/.numpy()/.tolist()`）

每个 `.item()`/`.cpu()` 都会强制一次 CUDA 同步，让 host 等 device、打断流水线。它们散落在每步/每 minibatch 循环里，单个便宜、累计可观。典型站点：

- `online_dreamervla.py:1081-1110` —— 每次 PPO 更新约 9 个 batch 元数据张量各 `.detach().cpu().tolist()`（≈9 次 D2H）+ 双文件 `flush()`。
- `algorithms/ppo/outcome.py:679,735-739` —— 每 `epoch×slice×chunk` 累加标量都 `.item()`，而 `loss_c.backward()` 紧邻其前，`.item()` 会阻塞到反向完成。
- `algorithms/dreamervla.py:1221-1338` / `ppo/dense.py:713-784` —— 收尾构建 50~60 个 `float(t.mean().cpu())` 标量，每个一次同步。
- `dreamerv3_pixel_runner.py:394` / `dreamerv3_token_runner.py:325` —— 每步 `{k:.cpu()}` 物化在 `log_every` 判断**之前**。
- `dreamervla_runner.py:1271-1277` —— val loop 每个 batch `.item()` 两次。
- `inference_worker.py:115-122` —— 逐行 `.cpu().numpy()`。
- `runners/online_*`、`oft_collect_common.py:25` —— 每步动作 numpy 往返（`.cpu().numpy()` → `torch.from_numpy().to(device)`）。
- `modeling_xllmx_chameleon_ck_action_head.py:684-721` —— 掩码构建里逐 block `.cpu().numpy()`。

**统一手法**：循环内把标量累加在 **GPU 张量**上（`acc += x.detach()`），循环外只 `.item()` 一次；多张量一次性 `.cpu()` 后在 CPU 上索引；动作切片直接留在 device 上传递，避免 numpy 往返。

### B. 诊断 / 日志未按 cadence 收口

大量诊断指标在训练热路径**每步无条件**计算，且很多是额外的前向/反向。它们只产出标量，却付出大模型级的算力/显存。应统一 gate 在 `debug` flag 或 `log_every` 之后。

- `algorithms/dreamervla.py:1009-1066`（H4）—— `_flat_grad` 最多 4 次额外 `autograd.grad(retain_graph=True)` + 最终 `backward()`，仅为 norm/cosine。
- `algorithms/dreamervla.py:1067-1073` / `ppo/dense.py:390` —— `_named_grad_norm` 对全参数名做 5 次独立遍历。
- `online_dreamervla.py:1030-1073` —— 每次更新两次 `torch.cat([p.float().flatten() ...])` 拍平全部可训练参数算 drift，外加多次 `.item()`。
- `ppo/dense.py:237-291,602-658` —— 即便 `actor_bc_ref_scale==0`，每 step×epoch 仍跑一次确定性 `sample(return_chunk=True)` 前向算 drift 指标。
- `dreamervla_runner.py:1779-1937` —— 每步组装 80+ key 指标字典并 `reduce_mean_dict`，不受 `log_every` 约束。
- `utils/resource_metrics.py:46-65` —— 子进程 `nvidia-smi`（~50–200ms），若按 iteration 调用则远超被记录指标本身的成本。

**统一手法**：诊断默认关，或按 N 步采样；drift/grad 分解类指标只在最后一个 epoch 算一次。

### C. 自回归 imagination / rollout 全序列重算（无 KV-cache，掩码/位置编码每步重建）

世界模型在 horizon 上滑动推理，但每步都从头对整段历史跑完整注意力，是经典的"无 KV-cache 自回归"，单条 rollout 复杂度 O(W²) 而非 O(W)。同时因果掩码、正弦位置编码这些**只依赖序列长度的常量**也每步重建。

- `dino_wm.py:667-703`（`predict_next`）、`:1081-1107`（`_rollout_hidden`，同一帧最多被 `encode` 3 次，并在循环里 `torch.cat` 累积 → O(H²) 拷贝）。
- `dino_wm.py:401-417`（`_block_causal_mask` 每次重建）、`:452-457`（`replace_actions_from_z` 为改最后 slot 而 `z.clone()` 整段）。
- `tssm_torch.py:490-537,784-837`（`observe_next` 每步重跑整窗 Transformer）、`:246-264`（位置编码/掩码每 forward 重算）。
- `dino_wm_chunk.py:397-400`（chunk K 步逐步循环 + 重复 encode 历史帧）。
- `algorithms/dreamervla.py:848-913`（每个 imagined latent 单独调 WM 头，`4*(H+1)` 次小前向）、`ppo/dense_chunk.py:251-259`（逐帧 WM reward 头）。
- `modeling_xllmx_chameleon_ck_action_head.py`（H9，O(L³)/rollout）。

**统一手法**：(1) 掩码/位置编码按 `(len, device, dtype)` 缓存或预建 buffer 切片；(2) rollout 复用上一步已编码结果、预分配缓冲区按 index 写入替代 `cat`；(3) WM 头把堆叠后的 latent 一次批量前向；(4) 大序列注意力改 `F.scaled_dot_product_attention`。

### D. 混合精度缺失 / 静态 cast / 大张量 `.float()` 升精度

主 DreamerVLA 路径没有 `autocast`/`GradScaler`（`grep` 确认算法目录零命中），而是把整个 WM 静态 `.to(bf16/fp16)`，再把 policy/critic 用 `.float()` 升回 fp32 —— 既丢了 fp32 master weights，又在每步物化额外 fp32 激活。

- `dreamervla_runner.py:1364-1366,1678`（H7，静态 cast 无 scaler）。
- `base_runner.py:469-516` / `dreamervla_runner.py:1328`（冻结 VLA encoder 仍 fp32、用 `no_grad` 而非 `inference_mode`，未 bf16）。
- `dino_wm.py:746-756`（`_hidden_loss_terms` 对 `[B,T,35,1024]` 大 hidden 整体 `.float()` 再算 MSE+两次 normalize）。
- `modeling_chameleon.py:1835-1841`（rollout 路径仍把 `[B,L,~65k]` 全词表 logits `.float()`）。
- `utils/hf_checkpoint.py:115-136`（加载时把所有权重强制 `.to(fp32)`，翻倍 host 内存与加载时间）。

**统一手法**：参数留 fp32，forward/backward 包 `torch.autocast(bf16)`，fp16 路径加 `GradScaler`；冻结 eval-only 子模块直接 bf16 + `inference_mode`；loss 归约用 fp32、但避免对整张大特征统一升精度；rollout 只对 `logits[:, -1:]` 升精度。

### E. DataLoader 配置欠优 + H2D 阻塞

主路径 dataloader 配置偏保守，且 H2D 拷贝全是阻塞的。

- dreamervla 系列 config `pin_memory: false`、`prefetch_factor: 1`、`num_workers: 2`；而 `dreamerv3_pixel/token` 默认 `pin_memory: true` 且 `to_device(non_blocking=True)`（对照可见差距）。
- `dreamervla_runner.py:293,309,335,360-419,1262` —— 一律 `.to(self.device)`，无 `non_blocking=True`，host buffer 未 pin。
- `base_runner.py:322-375` `make_distributed_dataloader` 不强制 `pin_memory`/`persistent_workers`；`latent_classifier_runner.py:143` val loader 写死 `num_workers=0`；`dreamerv3_token_runner.py:41` loader 无 `prefetch_factor`。

**统一手法**：config 开 `pin_memory: true`、调大 `prefetch_factor`，device move 全部 `non_blocking=True`；CUDA 设备下 `pin_memory=false` 时 runner 给 warning。

### F. `__getitem__` / `__init__` 重复重活（整段读取、逐帧 pickle、无 per-worker 缓存）

- **整段读 actions**：`vla_sft_hdf5_dataset.py:210`、`pixel_sequence_dataset.py:176` 每个窗口读整段（O(T)），同一 demo 多窗口 → 总读取 O(窗口数×T)，而真正需要的只有 `horizon` 行。
- **构造期全量解 pickle**：`pretokenize_dataset.py:300-343` 在 `__init__` 对每个 pkl `pickle.load` 仅为读图片路径建索引；manifest（`meta.next_obs.image`）其实已有该字段，`one_trajectory_pretokenize_dataset.py:88` 已示范优先用 manifest。
- **逐帧 pickle 无缓存**：`pretokenize_dataset.py:374-424` 每个 `__getitem__` 对窗口内每帧 `pickle.load`，stride=1 时相邻窗口几乎全重叠却无 per-worker LRU 缓存。根因之一是 `pre_tokenize_action_local.py:265` 的"一帧一 pkl"存储布局。
- **全量进 RAM**：`wmpo_aligned_latent_dataset.py:95-112` 构造时把整库 `obs_embedding` 读进常驻；train/val 各持一份。
- **swap 负样本重读全 demo**：`wm_replay_classifier_dataset.py:264-269` 仅为拿 `actions` 却开 2 个 HDF5 读全段 obs/dones/rewards。
- **token→img 映射 Python 逐元素**：`token_sequence_dataset.py:173-209` 逐 token `if tok in set` / dict 映射，可用 numpy 查表数组向量化。

**统一手法**：HDF5 按需切片读；优先用 manifest 字段避免解 pickle；加 per-worker LRU payload 缓存（仓内已有 `cached_hdf5_file` 句柄缓存的范式）；轻量 loader 只读需要的字段。

### G. Replay buffer：list-of-dict 每次重组 + 每步全量重扫

- `online_replay.py:263-298`（H2）—— `sample()` 对每个字段用嵌套列表推导 `np.stack`，对 22 万维 `obs_embedding` 尤甚，外加双重 Python 循环填 `actions/current_actions`。
- `online_replay.py:360-410` —— `sample_classifier_windows` 每个样本 `np.stack` + `np.ascontiguousarray` 拷贝。
- `online_replay.py:144-200`（H3）—— `_valid_records()` 每步被 `get_replay_task_stats_global` 调用做全量重扫，DDP 下还附带每步 `all_reduce`。

**统一手法**：`add_episode` 时就把每条 episode 存成**每字段一块连续 numpy 数组**，`sample` 切片视图 + 单次向量化堆叠；per-task 计数增量维护、按 N 步而非每步做 readiness 检查。

### H. 显式 Python 循环本可向量化 / 批处理

- `outcome.py:117-125`（Q5，sparse reward scatter）。
- `latent_success_classifier.py:241-250`（H10，滑窗逐窗前向）。
- `openvla_oft/official/openvla_oft_action_model.py:135-225`（OFT `prepare_inputs` 逐样本 PIL+processor 循环；prompt 重复 tokenize N 次）。
- `backbone_dreamerv3_wm_runner.py:102-130`（逐帧逐视图 `.cpu().numpy()` 转 PIL）。
- `ema.py:41-45`（Q1，逐参数 → `_foreach_`）。
- `openvla_action_model.py:332-354`（多 scale 循环里重复 `/255`）。

### I. 冗余内存拷贝 / clone / 序列化

- `weight_syncer/*` + `learner_worker.py:537` + `inference_worker.py:267`（Q2，多余 `.clone()`，对多 B 参数模型是两份额外 CPU 拷贝 + ray.put 第三份）。
- `base_runner.py:1054`（H8，checkpoint 深拷贝 + 阻塞 save + latest/top-k 双写）。
- `vec_rollout_env.py:78-168`（每 env 每步把 2 张 256² 帧 + sim state pickle 过 Pipe，文件自身 TODO 已承认）。
- `dino_wm.py:452`（`z.clone()` 整段只为改一个 slot）。
- `image_tokenizer.py:89-96`（PIL→fp64 numpy 归一化在 CPU 上做再降精度）。

### J. 异步 / 并发机会（CPU sim 与 GPU 计算未重叠）

- **采集**：H1 单 env 串行（env.step ↔ forward 互等）；批量路径已存在。
- **Ray**：`env_worker.py:85` 每个 episode 边界阻塞 `ray.get(add_episode)`；`bucket.py:80,95`（Q11）分桶同步串行 `ray.get`。
- **落盘**：`fixed_step_video.py:84` / `eval_env.py:108` / `wm_image_viz.py` 视频/面板编码在采集线程内联（recorder 已缓存帧，改后台线程消费即可）；checkpoint 阻塞（H8）。

**统一手法**：双缓冲（边 step 边 forward）；Ray 调用 fire-and-forget + 有界 backpressure；编码/落盘移到后台线程。

---

## 3. 按子系统详单

> 列出每个子系统的完整发现。"现状/优化"力求一句话；展开解释见 [[#2 跨子系统的共性模式]] 对应字母。置信度 ==已核验== 表示审查者复读确认。

### 3.1 在线 RL 采集热路径

| 位置 | 类别 | 现状 → 优化 | 影响 | 置信 |
|------|------|-------------|:----:|:----:|
| `online_cotrain_runner.py:555` / `collect_parallel_rollouts.py:475` | 异步/向量化 | 单 env 串行采集；改用已存在的 `VecRolloutEnv`+批量 decoder 或双缓冲（H1） | High | High |
| `online_replay.py:237-322` | 数据加载 | list-of-dict 每次 sample 重组大张量（H2/G） | High | ==已核验== |
| `online_dreamervla.py:909`（每步） | 冗余 | 每步全量 `_valid_records` 重扫 + DDP all_reduce（H3/G） | Med | ==已核验== |
| `online_dreamervla.py:1030-1073` | 显存/冗余 | drift 每更新两次拍平全参 + 多次 `.item()`（B） | Med | ==已核验== |
| `online_dreamervla.py:1081-1110` | IO/同步 | ~9 个张量逐个 `.cpu().tolist()` + 双 flush（A） | Med | High |
| `rollout_hidden_extractor.py:230-267` | 缓存/冗余 | 每步重 tokenize 不变 prompt，逐视图单独走 processor | Med | High |
| `online_utils.py:143-188` | 冗余/精度 | `obs_to_action_hidden` 每步 `training=True` 跑冻结 backbone + 逐 token Python list + 二次前向 | Med | Med |
| `online_replay.py:360-410` | 数据加载 | classifier 窗口逐样本 `np.stack` + contiguous 拷贝（G） | Med | High |
| `runners/online_*`（每步） | 精度 | 动作 numpy 往返强制每步 CUDA 同步（A） | Med | High |
| `collect_parallel_rollouts.py:224` 等 | IO | per-episode `print(..., flush=True)` | Low | High |
| `oft_collect_common.py:25` / `rollout_hidden_extractor.py:257,523` | 缓存 | 每步重算 `device` / `num_patches` / `process_action` `.copy()` 等不变量 | Low | High |

### 3.2 离线训练 runner / checkpoint

| 位置 | 类别 | 现状 → 优化 | 影响 | 置信 |
|------|------|-------------|:----:|:----:|
| `dreamervla_runner.py:1364,1678` | 精度/显存 | 静态 cast WM 无 autocast/GradScaler，policy `.float()` 升回（H7/D） | High | Med |
| `dreamervla_runner.py:1590,1667` | 显存 | `gradient_accumulate_every` 被读但训练循环无累积，LR 计划按累积算（不一致） | High | Med |
| `base_runner.py:800-821,1054` | IO/异步 | checkpoint 同步深拷贝 + 阻塞 save + latest/top-k 双写（H8/I） | Med | High |
| `dreamervla_runner.py:293-419,1262` + config | 数据加载 | H2D 无 `non_blocking`，config `pin_memory:false`（E） | Med | High |
| `dreamervla_runner.py:1249-1295` | 冗余 | val loop 每 batch `.item()`×2，用 `no_grad` 而非 `inference_mode`（A） | Med | High |
| `dreamerv3_pixel_runner.py:394` / `token_runner.py:325` | 冗余/同步 | 每步 `.cpu()` 物化指标在 `log_every` 之前（A/B，Q7） | Med | High |
| `distributed.py:185-213` | 异步 | `reduce_mean_dict` 每 key 一次 all_reduce（Q8） | Med | High |
| `base_runner.py:469-516` / `:1328` | 显存/冗余 | 冻结 VLA encoder fp32 + `no_grad`（应 bf16 + `inference_mode`）（D） | Med | Med |
| `backbone_dreamerv3_wm_runner.py:102-130` | 向量化 | 逐帧逐视图 `.cpu().numpy()`→PIL（H） | Med | Med |
| `base_runner.py:322-375` / `latent_classifier_runner.py:143` | 数据加载 | dataloader 不强制 pin/persistent；val `num_workers=0`（E） | Med | Med |
| `dreamervla_runner.py:1779-1937` | 冗余 | 每步组 80+ key 指标并 reduce，无 log gate（B） | Low-Med | Med |
| `latent_classifier_runner.py:428-475` | 向量化 | episode 级评估逐 chunk numpy↔tensor↔device 往返 + Python max | Low-Med | Med |
| `utils/hf_checkpoint.py:115-136` | IO/精度 | 加载强制全 fp32；`checkpoint_format="both"` 双写（D） | Low-Med | Med |

### 3.3 RL / WM 算法更新

> [!info] 参考实现
> `algorithms/ppo/outcome.py` 已实现 MEM-RL-01 的 micro-batch（`update_micro_batch_starts`、`_slice_latent`、逐 chunk 反向、CPU offload、bf16 feat 存储），是本仓**显存管理的范本**。下面的缺口主要在 **没有同等改造的 `dense.py` / `dense_chunk.py`**。

| 位置 | 类别 | 现状 → 优化 | 影响 | 置信 |
|------|------|-------------|:----:|:----:|
| `dreamervla.py:1009-1066` | 冗余 | `_flat_grad` 诊断每步最多 4 次额外反向 + retain_graph（H4/B） | High | ==已核验== |
| `ppo/dense.py:237-291,602-658` | 冗余 | BC-off 时仍每 step×epoch 跑确定性 sample 前向算 drift（B） | High | High |
| `ppo/dense.py:580-706` / `dense_chunk.py:290-332` | 显存 | 整条 imagined 轨迹一次 `backward()`，未 micro-batch（OOM 暴露面） | High | Med |
| `outcome.py:117-125` | 向量化 | sparse reward Python 循环 + `.item()`（Q5/H） | Med | ==已核验== |
| `outcome.py:679,735-739` | 异步 | 每 chunk/epoch `.item()` 累加标量（A） | Med | High |
| `outcome.py:219,651` | IO | `actor_feats` offload CPU 未 pin，每 epoch 同步 `.to(device)` 重传 | Med | Med |
| `dreamervla.py:848-913` | 向量化 | 每个 imagined latent 单独调 WM 头（`4*(H+1)` 次小前向）（C） | Med | Med |
| `outcome.py:688-713` | 冗余 | BC 路径每 chunk 额外 sample 前向；ref chunk 每 epoch 重算（应缓存） | Med | Med |
| `dreamervla.py:1221-1338` / `dense.py:713-784` | 异步 | 收尾 50~60 个 `.cpu()` 标量逐个同步（A） | Low-Med | Med |
| `dreamervla.py:1067-1073` / `dense.py:390` | 冗余 | `_named_grad_norm` 5 次全参遍历（B） | Low | Med |
| `dense_chunk.py:251-259` | 向量化 | 逐帧 WM reward 头（应 reshape 批一次）（C） | Low-Med | High |
| `outcome.py:626,681-767` | 内存 | 末 epoch ratio 记录进 list 再 `cat` 只为 4 个标量（应跑动归约） | Low | Med |

> [!check] 已确认无问题（避免误改）
> `grpo.py` 的 `_group_advantage` / `masked_mean_ratio_chunk_term` 已向量化；`tdmpc_mpc.py` eval-only `@no_grad` 且预分配 buffer；`compute_lambda_returns` 的反向递归是固有顺序扫描（H 很小）。

### 3.4 世界模型 / 编码器

| 位置 | 类别 | 现状 → 优化 | 影响 | 置信 |
|------|------|-------------|:----:|:----:|
| `tssm_torch.py:490-537,784-837` | 冗余/显存 | `observe_next` 每步重跑整窗 Transformer（无 KV-cache）（H5/C） | High | High |
| `tssm_torch.py:100-114` | 显存 | einsum 显式 score 矩阵 + `.float()` 往返，未走 SDPA/Flash（H6） | High | ==已核验== |
| `dino_wm.py:667-703,1081-1107` | 冗余 | `predict_next`/`_rollout_hidden` 重复 encode 历史帧 + 循环 `cat`（C） | Med-High | High |
| `latent_success_classifier.py:241-250` | 向量化 | 滑窗逐窗前向（H10/H） | Med-High | High |
| `dino_wm.py:401-417` | 冗余 | `_block_causal_mask` 每次重建（C） | Med | High |
| `rynnvla_encoder.py:388-405,467-521` | 缓存/显存 | 冻结 backbone `training=True`、`output_hidden_states=True` 取全层、无帧级缓存、可能被 `train()` 切回（C/D） | Med-High | Med |
| `chameleon_latent_action.py:763-795` | 向量化/显存 | 每层 ×2 stream 的 `permute→reshape→permute` 强制 contiguous 拷贝（深度 8 ≈ 32 次/前向） | Med | Med |
| `dino_wm_chunk.py:397-400` | 冗余 | chunk K 步逐步循环 + 重复 encode（C） | Med | Med-High |
| `dino_wm.py:746-756` | 精度/显存 | `_hidden_loss_terms` 对大 hidden 整体 `.float()`（D） | Low-Med | Med |
| `tssm_torch.py:246-264` | 缓存 | 正弦位置编码 + 掩码每 forward 重算（C） | Low-Med | High |
| `dino_wm.py:452-457,687` | 显存 | `z.clone()` 整段只为改一个 slot；历史逐步 `cat`（I） | Low-Med | Med |

### 3.5 VLA embodiment 模型

> [!warning] vendored 上游
> 标 🔒 的在 `chameleon/`、`chameleon_vae_ori/` 等 vendored 目录，改动会偏离上游，应 gate / 文档化。其余在 DreamerVLA 适配层，较安全。

| 位置 | 类别 | 现状 → 优化 | 影响 | 置信 |
|------|------|-------------|:----:|:----:|
| `modeling_xllmx_chameleon_ck_action_head.py:497-519,640` | 冗余 | 每步重建 O(L²) `tril` 掩码 + Python 循环 + `.cpu().numpy()`（H9/A） | High | High |
| `openvla_oft/official/openvla_oft_action_model.py:135-225` | 向量化 | `prepare_inputs` 逐样本 PIL+processor，prompt 重复 tokenize（H） | Med-High | High |
| `openvla/openvla_action_model.py:664-728` | 缓存 | 固定长度动作仍走 HF `generate()` 逐 token + 存全层 hidden（OFT 变体已单次前向） | Med-High | Med |
| 🔒 `modeling_chameleon.py:1261-1264` | 异步 | `img2bpe` 查表常驻 CPU，GPU→CPU→GPU 往返（Q9/A） | Med | High |
| 🔒 `modeling_chameleon.py:1835-1841` | 精度 | rollout 仍把 `[B,L,~65k]` 全词表 logits `.float()`（D） | Med | Med |
| `openvla/.../openvla_action_model.py:722` / `openvla_oft/dreamervla/...:350` | 显存 | 对全词表写两片 `-inf` 再全词表 cross-entropy；应先切 256 bin（official 已对） | Med | Med |
| 🔒 `chameleon_vae_ori/image_tokenizer.py:86-99` | 显存 | VQGAN encode 无 `no_grad` | Med | High |
| `openvla_oft/official/...:208-218,524-533` | 冗余 | 已 left-pad 的 input 在 `_build_embedding` 再做一次 argsort+gather | Low-Med | Med |
| 🔒 `chameleon_vae_ori/image_tokenizer.py:89-96` | 数据加载 | PIL→fp64 numpy 归一化在 CPU（应 GPU + 目标 dtype）（I） | Low-Med | Med |
| 🔒 `chameleon_vae_ori/vqgan.py:100-107` | 冗余 | VQ 量化器每次重算常量 codebook `‖e‖²` | Low-Med | Med |
| `openvla/openvla_action_model.py:699` / `openvla_oft/dreamervla/...:397` | 向量化 | `bin_centers` 列表推导（Q10） | Low | High |
| `openvla/openvla_action_model.py:332-354` | 向量化 | 多 scale 循环重复 `/255` + normalize | Low | Med |

### 3.6 数据 / 预处理流水线

| 位置 | 类别 | 现状 → 优化 | 影响 | 置信 |
|------|------|-------------|:----:|:----:|
| `vla_sft_hdf5_dataset.py:210` | IO | 每个 item 读整段 actions（O(T²)）→ 切片（Q3/F） | High | ==已核验== |
| `pretokenize_dataset.py:300-343` | 数据加载 | 构造期全量解 pickle 建索引；应优先用 manifest（F） | High | High |
| `pretokenize_dataset.py:374-424` | 缓存 | 逐帧 pickle，重叠窗口无 per-worker 缓存（F） | Med-High | High |
| `pixel_sequence_dataset.py:176,183` | IO | 整段读 actions + 多余 `.copy()`（Q4/F） | Med | ==已核验== |
| `wmpo_aligned_latent_dataset.py:95-112` | 内存 | 整库 obs_embedding 进 RAM，train/val 各一份（F） | Med | Med |
| `preprocess_oft_action_hidden.py:374-403` | 向量化 | 预处理逐图调 processor，prompt 重复 tokenize（H） | Med | Med |
| `wm_replay_classifier_dataset.py:264-269` | 冗余 | swap 负样本开 2 HDF5 读全 demo 只为 actions（F） | Med | High |
| `token_sequence_dataset.py:173-209` | 冗余 | 逐 token Python BPE→img 映射（应 numpy 查表）（F） | Med | Med |
| `pretokenize_dataset.py:230-411,494` | 数据加载 | token 在 getitem 与 collate 各复制一遍 list，逐 token `int()`（F） | Med | Med |
| `rollout_dump_writer.py:159-238` / `online_rollout_dumper.py:133` | IO | 每 episode 双 flush + gzip 压高熵 fp16（应 lzf/none） | Med | Med |
| `pixel_hidden_sequence_dataset.py:257-267` | IO | `_validate` 反复 glob+open 找 hidden_dim | Low | Med |

> [!check] 已确认良好实践
> `pixel_sequence_dataset` / `vla_sft_hdf5_dataset` 已有 per-worker HDF5 句柄缓存（`cached_hdf5_file`、`swmr=True`）；`item_processor.py` 的 `process_image/action` 已加 `@no_grad`。

### 3.7 env / utils / workers

| 位置 | 类别 | 现状 → 优化 | 影响 | 置信 |
|------|------|-------------|:----:|:----:|
| `utils/ema.py:41-45` | 向量化 | 逐参数循环 → `_foreach_`（Q1/H） | Med | ==已核验== |
| `weight_syncer/objectstore.py:70` 等 5 处 | 内存 | 冗余 `.clone()`（Q2/I） | High | ==已核验== |
| `inference/inference_worker.py:115-122` | 异步 | 逐行 `.cpu().numpy()`（Q6/A） | Med | High |
| `weight_syncer/bucket.py:80,95` | 异步 | 分桶同步串行 `ray.get`（Q11/J） | Med | High |
| `env/env_worker.py:85-88` | 异步 | 每 episode 阻塞 `ray.get(add_episode)`（J） | Med-High | Med |
| `envs/train_env.py:526-595` | 冗余 | `_format_obs` 每步建全 VLA record（4×PIL+prompt），多数消费方只要 64² 图（J） | Med-High | Med |
| `runners/vec_rollout_env.py:78-168` | IO | 每 env 每步 pickle 2×256² 帧过 Pipe（I/J，文件自带 TODO） | Med | Med |
| `utils/resource_metrics.py:46-65` | IO | 子进程 `nvidia-smi` 若按 iteration 调用（B） | Low-Med | Med |
| `utils/fixed_step_video.py:84` / `eval_env.py:108` / `wm_image_viz.py` | 异步 | 视频/面板编码内联在采集线程（J） | Low-Med | Med |
| `utils/latent.py:15` + 诊断脚本 | 异步 | `reward_of` 逐步 `.cpu().item()`（诊断路径，低优先） | Low | Med |

---

## 4. 改动前必读的约束

> [!danger] 不要直接照搬，先满足这些前提
> 1. **数值契约**：`tssm_torch.py`、`dino_wm.py` 源码注明 1:1 复刻 TransDreamer / DINO-WM；`online_utils.obs_to_action_hidden` 有文档化的数值契约（`training=True` 是否可翻需先验证 backbone 分支）。任何注意力/掩码/精度/KV-cache 改动都应放在**可开关路径**后，并加等价性单测（仿 `test_wmpo_microbatch_equivalence.py`）。
> 2. **冻结模型**：参数来自 Hydra config（见记忆 [[no-hardcoded-values-style]]），别在函数里写死；冻结 encoder 切 bf16 前确认 eval-only 无 master-weight 需求。
> 3. **vendored 上游**：3.5 节标 🔒 的改动会偏离上游，应 gate + 在 docs 记录。
> 4. **fire-and-forget Ray**：删 `.clone()` / 改异步前确认推送的 state dict 在序列化前不被原地改写；异步 `add_episode` 要有有界 backpressure，否则在飞 episode 无界增长。
> 5. **config 改动**（pin_memory / prefetch / autocast）需在 GPU 上跑一次 smoke（见记忆 [[oft-online-cotrain-default-debug]]，full 配置在 80GB 已接近 OOM，留余量）。

## 5. 建议落地顺序

> [!todo] 分三批，先稳后险
> **第一批（本周，零/低风险）**：Q1–Q11 全部。删 `.clone()`、EMA `_foreach_`、HDF5 切片读、sparse reward scatter、批量 D2H、把诊断指标物化收进 `log_every` 分支、`reduce_mean_dict` 合并 all_reduce、`img2bpe` 留 GPU、并行 `ray.get`。这批基本数值等价、收益直接落在每步/每 sync 热路径。
>
> **第二批（需 smoke run）**：E（dataloader config + non_blocking）、B（诊断/grad 分解 gate 在 debug flag）、H8（checkpoint 原子写 + 后台线程，复用 `_dreamer_runner_common.py:86` 的 temp-then-rename）、H3（replay readiness 增量化）。
>
> **第三批（需等价性测试 + 重构）**：H1（采集向量化/双缓冲）、H2（replay 连续数组布局）、H5/H6（WM KV-cache / SDPA）、H7（autocast/GradScaler）、H9（Chameleon 掩码）、把 `dense.py`/`dense_chunk.py` 对齐 `outcome.py` 的 micro-batch。

---

> [!quote] 一句话总结
> 本仓的瓶颈不是"缺少 `no_grad`"（rollout/inference 路径基本都正确包了），而是 **(a) 热路径里散落的 GPU 同步点与未收口的诊断日志**、**(b) replay/dataset 把大张量存成 list-of-dict / 每次整段重读**、**(c) 世界模型自回归无 KV-cache、注意力未走 Flash、混合精度缺失**，以及 **(d) 大量本可批处理/异步/缓存却逐元素串行的 Python 循环**。好消息是：仓内多处已有正确范式（`outcome.py` 的 micro-batch、`dreamerv3_pixel_runner.py` 的批量 all_reduce、`_dreamer_runner_common.py` 的原子写、`cached_hdf5_file` 句柄缓存），很多优化就是"把已有的好写法搬到还没用上的路径"。
