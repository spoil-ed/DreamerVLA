# Success Classifier 深度/宽度/token 消融结果

Date: 2026-06-27
Branch: `dinowm-align-query-before-proprio-language`

## 实验状态

本轮完成的是 profiler 消融:参数量、RL 形态前向峰值显存、前向耗时。真实 val-F1 /
episode-F1 未跑,原因是本地没有 OFT input-token success/failure sidecar:

- composed success path:
  `./data/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_oft_input_token_embedding_vla_policy_h1`
- composed failure path: `None`

因此 F1 列均标为 `N/A -- no input-token data`。下面的显存/耗时是 H100 80GB 上
`batch=16, window=8, token_dim=4096` 的实测 profiler-only 数字。

## Profiler 表

| num_layers | hidden_dim | token_count | params(M) | fwd_mem(MB) | fwd(ms) | val-F1 | episode-F1 |
|---:|---:|---:|---:|---:|---:|---|---|
| 4 | 1024 | 512 | 55.3 | 5115.0 | 259.5 | N/A -- no input-token data | N/A -- no input-token data |
| 6 | 1024 | 512 | 80.5 | 5211.1 | 379.0 | N/A -- no input-token data | N/A -- no input-token data |
| 8 | 1024 | 512 | 105.7 | 5307.2 | 503.0 | N/A -- no input-token data | N/A -- no input-token data |
| 12 | 768 | 512 | 88.8 | 4544.1 | 526.2 | N/A -- no input-token data | N/A -- no input-token data |
| 8 | 1024 | 196 | 105.4 | 2304.0 | 148.9 | N/A -- no input-token data | N/A -- no input-token data |
| 12 | 768 | 196 | 88.5 | 1976.0 | 144.9 | N/A -- no input-token data | N/A -- no input-token data |

CPU-only param pass produced the same parameter counts; CPU memory/time are intentionally `nan`.

## WMPO 对照

WMPO 判别器约为 VideoMAE-base:约 86M 参数、12 层、hidden 768、约 784 个像素时空 token。
本 classifier 在 latent token 上对齐结构,不是像素 patch:

- `12 x 768 x 512`:88.8M 参数,最接近 WMPO 的 depth/width 和参数量,并保留当前 OFT
  input-token 网格。代价是本轮 profiler 中最慢,约 526 ms/forward。
- `12 x 768 x 196`:88.5M 参数,depth/width 仍贴近 WMPO,显存和耗时降到约 2.0 GB /
  145 ms,但 token 数来自瘦身设定,真实训练前需要明确 token selection/pooling 策略。
- `8 x 1024 x 512`:参数量更大但深度更浅,不如 `12 x 768` 贴近 WMPO。

## 推荐档位

推荐下一轮真实 F1 训练优先跑:

1. `num_layers=12, hidden_dim=768, token_count=512` 作为 WMPO-aligned 主档位。
   这是目前最保守的架构对齐选择:参数量接近 86M,depth/width 对齐 VideoMAE-base,
   且不引入 token 瘦身带来的数据语义变化。H100 profiler 显示 batch=16 前向峰值约
   4.5 GB,可接受。
2. `num_layers=12, hidden_dim=768, token_count=196` 作为低延迟候选档位。
   它保留 WMPO 的 12x768 主干形态,但显存约降 56%,耗时约降 72%。不能在没有 F1 的情况下
   设为默认,因为 token 瘦身可能丢成功判别需要的空间细节。

不推荐把 `8 x 1024` 作为 WMPO 对齐默认:虽然参数更多,但深度少于 WMPO,且 512-token
版本比 `12 x 768 x 512` 更占参数、显存略高,结构对齐更弱。

## 实测边界

- 实测:参数量、CUDA forward 峰值显存、CUDA forward 平均耗时。
- profiling-only:所有推荐结论都只基于架构贴近度和资源曲线。
- 未实测:val-F1、episode-F1、真实 online RL 成功率。
- 下一步需要补齐 input-token success/failure sidecar 后,按 `4/6/8/12` depth 跑短训,
  再用 best-F1 和 episode-F1 决定是否采用 512-token 主档或 196-token 低延迟档。
