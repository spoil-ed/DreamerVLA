# π0.5 + LeRobot / RLinf 迁移审查

日期：2026-08-18

## 结论

DreamerVLA 的 `pi05_libero_sft` 现在复用迁入仓库的 RLinf/OpenPI 实现：

```text
physical-intelligence/libero (LeRobot)
  -> RLinf official OpenPI dataloader + LIBERO transforms
  -> Pi05Policy SFT forward
  -> RLinf-style gradient accumulation / FSDP update
  -> DreamerVLA BaseRunner logging + checkpoints/latest.ckpt
```

数据加载不依赖 model/env 名称做分派。Hydra 的 `data.loader._target_` 是唯一构造
入口；model、normalization assets 和 dataset 是三个显式独立路径。迁移后的
运行代码不 import 同级 `RLinf` Python package，提交时不会因工作区相对位置耦合。

## 迁移来源

| DreamerVLA 文件 | RLinf 参考实现 | 迁移内容 |
| --- | --- | --- |
| `dreamervla/dataset/pi05_sft.py` | `rlinf/data/datasets/openpi_rlinf/official_sft_data_loader.py` | OpenPI PyTorch loader、inner torch loader/长度合同 |
| `dreamervla/models/embodiment/pi05/libero_dataconfig.py` | `rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py` | LeRobot repack、LIBERO data/model transforms |
| `dreamervla/models/embodiment/pi05/libero_policy.py` | `rlinf/models/embodiment/openpi/policies/libero_policy.py` | LIBERO observation/action 映射 |
| `dreamervla/models/embodiment/pi05/openpi_config.py` | `rlinf/models/embodiment/openpi/dataconfig/__init__.py` 的 `pi05_libero` 条目 | π0.5 architecture、assets、optimizer 元数据 |
| `dreamervla/runners/vla_sft_training_runner.py` | `fsdp_sft_worker.py`、`fsdp_vla_sft_worker.py` | accumulation、`no_sync`、clip、optimizer/scheduler step、epoch rollover |
| `dreamervla/models/embodiment/pi05/policy.py` | `openpi_action_model.py` | expert-only freeze、SFT batch/reduction、FSDP wrap targets |

保留 DreamerVLA 代码的部分只有外层生命周期：Hydra 构造、`BaseRunner` 指标、
统一 run root、delta checkpoint 和 resume。没有复制 RLinf 的 scheduler/cluster，
因为这是单机离线 SFT，不需要为它引入第二套 public runner backend。

## 固定输入合同

| 输入 | 默认值 | 用途 |
| --- | --- | --- |
| base weights | `lerobot/pi05_base` | 初始化 π0.5 模型 |
| dataset | `physical-intelligence/libero@a4336d589d589045d1c56423ffdf3b88a0e19b1f` | LeRobot SFT 样本；已由上游 no-op-filtered RLDS suites 合并 |
| assets | `RLinf/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero/norm_stats.json` | LIBERO normalization statistics |

默认 JFS 路径分别由 `PI05_BASE_CKPT`、`PI05_LIBERO_DATA`、
`PI05_ASSETS_CKPT` 覆盖。`PI05_LIBERO_CKPT` 仍是评估/rollout 使用的已训练策略，
并默认同时充当 normalization assets 根目录。

数据使用独立配置组 `configs/data/physical_intelligence_libero.yaml`。SFT 和
latent pixel decoder 共用同一个 loader factory；后者仍保持 DDP，SFT 默认使用
RLinf 对应的 FSDP 模式。`train.py` 只为未显式选择 FSDP 的 torchrun 配置自动
启用 DDP，因此不会覆盖 π0.5 的策略。

## 主线兼容边界

| 路线 | 数据/训练行为 | 状态 |
| --- | --- | --- |
| `pi05_libero_sft` | LeRobot + OpenPI loader + FSDP | 本次迁移 |
| `pi05_pixel_decoder` | 同一 OpenPI loader；decoder DDP | 已共用数据合同 |
| π0.5 rollout/eval | LIBERO simulator + trained checkpoint | 不经过离线 loader，保持不变 |
| OpenVLA-OFT collect/warmup/cotrain | HDF5、rollout shards、hidden sidecars、Ray groups | 保持不变 |
| OpenVLA official-data capacity checks | 原有 Hydra dataset target | 保持不变 |

## 下载与运行

受控下载使用固定 revision、`hf-mirror.com`，并在下载子进程中清除大小写全部
proxy 环境变量：

```bash
bash scripts/download_assets.sh 'only=[20_libero_dataset]'
```

8 GPU SFT：

```bash
torchrun --standalone --nproc-per-node=8 -m dreamervla.train \
  experiment=pi05_libero_sft
```

服务器路径覆盖示例：

```bash
PI05_BASE_CKPT=/checkpoints/lerobot/pi05_base \
PI05_LIBERO_DATA=/runtime/data/datasets/lerobot/physical-intelligence/libero \
PI05_LIBERO_CKPT=/checkpoints/RLinf/RLinf-Pi05-LIBERO-SFT \
torchrun --standalone --nproc-per-node=8 -m dreamervla.train \
  experiment=pi05_libero_sft
```

## 审查重点

1. `data.loader` 是否保持纯数据边界，并且只由 Hydra target 选择。
2. 三个输入路径是否符合服务器挂载布局。
3. FSDP 是否只作用于 SFT，且没有改变 OpenVLA Ray 主线。
4. `latest.ckpt` 的 trainable delta、optimizer 和 dataloader/RNG resume 合同是否满足提交需求。

## 验证记录

- Ruff、Python compile、`git diff --check`：通过。
- 无代理 uv 锁定同步：通过，安装 317 packages；Torch `2.11.0+cu128`、
  LeRobot `0.3.3`、`rlinf-openpi 0.1.1`、RLinf `0.4.0`。
- 实际 OpenPI config 构造：通过；`ModelType.PI05`、action horizon `10`，并从
  发布版 assets 读到 `state/actions` normalization statistics。
- π0.5 SFT / pixel decoder / dataset public API 定向测试：`25 passed`。
- 全局 config validation、packaged Hydra、π0.5 rollout/replay、旧 DDP opt-in
  兼容测试：`79 passed`。合计 `104 passed`。
- 当前节点未执行 3.6B 参数完整 GPU forward/backward；最终需在提交 GPU 上做短 smoke run。
