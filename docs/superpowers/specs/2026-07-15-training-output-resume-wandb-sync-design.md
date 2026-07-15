# 训练输出、断点恢复与 W&B 上传设计

## 目标

统一 WM warmup、success classifier 训练和 cotrain 的输出目录与恢复行为。恢复范围包括
checkpoint、optimizer、epoch/global step、best metric、classifier threshold、RNG、
TensorBoard 历史和 W&B run identity。同时提供一个 shell 命令，把本地 W&B 目录上传到
online 服务。

本次不处理 replay buffer 的持久化或恢复，现有 replay 行为保持不变。

## 范围与前提

- checkpoint 在 epoch 边界保存。epoch 级任务恢复后从下一个 epoch 开始，不保存
  dataloader batch cursor，也不保存预取队列。
- 保存 RNG state，确保下一个 epoch 的 shuffle、dropout、sampling 和随机数据增强从正确
  状态继续。
- 适用于当前 WM、classifier 和 cotrain 三条训练路线。
- `python -m dreamervla.train` 使用 Hydra 原生 runtime，由 Hydra 自己生成
  `.hydra/config.yaml`、`.hydra/overrides.yaml` 和 `.hydra/hydra.yaml`。
- 继续兼容 canonical 和历史 checkpoint 路径；新产物只写 canonical 路径。
- 显式启用的 top-k checkpoint 和 Hugging Face export 有独立用途，不算无意义重复。
- 不修改、不迁移、不测试 replay buffer 内容、replay sampling state、replay schema 或
  replay restore failure。

## Run root 结构

三条路线共用同一套浅层结构：

```text
<output_root>/<run.name>/<YYYYMMDD_HHMMSS>/
├── checkpoints/
├── tensorboard/
├── wandb/
├── .hydra/
└── run_manifest.json
```

`.hydra/` 完全由 Hydra 管理，DreamerVLA 不复制这些文件，也不自行模拟生成。
`.hydra/config.yaml` 是该次运行的标准应用配置。当前读取根目录 `resolved_config.yaml` 的
诊断和评估代码统一迁移到一个共享 loader：先注册 DreamerVLA resolver，再读取并解析
`.hydra/config.yaml`。根目录不再生成 `resolved_config.yaml`。

`run_manifest.json` 仍然保留，因为其中记录的是 Hydra 不知道的运行时事实：runner identity、
Git revision、分布式拓扑、启用的 logger backend，以及运行时计算出的模型统计。manifest
不得重复完整配置、可由 run root 推导出的目录路径，也不记录 setup 时很快过期的
step/epoch。

每次 invocation 只运行一条路线，因此 `checkpoints/` 只包含当前路线的文件：

- WM warmup：`wm_warmup.ckpt`；
- classifier warmup：`classifier_warmup.ckpt`；
- cotrain：`global_step_<N>/manual_cotrain.ckpt` 和 `latest.ckpt`。

WM 和 classifier 在每个配置指定的 epoch 边界原子覆盖各自 checkpoint，不额外创建
progress checkpoint 目录。Cotrain 保留按 global step 保存的正式 checkpoint，并维护
`latest.ckpt`。文件系统支持时，latest 使用原子 hard link；无法 link 时才使用原子 copy。

W&B 不得生成 `<run_root>/wandb/wandb/`。TensorBoard 也不再复制配置文件到
`tensorboard/config.yaml`。

Top-k retention 和 Hugging Face export 仍由 Hydra 显式控制。启用后可以在
`checkpoints/` 下增加对应产物，但不属于默认 mainline 结构。

## 统一恢复契约

Hydra 负责首次创建 run directory 和保存配置快照。`BaseRunner` 负责推断 resume run root、
提供训练产物的 canonical path、写运行时 manifest，以及延迟创建 metric logger。恢复时，
必须先找到 checkpoint 所属的原 run root，再写任何 DreamerVLA 产物。

每个可恢复 checkpoint 按路线保存实际存在的状态：

- model/module state dict；
- 所有启用的 optimizer state dict；
- 已完成的 epoch 和 global step；
- 适用时保存 best metric 和选中的 checkpoint path；
- 适用时保存 classifier threshold；
- Python、NumPy、PyTorch CPU 和 PyTorch CUDA RNG state；
- 与 replay buffer 无关的路线级轻量进度。

恢复顺序固定为：

1. 解析 resume checkpoint 和原 run root。
2. 构建模型与 optimizer。
3. 恢复模型和 optimizer state。
4. 恢复 epoch/global step、best metric、threshold 和 RNG state。
5. 在 metric logger 创建之前设置第一个有效 metric step。
6. 从下一个 epoch 或 global unit 继续。

loader 继续兼容 `checkpoints/` 和历史 `ckpt/`；writer 不再产生新的 legacy 路径。

## Metric 恢复语义

TensorBoard 和 W&B 共用路线提供的 `metric_resume_step`。这个值表示恢复后第一个允许覆盖
旧值的 step。

TensorBoard 使用 `SummaryWriter(purge_step=metric_resume_step)`。这样可以保留 checkpoint
之前的有效历史，同时隐藏进程崩溃前已经写入、但晚于 checkpoint 的脏尾部。

`metric_resume_step` 必须与该路线实际记录指标的全局横轴一致：

- standalone classifier：恢复出的 global step；
- WM warmup：恢复出的 WM progress step；
- WM 之后的 classifier warmup：`wm_total_steps + restored_classifier_step`；
- cotrain：恢复出的 cotrain global step。

这会修复两个已确认的问题：cotrain 恢复时错误回落到 step 0；联合 warmup 恢复 classifier
时把 component-local step 当成 WM+classifier 共用横轴。

## W&B 本地身份与 online resume

W&B 的 canonical 目录是 `<run_root>/wandb`。DreamerVLA 在
`<run_root>/wandb/run_id.txt` 中持久化并校验唯一 run ID。离线恢复后，每个新进程仍然使用
同一个 ID；W&B offline binary stream 不能原地重开，因此物理上会产生多个 segment，但它们
属于同一个 logical run。

Online 模式也复用该 ID。SDK 支持 `resume_from` 时，传入
`<run_id>?_step=<metric_resume_step>`，先截断 server 上晚于 checkpoint 的历史，再继续写入。
旧版 SDK 不支持该参数时使用 `resume="allow"`，只恢复 run identity，不提供 rewind。

历史目录兼容两种位置：

- `<run_root>/wandb/{offline-run,run}-*/run-*.wandb`；
- `<run_root>/wandb/wandb/{offline-run,run}-*/run-*.wandb`。

新运行只写第一种结构。

## 一条命令上传 offline W&B

仓库提供：

```bash
bash scripts/utils/wandb_sync.sh /path/to/run_root/wandb
```

脚本只接受一个 W&B 目录参数，执行以下工作：

1. 校验目录和 `wandb` CLI。
2. 发现 canonical 与 legacy offline segment。
3. 优先读取 `run_id.txt`；旧 run 没有该文件时，从最早的 segment 推导 canonical ID。
4. 按时间排序 segment。
5. 把第一个未同步 segment 上传到 canonical ID。
6. 后续 segment 使用 `wandb sync --append --id <run_id>`，追加到同一个 online run。
7. 根据 W&B sync marker 跳过已经上传的 segment，因此命令可以安全重复执行。
8. 不删除或改名任何本地文件。

用户只需提前执行一次 `wandb login`，或者提供 `WANDB_API_KEY`。Entity 和 project 默认读取
offline run metadata，本次不增加必填的 entity/project 参数。若以后遇到缺少这类 metadata
的历史文件，再增加可选 override，不提前扩展接口。

以下情况脚本必须以非零状态退出，并给出明确错误：CLI 不存在、目录非法、没有 segment、
run ID 非法或相互冲突、上传失败。脚本不能悄悄把一个本地 logical run 拆成多个 online run。

## 产物生命周期

- 新运行只创建一次 canonical 目录。
- Resume 复用 checkpoint 所属的原 run root，并继续写日志。
- Hydra 在 `.hydra/` 下保存三份原生快照；DreamerVLA 只把非重复的运行时信息写入
  `run_manifest.json`。
- TensorBoard 在原目录创建新的 event file，并按 checkpoint step 逻辑清除脏尾部。
- W&B 在 canonical W&B 目录下创建新的 offline segment，并复用稳定 run ID。
- WM 和 classifier 只在 epoch 边界原子替换各自 checkpoint，不保留临时 progress 文件。

## 错误处理

- Resume checkpoint 缺少当前路线所需的 optimizer、进度字段或模型 state 时直接报错，不能
  静默退化成只加载权重的 warm start。
- 旧 checkpoint 可能没有 RNG state。此类 checkpoint 继续可读，但只发一次兼容性 warning，
  并从新 seed 开始；新 checkpoint 必须包含 RNG state。
- Resume state 恢复完成之前不得初始化 metric logger。现有 guard 扩展到三条路线。
- W&B ID 非法、多个本地 segment 的 ID 冲突或上传失败时，上传脚本立即停止，但不删除或
  改名本地数据。

## 验证范围

测试覆盖：

- canonical 浅层目录，不产生 `wandb/wandb`，TensorBoard 不复制配置；
- Hydra 原生生成 `.hydra/config.yaml`、`.hydra/overrides.yaml` 和 `.hydra/hydra.yaml`，根目录
  不再生成 `resolved_config.yaml`；
- 诊断和评估从 `.hydra/config.yaml` 发现并加载配置；
- canonical 与 legacy resume path 都能复用原 run root；
- WM、classifier 和 cotrain checkpoint round trip，覆盖模型、optimizer 和进度，但不增加
  replay buffer 断言；
- Python、NumPy、PyTorch CPU 以及可用 CUDA device 的 RNG round trip；
- epoch 边界恢复，不需要 dataloader cursor；
- standalone classifier、WM-to-classifier warmup 和 cotrain 的 TensorBoard purge step 正确；
- offline segment 和 online resume 使用稳定 W&B identity；
- 上传脚本覆盖单 segment、多次 resume、legacy 双层目录、已同步 segment、ID 冲突和 CLI
  failure；
- shell syntax、Ruff、格式检查、相关单元测试和 `git diff --check`。

验收范围明确排除 replay buffer 的保存和加载行为。
