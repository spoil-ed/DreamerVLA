# `dreamervla-2` 原生环境复现记录

本文记录 2026-07-17 从仓库基线
`d59458e7e272f99dd3b3a682ddae1585028d8990` 开始实际执行的原生安装、资产准备和
有界训练验收，以及执行中形成的配套修复。它不是根据配置推测出的安装说明；除发布
预算过大的训练时长外，下面记录的路径都真实走过，并保留了 manifest、Hydra 配置和
checkpoint。

## 复现结论与边界

- Conda 环境名为 `dreamervla-2`；激活后以 `${CONDA_PREFIX}` 指代其安装位置，Python
  为 3.11.15。机器上的精确 prefix 保存在本地 host snapshot，避免在版本化文档中固定
  某台机器的挂载根。
- 仓库安装器的 `10_conda_env`、`20_torch`、`30_python_deps`、
  `40_third_party`、`50_special_packages` 和 `60_verify` 均实际执行成功。
- `pip check`、`uv pip check`、8 张 H100 的 CUDA 分配和真实 FlashAttention CUDA
  kernel 均通过，没有发现需要补进 `requirements.txt` 的缺包。
- 公开资产入口完成了固定 revision 的 OpenVLA-OFT 权重、LIBERO Goal 数据、reward
  预处理和 10 个 hidden-token sidecar；sidecar 共包含 432 条 demo、52,639 帧。
- 公开训练入口完成了 WM 一次两卡 DDP update、分类器一次两卡 DDP update及完整
  held-out window 验证，以及 Dreamer 一次真实 LIBERO rollout、一次 imagined rollout、
  7B FSDP PPO update 和完整 checkpoint。
- 发布预算仍是 WM 30 epochs、CLS 8 epochs、Dreamer 20,000 steps。本次将三个阶段
  限制为最小合法更新量，只证明环境和完整数据/训练链能够运行，不声称得到发布模型
  的训练质量。

本机同时存在用户的多卡训练作业，因此资产提取和训练只使用当时空闲的物理 GPU 0、7。
现有作业和仓库内已有的 `third_party/` 修改均未改动。

## 主机基线

| 项目 | 实测值 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.2 LTS |
| 内核 | 6.8.0-106-generic |
| Conda | 26.1.1 |
| GPU | 8 × NVIDIA H100 80GB HBM3 |
| NVIDIA driver | 580.95.05 |
| 数据盘可用空间（开始时） | 约 17 TiB |
| PyTorch | 2.5.1+cu124 |
| torchvision / torchaudio | 0.20.1+cu124 / 2.5.1+cu124 |
| CUDA runtime / cuDNN | 12.4 / 90100 |
| Ray | 2.55.1 |
| flash-attn | 2.7.1.post1 |
| Transformers | 4.40.1，OpenVLA-OFT fork |
| PEFT | 0.11.0 |

## 环境安装

安装器本身支持通过 Hydra 修改环境名。等价的一次性命令是：

```bash
bash scripts/install_env.sh \
  env.CONDA_ENV_NAME=dreamervla-2 \
  env.INSTALL_APT_TOOLS=false
```

本次为了给每一步单独留日志，实际按顺序执行了以下阶段：

```bash
bash scripts/install_env.sh only=[10_conda_env] env.CONDA_ENV_NAME=dreamervla-2
bash scripts/install_env.sh only=[20_torch] env.CONDA_ENV_NAME=dreamervla-2
bash scripts/install_env.sh only=[30_python_deps] env.CONDA_ENV_NAME=dreamervla-2
bash scripts/install_env.sh only=[40_third_party] env.CONDA_ENV_NAME=dreamervla-2
bash scripts/install_env.sh only=[50_special_packages] env.CONDA_ENV_NAME=dreamervla-2
bash scripts/install_env.sh only=[60_verify] env.CONDA_ENV_NAME=dreamervla-2
```

`00_apt_tools` 的真实 `sudo apt` 调用因本会话没有 sudo credential 而退出。随后逐项用
`dpkg-query`、`ldconfig` 和 `apt-get --simulate` 检查；`build-essential`、CMake、
Ninja、Git LFS、FFmpeg、OpenGL、OSMesa 等声明依赖均已安装，模拟结果为 0 个新增或
升级包。因此明确走了安装器的 `INSTALL_APT_TOOLS=false` 路径，而没有把权限问题写成
安装成功。新主机若尚未安装这些系统包，仍应让默认 apt 阶段以有 sudo 权限的账号执行。

安装后验证命令：

```bash
conda activate dreamervla-2
CONDA_ENV_NAME=dreamervla-2 bash scripts/install/60_verify.sh
python -m pip check
uv pip check
```

完整包快照由 `conda list --explicit` 和 `pip freeze` 生成，分别保存在本机忽略目录
`data/reproduction/environment/dreamervla-2/snapshots/conda-explicit.txt` 和
`pip-freeze.txt`。该目录是执行证据，不作为大型运行产物提交到 Git。

## 固定 third-party 源码

仓库已有的部分 third-party checkout 含用户修改，`third_party/openvla-oft` 也不是 Git
checkout。为避免把它们当成固定源码，本次从网络重新克隆到
`/tmp/dreamervla-2-third-party-repro`，逐个 checkout 并验证：

| 源 | revision |
| --- | --- |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| robosuite | `b9d8d3de5e3dfd1724f4a0e6555246c460407daa` |
| robosuite-task-zoo | `74eab7f88214c21ca1ae8617c2b2f8d19718a9ed` |
| robomimic | `d0b37cf214bd24fb590d182edb6384333f67b661` |
| mimicgen | `72bd767c255545f462e7ccfb2731f2e5d4c1d9bb` |
| openvla-oft | `e4287e94541f459edc4feabc4e181f537cd569a8` |
| egl_probe | `3ddf90db69264de2c621af567bfc557849126cff` |
| dlimp_openvla | `040105d256bd28866cc6620621a3d5f7b6b91b46` |
| transformers-openvla-oft | `bc339d9ad707454c0c115970db43c260067c61ab` |

资产工作流新增了 `DVLA_THIRD_PARTY_ROOT` 配置入口，并自动把已验证根目录下的
`openvla-oft` 传给预处理子进程。实际训练还显式把干净 LIBERO/robosuite 等路径放在
`PYTHONPATH` 前部；real env 日志确认从 `/tmp` 固定提交导入 robosuite。

```bash
export DVLA_THIRD_PARTY_ROOT=/tmp/dreamervla-2-third-party-repro
export OPENVLA_OFT_ROOT=${DVLA_THIRD_PARTY_ROOT}/openvla-oft
export PYTHONPATH=${DVLA_THIRD_PARTY_ROOT}/LIBERO:${DVLA_THIRD_PARTY_ROOT}/robosuite:${DVLA_THIRD_PARTY_ROOT}/robosuite-task-zoo:${DVLA_THIRD_PARTY_ROOT}/robomimic:${DVLA_THIRD_PARTY_ROOT}/mimicgen:${DVLA_THIRD_PARTY_ROOT}/openvla-oft:${PYTHONPATH:-}
```

## 资产准备

为不覆盖仓库已有数据，本次使用独立数据根：

```bash
export DVLA_DATA_ROOT=$(pwd -P)/data/reproduction/environment/dreamervla-2/runtime_data
export DVLA_THIRD_PARTY_ROOT=/tmp/dreamervla-2-third-party-repro
export OPENVLA_OFT_ROOT=${DVLA_THIRD_PARTY_ROOT}/openvla-oft

bash scripts/reproduce/01_prepare_assets.sh \
  preprocess.ngpu=2 \
  "preprocess.gpus='0,7'"
```

第一次按发布配置使用 8 卡提取时，GPU 1–6 正被外部任务各占用约 66 GiB，rank 6 在加载
86 MiB allocation 时 OOM。没有终止外部任务；改用空闲 GPU 0、7 后，公开脚本自动跳过
已完成的三个 CPU 阶段、修复原子临时文件并完成全部 sidecar。最终资产占用约为：模型
29 GiB、原始 LIBERO 数据 6 GiB、处理后数据 161 GiB。hidden sidecar 的外部合同是
`token_count=256`、`token_dim=4096`、`history=1`、`chunk_size=1`。

完整 manifest：
`data/reproduction/environment/dreamervla-2/runtime_data/reproduction/manifests/assets.json`。
它记录了模型和数据 revision、third-party SHA、文件大小及 SHA-256，并且状态为
`complete`、profile 为 `cu124-h100-libero-goal-v1`。

## 有界完整训练验收

公开工作流支持每阶段追加 Hydra override；默认 YAML 的发布预算没有改变。本次最终成功
命令如下（已先执行同参数的 `dry_run=true`）：

```bash
bash scripts/reproduce/02_train_dreamer.sh \
  stages.world_model.budget=1 \
  'stages.world_model.overrides=["training.wm_warmup_steps=1","training.warmup_replay_epochs=1","+training.warmup_replay_max_steps=1","+offline_warmup.max_episodes_per_task=1","dataloader.batch_size=1","ngpu=2","gpus=0,7"]' \
  stages.classifier.budget=1 \
  'stages.classifier.overrides=["training.num_epochs=1","training.steps_per_epoch=1","training.batch_size=2","training.val_batch_size=32","training.episode_eval_enabled=false","ngpu=2","gpus=0,7"]' \
  stages.dreamer.budget=1 \
  'stages.dreamer.overrides=["profile=smoke","training.debug=false","manual_cotrain.global_steps=1","manual_cotrain.max_steps_per_rollout_epoch=64","manual_cotrain.real_max_steps_per_rollout_epoch=64","actor.train_cfg.global_batch_size=64","ngpu=2","gpus=0,7"]'
```

结果：

| 阶段 | 实际执行 | 结果 | 选中 checkpoint |
| --- | --- | --- | --- |
| WM | 415.5M 参数，两卡 DDP；10 tasks × 1 episode；1 update | loss `1.916091`，grad norm `3.168638` | `epoch=0000-loss=1.916091.ckpt`，4,952,249,136 bytes，SHA-256 `650cd611a1eb8707e767eff6b0d28c9c3d1b9eb5fb4f3a60e2ec710bde2219eb` |
| CLS | 155.8M 参数，两卡 DDP；1 train step；213 val windows | checkpoint/验证完成；一步训练 F1 为 `0.0` | `epoch=0001-f1=0.000000.ckpt`，1,870,071,878 bytes，SHA-256 `a8e3c494edfc65966da200b75b6a0106ae3dfbdbaf896761b67df4cddf1bf150` |
| Dreamer | real 8 chunks；imagined 8 trajectories/64 chunks；7B FSDP PPO 2 optimizer steps | 128 samples，loss `0.001526`，approx KL `0.006716` | `latest.ckpt`，44,324,316,984 bytes，SHA-256 `e181d2d94de7549ef7ea6ff23e6632446c35d05e4daa457111dba6788151e250` |

三阶段状态、最终命令、选择指标和 checkpoint hash 保存在
`data/reproduction/environment/dreamervla-2/runtime_data/reproduction/manifests/training_state.json`。
重新执行相同命令时，工作流会校验已完成阶段的 budget、文件存在性和 SHA 后跳过或续训。

另外还执行了 CPU tiny cotrain，真实覆盖 real rollout、staged VLA SFT、WM/CLS update、
imagined rollout、PPO 和 checkpoint。其
`data/reproduction/environment/dreamervla-2/cotrain-smoke/checkpoints/latest.ckpt`
可反序列化，`global_step=1`。

## 真实执行中发现并修复的问题

1. 公开资产校验原先只能使用仓库 `third_party/`，无法在不改用户 checkout 的情况下验证
   干净源码。现在 `prepare_assets.yaml` 支持 `DVLA_THIRD_PARTY_ROOT`，并保证预处理实际导入
   的 OpenVLA-OFT 与校验 revision 相同。
2. 公开训练工作流原先不能给各阶段追加独立的有界 Hydra 参数。现在每个 stage 有
   `overrides`，发布默认值保持空列表。
3. tiny classifier 收到 `[B, window, chunk, hidden]` 时只池化一个时间轴，产生 rank-3
   logits；现已池化全部时间维，并添加 4-D 回归测试。
4. zero-GPU Ray placement 只有一个 rollout consumer，却有 real/WM 两个 env rank，导致
   imagined channel 等待；现在 rollout rank 与 env rank 对齐，并有放置回归测试。
5. classifier metric checkpoint 只让 rank 0 进入包含 RNG gather 的保存函数，随后 final
   save 造成 DDP collective 次序错位。现在先广播 top-k 路径，让所有 rank 同步保存，并
   复用同一 step 已写的 latest checkpoint。真实两卡复跑从 `val_final` 正常进入 `done`。
6. 旧 e2e 文件仍引用已从主线删除的 world-model-env experiment；该孤立测试已删除，
   `LatentWorldModelEnv` 由当前 cotrain smoke 覆盖。

运行过程中被 validator 拒绝的配置目录没有删除，而是以
`*.failed-<reason>-<timestamp>` 保存在独立输出根，便于区分“配置前置失败”和“运行时
依赖失败”。这些失败最终都由合法 Hydra 合同解决，没有用关闭 validator 的方式绕过。

## 最终复核命令

```bash
conda activate dreamervla-2
CONDA_ENV_NAME=dreamervla-2 bash scripts/install/60_verify.sh
python -m pip check
uv pip check
python -m pytest tests/unit_tests -q
DVLA_MANUAL_COTRAIN_RAY_SMOKE=1 \
  python -m pytest tests/e2e_tests/test_cotrain_smoke.py -q
ruff check dreamervla tests
```

本次最终复核结果为：`60_verify.sh` 通过，`pip check` 和显式绑定该 Python 的
`uv pip check --python ${CONDA_PREFIX}/bin/python` 均通过（196 个包），Ruff 通过，
unit tests 为 `1576 passed, 3 skipped`，opt-in Ray cotrain e2e 为 `1 passed in
34.32s`。3 个 skip 是测试自身声明的可选条件，不是依赖导入失败。

若要运行发布预算，不传本文的 `stages.*.budget` 和 `stages.*.overrides`，直接使用：

```bash
bash scripts/reproduce/02_train_dreamer.sh
```

## 发布后验收

发布镜像必须来自最终 Git 提交，而不是未提交的工作树。令 `FINAL_SHA` 为 GitHub 远端
分支的 40 位提交，验收时使用不可变标签拉回镜像并核对 OCI revision：

```bash
FINAL_SHA=$(git rev-parse HEAD)
docker pull "spoil/dreamervla:sha-${FINAL_SHA:0:12}"
docker image inspect "spoil/dreamervla:sha-${FINAL_SHA:0:12}" \
  --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}'
docker run --rm "spoil/dreamervla:sha-${FINAL_SHA:0:12}" \
  python -m dreamervla.diagnostics.verify_install
docker buildx imagetools inspect "spoil/dreamervla:sha-${FINAL_SHA:0:12}"
```

验收要求依次为：远端分支解析到 `FINAL_SHA`；镜像 revision 与 `FINAL_SHA` 完全一致；
容器内诊断退出码为 0；Docker Hub 能解析同一不可变标签及其远端 digest。稳定标签
`cu124-h100-v1` 和 `v1` 只在这些检查通过后才视为本次发布完成。
