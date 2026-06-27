# EGL Cotrain 渲染修复 — 验证与收尾 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 验证已落地的 EGL 渲染修复(多环境默认 osmesa + egl 需 disjoint render_devices + 设备下标诊断)
**在真实 GPU 上确实有效**,并收尾(提交新文件、全量回归)。

**Architecture:** brief `docs/superpowers/specs/2026-06-26-egl-cotrain-render-codex-brief.zh-CN.md` 的 §4
(Task 1–4)**已全部实现**:`dreamervla/utils/egl_device.py`(eglQueryDevicesEXT 计数 + 下标语义诊断)、
`dreamervla/runners/render_device_config.py`(`validate_render_device_pool` disjoint 校验)、
`validate_rollout_cfg` 已接 render/compute devices、num_envs>1 默认 `osmesa`、respawn 已标注为缓解。
§3 根因复审 Codex 也做了(findings 复审补充)。**本计划不重做实现**,只补单测覆盖不到的部分:
GPU 端到端冒烟 + 提交 + 回归。可选:补做 brief §3.2(抓 native abort 决定性字符串,Codex 此前故意跳过)。

**Tech Stack:** PyTorch · Hydra · LIBERO/robosuite/mujoco(EGL/osmesa)· conda env `dreamervla`。

**全局约束:**
- 不重做已实现的 §4;不改 disjoint 校验语义。
- GPU 步骤:先 `nvidia-smi` 选**空闲**卡;render_devices 必须与 compute 卡**不相交**;跑完清理自己的
  进程(mp-spawn 子进程 `pkill -f` 抓不到,按 `nvidia-smi --query-compute-apps=pid` 逐个 kill,别动别人的)。
- 参数走 Hydra config 不硬编码;commit `--signoff`;改动 py 跑 `ruff`;复现/临时脚本不提交。

参考:`docs/superpowers/rlinf_dreamervla_egl_rollout_findings.md`(含 Codex 复审补充)。

---

## File Structure

- (验证)无代码改动;产出冒烟日志 + 结果记录。
- (收尾)提交未跟踪新文件:`dreamervla/runners/render_device_config.py`、`dreamervla/utils/egl_device.py`,
  及相关 modified 文件。
- (可选 Task 3)临时改 `dreamervla/workers/env/env_worker.py::_env_subprocess_main` 加 stderr 落盘
  —— **仅用于抓 native abort,验证后回退,不提交**。
- 结果记录:`docs/superpowers/results/2026-06-27-egl-render-gpu-smoke.zh-CN.md`。

---

## Task 1: GPU 端到端冒烟 —— egl + disjoint render_devices 真的稳吗(核心)

修复的核心前提是"egl 多环境渲染卡与算力卡不相交就稳"。这只在 config 层被单测验证过,**从未端到端
实跑**。本 Task 用真实 GPU 跑一小段,确认不再 native crash。

**Files:**
- 产出:`docs/superpowers/results/2026-06-27-egl-render-gpu-smoke.zh-CN.md`

- [ ] **Step 1: 选空闲卡 + 确认 override 接线**

Run:
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
grep -rnE "render_devices|compute_devices|validate_rollout_cfg" dreamervla/runners/online_cotrain_runner.py | head
```
记录:① 哪些卡空闲;② egl 多环境的 `render_devices` 用哪个 Hydra key 传(`online_rollout.render_devices`?)、
`compute_devices` 怎么解析(训练 DDP 的卡)。**据此定下 disjoint 的卡划分**(如 compute=0,1 / render=2,3)。

- [ ] **Step 2: 反向验证早失败(相交即报错)**

用**相交**的 render/compute 卡跑一次,确认在**启动阶段**就抛 `ValueError`(给"用 osmesa / 设 disjoint"两条出路),
而不是 rollout 中途崩。命令形如(以 Step 1 确认的 key 为准):
```bash
CUDA_VISIBLE_DEVICES=2,3 conda run -n dreamervla python -m dreamervla.train \
  experiment=<egl多环境route> task=libero_goal \
  online_rollout.num_envs=4 online_rollout.render_backend=egl \
  online_rollout.render_devices=[2,3]   # 故意与 compute 相交
```
Expected: 启动即 `ValueError: ... must not overlap compute devices ... or use render_backend=osmesa`。

- [ ] **Step 3: 正例冒烟(disjoint egl 多环境跑一小段)**

render 卡与 compute 卡**不相交**,跑足以覆盖"以前会崩"的步数(参考真实崩溃发生在 rollout 中途):
```bash
conda run -n dreamervla python -m dreamervla.train \
  experiment=<egl多环境route> task=libero_goal \
  online_rollout.num_envs=4 online_rollout.render_backend=egl \
  online_rollout.render_devices=[<render卡>] \
  training.debug=true   # 小规模即可,只看渲染稳定性
  > /tmp/egl_smoke.log 2>&1
```
判定(grep 日志):
- **无** `egl spawn child died` / `respawn` / `exceeded egl_max_respawns` / `EOFError`;
- 有 `egl_device.py` 的诊断行(`eglQueryDevicesEXT count=... MUJOCO_EGL_DEVICE_ID=...`);
- rollout 正常推进 N 步。

- [ ] **Step 4: 对照 —— osmesa 默认路径可跑**

```bash
conda run -n dreamervla python -m dreamervla.train \
  experiment=<多环境route> task=libero_goal \
  online_rollout.num_envs=4 training.debug=true > /tmp/osmesa_smoke.log 2>&1
```
Expected: 默认 `render_backend=osmesa`,正常推进,无 egl 相关崩溃。

- [ ] **Step 5: 写冒烟结果 + 清理 GPU**

写 `docs/superpowers/results/2026-06-27-egl-render-gpu-smoke.zh-CN.md`:命令、卡划分、关键日志摘录、
结论(egl-disjoint 稳/不稳、osmesa 稳)。清理自己的 GPU 进程。**诚实记录**:若某 route/override 跑不起来
或环境缺数据,标注"未跑"而非伪造。

---

## Task 2: 提交未跟踪文件 + 全量回归(收尾)

**Files:**
- Add: `dreamervla/runners/render_device_config.py`、`dreamervla/utils/egl_device.py` 及相关 modified。

- [ ] **Step 1: 全量回归**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests -q`
Expected: PASS（基线 ≈1010 passed / 7 skipped,无回归)

- [ ] **Step 2: ruff**

Run: `conda run -n dreamervla ruff check dreamervla/ tests/unit_tests/`
Expected: 无新错误

- [ ] **Step 3: 提交(需用户确认后再做)**

> 仅当用户同意提交时执行;否则停在这里报告"待提交清单"。
```bash
git add dreamervla/runners/render_device_config.py dreamervla/utils/egl_device.py \
  dreamervla/runners/online_cotrain_runner.py dreamervla/envs/online_egl_venv.py \
  dreamervla/workers/env/env_worker.py dreamervla/runners/online_cotrain_ray_runner.py \
  tests/unit_tests/test_cotrain_render_backend.py tests/unit_tests/test_online_egl_venv.py
git commit --signoff -m "feat: egl multi-env requires disjoint render devices; osmesa default"
```

---

## Task 3(可选,仅当仍要 native abort 决定性字符串)

brief §3.2:Codex 此前**故意跳过**了在同卡算力下抓 native abort(认为空闲卡已证明设备 regime 不可靠)。
若仍要那条决定性报错(`Framebuffer incomplete`/`eglMakeCurrent`/`X Error`/mujoco assert/`CUDA error`):

- [ ] **Step 1: 临时给子进程加 stderr 落盘**

在 `env_worker.py::_env_subprocess_main` 顶部(import robosuite 前)临时:
```python
    import os
    _f = open(f"/tmp/egl_child_{os.getpid()}.log", "w", buffering=1)
    os.dup2(_f.fileno(), 1); os.dup2(_f.fileno(), 2)
```
**仅本地实验,验证后回退,不提交。**

- [ ] **Step 2: 故意同卡算力 + egl 渲染,逼出崩溃**

用 render 卡与 compute 卡**相交**(绕过校验,仅实验)+ 真实算力负载并发,跑到子进程死,读
`/tmp/egl_child_*.log` 抓崩前最后几行 native 报错 + `nvidia-smi`。

- [ ] **Step 3: 把字符串补进 findings,回退临时改动**

记入 `docs/superpowers/rlinf_dreamervla_egl_rollout_findings.md`(§3.3 判定门:co-location 还是版本 bug),
回退 Step 1 的 dup2。

---

## Self-Review

- **Spec coverage**:brief §4 已实现(本计划不重做,仅在 Architecture 注明);本计划覆盖 §4 的**端到端验证**
  (Task 1)、收尾提交+回归(Task 2)、brief §3.2 的可选补做(Task 3)。
- **Placeholder 扫描**:命令里的 `<egl多环境route>`/`<render卡>` 是**需 Codex 按 Step 1 实测填入**的占位,
  已显式标注"以实际为准";非代码占位。
- **诚实性**:Task 1/3 反复要求"跑不起来就标未跑、不要伪造";GPU 清理与不碰他人进程已写入约束。
