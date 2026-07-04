# Loop 报告 — EGL 维持性判断 + R4 回退基础设施(restore 脚本)

## 本步目标
1. 用**当前代码**实测判断 collect/cotrain/eval 三处 EGL 能否维持而非崩溃(应用户要求,不依赖旧结论)。
2. 推进 R4:补齐"可回退"硬前提——manifest 驱动的 `restore_from_archive.sh` + 单测。

## 改了哪些文件
- `scripts/restore_from_archive.sh`(新增)—— 以 `docs/superpowers/DEPRECATION-manifest.md` 表格为
  数据源,支持 `--dry-run` / `--all` / 指定原路径;幂等(原路径已在位则跳过);只做 `git mv`。
- `tests/unit_tests/test_restore_from_archive.py`(新增)—— 3 用例:dry-run 行数 == manifest 行数;
  临时 git repo 真实还原 + 二次幂等;指定不匹配路径还原 0 个。
- `docs/superpowers/DEPRECATION-manifest.md`:**未改**。并发会话已在 `30aab68` 提交了等价的 74 行
  中文版;本会话一度误重写为英文版,已 `git checkout 30aab68 --` 恢复原版(外科手术原则:不重写未坏的)。
- 提交 `2baf4b7`(仅上述两个新文件;74 个 staged archive rename 与在途 diff 均未卷入)。

## 验证命令与真实输出
- `pytest tests/unit_tests/test_restore_from_archive.py -q` → `3 passed`。
- `bash scripts/restore_from_archive.sh --dry-run | grep -c '^git mv '` → `74`(== staged rename 数)。
- compose 6 主线 experiment → 全 `OK`;`ruff check` 新测 → passed。
- EGL 实测(当前代码, 单卡 pin, render_backend=egl, dreamer ckpt manual_cotrain_step_5, libero_goal task0, 1 ep):
  - `max_steps=50` → 维持(rc=0, 写出 eval_libero_metrics.json, 0 abort)。
  - `max_steps=300` → 崩溃, **确定性 2/2**(GPU0+GPU3, abort 134 / core dumped, 无 metrics)。
  - gdb: `abort ← libnvidia-eglcore ← mjr_readPixels ← mujoco/_render`。

## 结论
- **EGL 维持性判断(DONE)**:collect / cotrain-real = 可维持(spawn 子进程渲染 + 原生崩溃捕获重生,
  `env_worker.py:535`);imagine/WM = 无渲染;**eval = 长 episode 不可维持**(进程内渲染无隔离,
  单 episode 内 readPixels 累积到驱动 abort)。两个只读追踪确认三处 wiring 都真实 egl 且统一到
  `apply_libero_render_regime`,故崩溃是运行期驱动问题,非 wiring。
- **R4 回退基础设施(DONE)**:restore 脚本 + 单测就位,`restore_from_archive.sh --dry-run` 可列全 74 还原动作。

## 下一步建议
- 修 eval EGL 崩溃:优先给 eval 套 collect 同款**子进程渲染隔离**(复用已验证可维持的机制);
  或修 per-step EGL 资源泄漏;osmesa 仅作 SPEC 授权兜底。**注意 eval runtime 与在途 diff 重叠,需协调。**
- R4 剩余:逐批 `git mv` SPEC §3 尚在原位的 ~80 个非主线文件(experiment/runners/algorithms/models/
  configs 组),每批 grep 确认主线无引用 + 追加 manifest + compose 验证。
- Step 5 文档。

## 残留风险
- 并发会话在改动工作树(56 个在途 modified + 已并发提交 30aab68)——继续只用 `git commit -- <path>`。
- eval EGL 崩溃未修:真实 LIBERO eval(max_steps~300 × 10 task)当前会 SIGABRT,R1 趋势验收因此仍阻塞。
