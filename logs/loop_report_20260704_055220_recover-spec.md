# Loop Report: recover-spec

## 本步目标

解除 `SPEC-0` 阻塞：把已存在但位于忽略 plan 路径的已批准 SPEC，恢复到用户指定的唯一事实源路径 `docs/superpowers/specs/2026-07-04-mainline-deprecation-egl-align-design.md`，并用 SPEC Step 1-5 替换 provisional 台账。

## 改了哪些文件

- `docs/superpowers/specs/2026-07-04-mainline-deprecation-egl-align-design.md`: 从 `docs/superpowers/plans/2026-07-04-mainline-deprecation-egl-align.md` 机械复制，恢复到粘贴目标指定的 SPEC 路径。
- `logs/loop_progress.md`: 将 `SPEC-0` 从 `BLOCKED` 更新为 `DONE`，并用 SPEC 的 Step 1-5 替换上一轮 provisional R1-R4 台账。
- `logs/loop_report_20260704_055220_recover-spec.md`: 记录本轮恢复、验证和结论。

## 验证命令与真实输出摘要

```bash
git fetch --all --prune
```

结果：`Fetching origin`，未发现新的远端 SPEC 路径。

```bash
find docs/superpowers -maxdepth 4 -type f -o -type d | sort
```

结果：发现 `docs/superpowers/plans/2026-07-04-mainline-deprecation-egl-align.md`。

```bash
sed -n '1,260p' docs/superpowers/plans/2026-07-04-mainline-deprecation-egl-align.md
```

结果：文件标题为 `主线收敛 · EGL 对齐 · 激进可回退废弃 — 设计规格 (SPEC)`，正文声明“状态：已批准，待实现（本文件是唯一事实源，loop agent 依据它推进）”，并包含 R1-R4、主线 keep 白名单、废弃清单、Step 1-5 及 verify 判据。

```bash
cmp -s docs/superpowers/plans/2026-07-04-mainline-deprecation-egl-align.md docs/superpowers/specs/2026-07-04-mainline-deprecation-egl-align-design.md; printf 'cmp_exit=%s\n' "$?"
```

结果：`cmp_exit=0`，说明恢复后的目标 SPEC 与原 plan 内容完全一致。

```bash
sed -n '1,80p' docs/superpowers/specs/2026-07-04-mainline-deprecation-egl-align-design.md
```

结果：目标路径可读，开头包含 SPEC 标题、日期、分支、主线事实源、参考 RLinf 路径和 R1/R2/R3 内容。

```bash
sed -n '1,220p' logs/loop_progress.md
```

结果：台账已记录 `SPEC-0 DONE`，后续 `Step 1` 到 `Step 5` 均为 `TODO`，与 SPEC 分阶段执行计划一致。

## 结论

`DONE`。本轮只恢复事实源和台账，没有修改训练代码。下一轮可按台账选择 `Step 1`，先置为 `DOING`，再实现 R2 的 32/256/512 锁定与早校验。

## 下一步建议

执行 `Step 1 — 冻结主线契约（R2）`：检查双写点当前值，补齐 `dreamervla/config.py` 告警校验，并用 compose / tiny override 验证。

## 残留风险

- 原始 SPEC 来自 ignored plan 路径；本轮通过完全一致复制恢复到指定 specs 路径，但未删除原 ignored plan。
- 当前 worktree 仍有大量本轮之前的 staged rename 和 modified 文件；本轮没有审查或改动它们。
