# Design: Training Console Output Layering + Per-Loop VLA-Improvement Signal

Date: 2026-06-20
Status: Draft (awaiting user review)

## Problem

DreamerVLA training floods the terminal with content that mixes three very
different concerns, and gives no clean per-loop read on whether the VLA policy
is actually improving:

1. **Model/config dumps** — full resolved config `pprint`, parameter counts,
   freeze summaries, tensor shapes. This is reference material, not runtime
   signal; it should live in files, not scroll past on the terminal.
2. **Normal runtime output** — checkpoint loads, seeding confirmations, warmup
   completion, errors. These belong on the terminal.
3. **Flow/phase structure** — there is no visible boundary between the WM
   warmup, classifier warmup, and online cotrain phases.

Separately: the per-loop "is the policy getting better" signal that does exist
(`rollout/success_rate`) is a **cumulative** rate that structurally hides
improvement (early failures sit in the denominator forever).

## Goals

- **Three-layer output separation** matching the three concerns above.
- **`===` phase banners** marking the start/end of each training part.
- **A per-loop VLA-improvement line** that actually reflects recent policy
  quality (windowed success rate + delta + best-so-far), reusing the rollout
  success signal that already exists — no new eval harness.
- Apply consistently across runners via shared helpers in `base_runner`.

## Non-Goals

- Building a new closed-loop eval harness for offline-only runners. The real
  flow already rolls out per loop in the cotrain phase; the existing dedicated
  `EmbodiedEvalRunner` remains the path for standalone closed-loop eval.
- New persistence machinery for config. Hydra (`.hydra/config.yaml`,
  `overrides.yaml`, `hydra.yaml`) plus the existing `resolved_config.yaml` and
  `run_manifest.json` already capture the full architecture spec.
- Re-styling TensorBoard/W&B logging. This is terminal-output only.

## Context: the real training path

The tutorial
`docs/experiment_tutorials/OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md`
runs `experiment=online_cotrain_pipeline_oft_action_hidden`, whose runner is
`OnlineCotrainPipelineRunner`
(`OnlineCotrainPipelineRunner → OnlineCotrainRunner → DreamerVLARunner → BaseRunner`).

Three phases:

| Phase | Code | Loop | Rollout? |
| --- | --- | --- | --- |
| [1] WM warmup | `online_cotrain_pipeline_runner.py:33-48` | `for i in range(steps)` | no |
| [2] Classifier warmup | `online_cotrain_pipeline_runner.py:50-64` | `for i in range(steps)` | no |
| [3] Online cotrain | `online_cotrain_runner.py:416` | `for env_step in range(1, total_env_steps+1)` | yes |

Phase [3] is gated by `online_rollout.total_env_steps` (`pipeline_runner.py:161-166`);
`=0` means warmup-only (exits after [2]).

Success signal already present but cumulative:
`n_success/n_episodes` updated at `online_cotrain_runner.py:442-445`
(`rec["success"]` defined in `online_replay.py:104-110`), surfaced as cumulative
`rollout/success_rate` at `online_cotrain_runner.py:471`. One `env_step` is one
env step, not one episode (episode_horizon≈200, `train_every`=8), so a
per-episode window — not per-step — is the correct unit.

Config dump: `base_runner.py:128-130` `print_config()` → `pprint`, called
unconditionally (main-process only) from `dreamervla_runner.py:106`.

## Design

### Layer 1 — Model/config to files, not terminal

- **Architecture/hyperparameters**: already fully in `resolved_config.yaml` +
  `.hydra/*` + `run_manifest.json`. **No change** beyond suppressing the dump.
- **Suppress the config dump**: gate `BaseRunner.print_config()` behind
  `training.print_config` (default **false**). `resolved_config.yaml` continues
  to be written, so nothing is lost.
- **Runtime-derived model info** (param count, freeze flags): append to the
  existing `run_manifest.json` (it already holds runtime metadata). No new file.
- **Verbose dumps** (full trainable-name list at `online_cotrain_runner.py:142-149`,
  first-batch shape at `:432-438`): removed from the terminal. If ever needed,
  gate behind a debug flag (`training.debug_shapes`, default false) — not built
  now (YAGNI).
- **Terminal keeps one line**: `[ok] model ready · {N} trainable` (this is a
  Layer-2 readiness signal, not a param dump).

### Layer 2 — Normal runtime output stays on terminal

Checkpoint loads, offline-seeding confirmation
(`online_cotrain_pipeline_runner.py:133`), warmup-complete
(`:164`), and errors stay. Standardize light prefixes `[load] / [ok] / [warn]`
for scanability. No semantic change.

### Layer 3 — Phase banners (`===`) + interval metric box

**Phase banners** at each phase boundary
(`online_cotrain_pipeline_runner.py` after `:48`, after `:64`, and at the start
of phase [3] in `online_cotrain_runner.py` near `:403`):

```
================= [1/3] WM WARMUP  ·  256 steps =================
[load] ...
============ [1/3] WM WARMUP — done · wm_loss 0.012 ============

================= [2/3] CLASSIFIER WARMUP · 256 steps =================
============ [2/3] CLASSIFIER WARMUP — done · acc 0.97 ============

================= [3/3] ONLINE COTRAIN · 8000 env steps =================
   (warmup-only: ONLINE COTRAIN — skipped · total_env_steps=0)
============ [3/3] ONLINE COTRAIN — done · succ 0.66 ============
```

**Interval metric box** — replaces the ad-hoc per-update line
(`online_cotrain_runner.py:542-550`); printed every `console.log_every`,
phase-aware:

```
  ╭──────── cotrain · env_step 1600/8000 · 20% · ETA 0:42 ────────╮
  │ VLA    succ@50=0.62 (Δ +0.08 · best 0.66)   return=0.71       │
  │ train  wm=0.182  actor=0.226  |grad|=0.34  cls_acc=0.95       │
  │ data   buf=10000  ep=128  cum_succ=0.55                       │
  ╰───────────────────────────────────────────────────────────────╯
```

Warmup phases (no rollout) omit the `VLA` row and show only the loss row.

### VLA-improvement signal (the `VLA` row)

Windowed, not cumulative:

- New `SuccessTracker` (`deque(maxlen=console.success_window)` + best + last
  printed value), fed `rec["success"]` at the episode-done site
  (`online_cotrain_runner.py:442-445`).
- `succ@N` = success rate over the last N episodes (primary signal).
- `Δ` = change vs the previous box print (answers "is it rising").
- `best` = best windowed rate so far.
- `return` = `rl/returns_mean` (already computed — the imagined-return proxy).
- `cum_succ` (cumulative) kept on the `data` row for reference.

This is where "both signals" (imagined return + real success) converge with
zero extra eval cost.

## Components (minimal, surgical)

- **New** `dreamervla/utils/console.py`: pure functions `phase_banner(...)` and
  `metric_box(header, rows)` — deterministic strings, unit-testable. Value
  formatting reimplemented locally using the same threshold rules as RLinf's
  `print_metrics_table` (sci notation for very small/large, else 3-4 decimals).
- **New** `SuccessTracker` in `dreamervla/runners/online_utils.py` (near the
  rollout code).
- **`base_runner.py`**: gate `print_config` on `training.print_config`; append
  runtime model info into `run_manifest.json`; expose the console helpers so any
  runner can call them.
- **Wiring**: phase banners + metric box in `online_cotrain_pipeline_runner.py`
  and `online_cotrain_runner.py` first; then the rest of the runners (see
  Rollout below).

### Config knobs (Hydra — no hardcoded values)

Per the repo's no-hardcoded-values convention, all thresholds come from config:

- `training.print_config` (default `false`)
- `console.success_window` (default `50` episodes)
- `console.log_every` — box print cadence. Default keeps **current** behavior
  (cotrain: every training update; warmup: every 50 steps); the knob only lets
  you throttle the box if updates are too frequent. No new tuning required.
- `console.banner_width` (default `65`)
- `training.debug_shapes` (default `false`, not wired until needed)

## Rollout across all runners (tiered)

Scope is all runners. Implementation order:

- **Tier 0 — base (one change, benefits all):** `print_config` gate, console
  helpers, `SuccessTracker`, run_manifest model info. Config suppression applies
  to every runner automatically once centralized.
- **Tier 1 — tutorial path:** `online_cotrain_pipeline_runner.py`,
  `online_cotrain_runner.py` (banners + box + VLA row).
- **Tier 2 — active training runners:** `dreamervla_runner`,
  `dreamerv3_pixel_runner`, `dreamerv3_token_runner`, `online_dreamervla`(+`_multiproc`),
  `frozen_wm_actor_critic`, `vla_sft_runner`, `latent_wm_runner`,
  `latent_classifier_runner` (banners + loss/metric box; VLA row only where a
  rollout success signal exists).
- **Tier 3 — collect/eval/ray:** `collect_*_runner`, `embodied_eval_runner`,
  `*_ray_runner`, `rlinf_libero_rollout`, `pretokenize_vla_runner`,
  `openvla_oft_runner` (banners + phase-appropriate box: eval shows success;
  collect shows collected/success counts).

The VLA-improvement row only appears where per-loop rollout/eval success exists;
offline-only phases get banners + a loss box.

## Testing

- Unit: `console.phase_banner` / `metric_box` render exact expected strings.
- Unit: `SuccessTracker` window/Δ/best behavior (including pre-fill and reset).
- Capture stdout of a warmup-only run (`total_env_steps=0`): asserts `===`
  banners present, no config `pprint`, `[ok] model ready` line present.
- Existing `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py` stays
  green.

## Assumptions

- `rl/returns_mean` is an acceptable stand-in for the "imagined return" signal
  in the cotrain box (it is what the runner already computes).
- A per-episode window (default 50) is the right unit given ~1 episode / ~200
  env steps; window size is a Hydra knob if the cadence differs per suite.
- Tier 2/3 wiring follows the same helper pattern; per-runner phase boundaries
  are identified during implementation, not enumerated to the line here.
