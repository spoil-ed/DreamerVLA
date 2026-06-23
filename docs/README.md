# DreamerVLA docs — index & reading guide

Start here. This maps the `docs/` tree and points to the single live to-do list.

## Read first

1. [`repository_structure.md`](repository_structure.md) — code-tree orientation.
2. [`install.md`](install.md) → [`experiment_tutorials/`](experiment_tutorials/) — set up and run an
   experiment end-to-end. [`experiment_tutorials/EXPLAINED.md`](experiment_tutorials/EXPLAINED.md) is the background.
3. [`dreamervla_writeup.md`](dreamervla_writeup.md) + [`performance_optimization_concepts.md`](performance_optimization_concepts.md)
   — the method and the perf-optimization concepts.

## Where things live

| Dir / file | What it is |
|---|---|
| `superpowers/` | Planning workspace (tracked): the live open-items to-do (`plans/2026-06-21-todo-backlog.md`) + plan drafts + eval-alignment findings. The routing snapshot below mirrors the to-do. |
| `plans/2026-06-23-perf-audit-execution-roadmap.md` | Perf-audit status ledger + per-item detail (merged items link into `history/`). |
| `plans/performance_optimization_audit.md` | Perf **findings** ("which line is slow"); the concepts are in `performance_optimization_concepts.md`. |
| `plans/2026-06-23-cotrain-*.md` | Cotrain vectorized-egl rollout + readiness-gate detailed plans. |
| `plans/2026-06-23-perf-prompt-tokenize-cache.md` | The one in-progress perf item. |
| `history/` | **Done** work: execution logs + archived per-item plans & design specs (perf `W*/Q*/H*`, `mem-rl-01`, RUN-01 landed, collector/warmup-cotrain design, …). |
| `experiment_tutorials/` | How to run each recipe (coldstart → warmup → cotrain, world-model, rollout collection). |
| `specs/` | Design specs (train-console output). |
| `model_datasets/` | Dataset cards (OFT / RynnVLA LIBERO). |
| `ray_online_cotrain_backend.md`, `ray_rlinf_alignment_{implemented,todo}.md` | Ray backend: design, shipped, remaining. |
| `PARAMETERS.md`, `data_layout.md` | Config knobs + on-disk data layout. |
| `paper/` | CoRL + NeurIPS LaTeX sources. |

## Everything to do

The authoritative live list is [`superpowers/plans/2026-06-21-todo-backlog.md`](superpowers/plans/2026-06-21-todo-backlog.md)
(in the superpowers planning workspace). It consolidates the backlog + the open cotrain-throughput and
performance-audit items. A high-level routing snapshot, grouped by **where it can run** — the GPU box (the
machine with cards) vs CPU:

### Runnable on the GPU box now
- **RUN-01** — RynnVLA multi-GPU save→resume smoke for the helper-routed online DDP (code landed; the
  smoke is the only open part).
- **COTRAIN-EGL** — verify the egl 4-env rollout (~4×) on a free GPU.
- **X-01②** — unify `online_dreamervla.save_checkpoint` into the BaseRunner envelope (needs a GPU save→resume).
- **RLINF-01 / RLINF-02 remainders** — multi-GPU RNG-exact save→resume; wire `Timers`/`Profiler` into the loops.
- **DECOUPLE-02 / DECOUPLE-03** — env/encoder/policy + action-head decoupling (GPU/LIBERO E2E to verify).
- **Perf, GPU-gated** — W7, H7, H2, W5 single-forward, W2 caller-wiring (see roadmap §5).

### CPU-doable next (unit-testable here)
- **DECOUPLE-04** — small impls (`ReturnPercentileTracker` / `BalancedTerminalDataset` / `soft_update`).
- **Perf** — W8, H3, H6, and the in-progress prompt-tokenize cache.

### Structural refactors
- **MEM-RL-01 remainder + MEM-RL-02** — promote imagination data to an explicit buffer / world-model-as-env.
- **`online_dreamervla.main()` split** (P3).

> The backlog is authoritative; this grouping is a routing aid. When an item lands, move it to
> `history/` and trim its backlog entry to any open remainder (see the recent commits for the pattern).
