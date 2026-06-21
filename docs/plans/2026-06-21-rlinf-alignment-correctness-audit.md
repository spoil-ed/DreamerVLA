# RLinf-alignment correctness audit (DreamerVLA vs RLinf)

Date: 2026-06-21
Reference: the upstream `RLinf` repo (workspace sibling), treated as the
"correct" PPO implementation DreamerVLA aligns with.
Method: 3 read-only comparison passes (PPO loss, advantages/returns, the
actor-critic update loop) against RLinf, each finding cross-checked by hand
before listing. **No code was changed by this audit.**

Severity legend: 🔴 likely bug · 🟠 robustness gap vs RLinf · 🟡 behavior diff
(mostly intentional Dreamer/GRPO design) · ✅ matches RLinf.

## Verified, actionable

### 🔴 A1 — `outcome.py` silently ignores the `actent` entropy key (ALG-03)
`dense.py:93` / `dense_chunk.py:112` read `cfg.get("actent", cfg.get("entropy_coef", 0.0))`,
but `outcome.py:192` reads `entropy_coef` **only**. The `wmpo_outcome` configs expose
the key **`actent`** (e.g. `configs/dreamervla/openvla_oft_wmpo_outcome.yaml:171`), so on
the cotrain-default outcome route any `actent` is a silent no-op (defaults to 0).
- Not biting today (all configs set entropy = 0, matching RLinf's `entropy_bonus: 0`),
  but it's a latent landmine: tuning exploration on the outcome route does nothing.
- RLinf uses a single key `algorithm.entropy_bonus` everywhere (`async_ppo_fsdp_worker.py:334`).
- **Fix (cheap, low-risk):** make `outcome.py` read the same `actent`/`entropy_coef`
  fallback as the dense routes (or, RLinf-aligned, rename all three to `entropy_bonus`).

### 🟠 A2 — no log-ratio clamp before `exp` (numerical stability)
All PPO routes compute `ratio = exp(logp - old_logp)` with **no clamp** on the
log-ratio (`dense.py:357`, `dense_chunk.py:314`, `outcome.py:472`, `relabel.py:56`).
RLinf clamps `log_ratio` before `exp` (`losses.py:241-246`, plumbed from
`clip_log_ratio_min/max`). DreamerVLA aggregates log-probs over the **whole
trajectory** (`log_prob_stack.sum(dim=1)`) before `exp`, so summed log-ratios are
more prone to blow up — the missing guard bites harder here than in RLinf's
per-token setup.
- **Fix (RLinf-aligned, behavior-preserving by default):** add an optional
  `clip_log_ratio_min/max` read; clamp before `exp` (no-op unless configured).

### 🟠 A3 — no dual-clip (`clip_ratio_c`)
`grpo._ppo_clip_term` returns only `max(-adv·ratio, -adv·ratio_clipped)`; there is
no `clip_ratio_c`/`pg_loss3` term. RLinf's embodiment GRPO default is
`clip_ratio_c: 3.0` (`losses.py:256-260`). Without it, a large negative advantage
paired with an exploded ratio yields an unbounded surrogate.
- Caveat: RLinf's *synchronous* embodied path also omits it, so RLinf is itself
  inconsistent. **Fix (optional, RLinf-aligned):** add an opt-in `clip_ratio_c`.

### 🔴/🟡 A4 — replay λ-return: `replay_target_values` computed then discarded
`dreamervla.py:1124-1131` computes `replay_target_values` (critic value on replay
states, with a `repl_slowtar` slow-vs-fast-target branch) and passes it as
`values=` to `compute_replay_lambda_returns` (line 1138) — but that function (and
`compute_lambda_returns`) **ignore `values`**; `_lambda_return_recurrence`
(`:402-407`) uses only `rewards` and `boot`. `boot` is `raw_returns[:,0]` (the
imagined return). `replay_target_values` is used nowhere else.
- **Confirmed:** the `values` parameter is dead in both lambda-return functions,
  and `replay_target_values` is wasted computation (a discarded critic + target
  forward pass on replay states).
- **Ambiguous as a correctness bug:** the docstring (`:448-449`) states `boot`
  *should* be the imagined return seeded from each replay state — which matches the
  code. So this is EITHER (a) vestigial computation to delete (docstring-consistent),
  OR (b) a real bug if the design intent was `boot=replay_target_values` (standard
  DreamerV3 λ-return bootstraps with the critic's per-state values, not a single
  imagined scalar). The elaborate slow/fast target selection that gets discarded is
  what makes (b) plausible. **Needs the author's intent on `repval_loss` to classify.**

## Behavior differences — mostly intentional (document, don't "fix")

- 🟡 **Loss normalization differs per route and from RLinf.** RLinf uses
  `masked_mean` / `masked_mean_ratio` (episode-length normalization). DreamerVLA:
  `dense`/`dense_chunk` plain `.mean()`, `outcome` masked-SUM / `mask_sum_total`,
  `relabel` weighted-sum. `outcome`'s global-mask-sum denominator down-weights
  early-finishing rollouts (RLinf's `masked_mean_ratio` up-weights them) → gradient
  leans toward long/failed rollouts. Worth a deliberate choice.
- 🟡 **No temporal GAE.** RL routes use GRPO group z-score on scalar trajectory
  returns (`grpo._group_advantage`); imagination uses DreamerV3 λ-returns +
  percentile (P95−P5) normalization. Intentional Dreamer/GRPO design; the GRPO math
  matches RLinf's `compute_grpo_advantages`.
- 🟡 **gamma/lambda defaults.** Routes read `ppo_gamma` (default 1.0) / `lam`
  (Dreamer default ≈0.95); RLinf embodiment PPO defaults `gamma: 0.99, gae_lambda:
  0.95`. Not hardcoded — verify your YAML sets these deliberately.
- 🟡 **KL-to-ref as reward penalty (verl-style), k1 signed estimator.** Subtracted
  from return before advantage (`dense.py:297`); the signed `old_lp - ref_lp` can
  *reward* drifting from ref on some samples (acknowledged in code comments). Low
  severity (`kl_coef` defaults 0).
- 🟡 **"Real-data PPO" is a frozen-old-logprob, constant-advantage BC anchor.**
  `_real_relabel_ppo_loss` uses `old_log_prob`/`advantage` frozen from a JSONL trace
  at collection time (`dreamervla_runner.py:551,556`); advantage is a fixed
  `acc - baseline`, never GAE/GRPO, never recomputed. Per its docstrings +
  `validate_real_rollout_relabel.py` this is an intentional sparse positive/negative
  anchor, **not** on-policy PPO — but the name (`real_relabel_ppo_loss`, registry
  alias `ppo`) overstates PPO fidelity. Consider renaming. Also: no value-clipping /
  critic baseline on the real path (RLinf has clipped-Huber value loss,
  `losses.py:347-358`).

## ✅ Matches RLinf
- PPO clip surrogate `max(-adv·r, -adv·r_clipped)` (`grpo._ppo_clip_term`) ==
  RLinf `losses.py:249-255`; asymmetric bounds 0.2/0.28 match embodiment defaults.
- Grad clipping: global-L2 `clip_grad_norm_`, post-backward / pre-step, default 1.0
  — same algorithm & placement as RLinf.
- Entropy/KL signs where they appear (entropy subtracted, KL subtracted from reward).

## Recommended actions
1. **Fix A1 (actent)** — cheap, clear, low risk. Align all three routes to one
   honored entropy key.
2. **Decide A4 (replay bootstrap)** — author confirms intent: delete the dead
   `replay_target_values` (vestigial) OR wire it in as the bootstrap (if it was the
   intended target). Either way removes a wasted critic forward pass.
3. **Optionally add A2/A3** (log-ratio clamp, dual-clip) as opt-in, RLinf-aligned
   numerical guards (default-off keeps current behavior).
4. Treat the 🟡 items as deliberate design choices; at most rename the relabel term
   and document the normalization/KL conventions.
