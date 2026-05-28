# Hidden Reconstruction Target — Structural Analysis & Decoder Bottleneck

Date: 2026-05-21
Author: diagnostic run
Script: `scripts/diagnose_hidden_token_structure.py`
Raw outputs: `data/diagnostics/hidden_token_structure/`

---

## 1. Setup

The DreamerVLA world model reconstructs `action_hidden_states` extracted from
the pi0 VLA action head. The tensor shape is:

```
[T, 5, 7, 1024]   →   flatten   →   [T, 35840]
```

where `35 = time_horizon (5) × action_dim (7)` and each "token" is the
transformer block output for one `(t, joint)` learned query embedding.

This report quantifies the internal structure of this target tensor across
the full LIBERO-goal preprocessed dataset (10 tasks, **433 demos, 52,998
frames**, fp16 → fp32 for stats; 12,000-frame random subsample used for PCA).

> **Marking convention**
> - **(M)** measured directly from data
> - **(I)** inferred from measured quantities + architecture knowledge
> - **(S)** speculative — needs follow-up experiment

---

## 2. Key Measurements

### 2.1 The 35 tokens are statistically near-identical

Pairwise cosine on the per-token marginal mean vectors `μ_i ∈ R^1024`:

| stat | value |
|---|---:|
| min  | 0.9912 |
| p25  | 0.9971 |
| **median** | **0.9986** |
| p75  | 0.9995 |
| max  | 0.9999 |
| mean | 0.9980 |

All 595 off-diagonal pairs have cosine ≥ 0.99. **(M)**

Per-token scalar stats are also nearly flat across token index:

| | range across 35 tokens |
|---|---|
| mean (avg over 1024 dims) | -1.5e-4 ± 3e-6 (essentially zero) |
| std  (avg over 1024 dims) | 0.486 – 0.520 |
| ‖μ_i‖₂                    | 26.66 – 27.35 |

Interpretation: the action-hidden tensor is **post-LayerNorm**: dims have
near-zero mean and similar scale. The slight elevation at the boundary tokens
(t=4, j∈{4,6}) is small but consistent. **(I)**

### 2.2 Structure is "time slab", not "time × joint grid"

Block-averaged distances between pairs of tokens, grouped by whether they
share the same time step `t` or the same joint index `j`:

| pair group | cos(μ_i, μ_j) | Fréchet (diag Σ) |
|---|---:|---:|
| same `t`, different `j` | **0.99964** | **0.71** |
| same `j`, different `t` |   0.99763   |   4.16   |
| cross `t`, cross `j`    |   0.99763   |   4.15   |

→ Same-`t` tokens are ~5.8× closer (in Fréchet) than tokens that differ in
`t`. Same-`j` pairs are **indistinguishable from cross pairs** — the joint
axis carries almost no statistical structure. **(M)**

**The 35-token grid is effectively a 5-time-step sequence, with 7 nearly
redundant copies per step.** **(I)**

### 2.3 The 35840-dim target lives on a ~6-dim manifold

Flat PCA on `(N=12000, D=35840)` via the sample-space gram matrix:

| metric | value |
|---|---:|
| top-5 PCs cumulative variance       | **80.1%** |
| top-19                              | 95.0% |
| top-50                              | 97.9% |
| top-200                             | 99.3% |
| effective rank @ 99%                | 139 |
| **participation ratio** `(Σλ)²/Σλ²` | **5.57** |

→ Roughly **6 effective dimensions** account for 80% of variance; **139**
dimensions for 99%. The 35840 nominal dimensions are massively redundant.
**(M)**

The "5 PCs explain 80%" / "5 time steps" coincidence is consistent with —
but does not prove — that the 5 dominant components are one-per-time-step.
**(S; testable by inspecting loadings)**

### 2.4 Each single 1024-dim token is also low-rank

Per-token PCA (35 separate runs on `[N, 1024]`):

| metric | mean | min | max |
|---|---:|---:|---:|
| participation ratio  | **5.10** | 4.88 | 5.84 |
| effective rank @ 99% | 96       | 72   | 164  |

→ Even within a single 1024-dim token, only **~5 effective directions** carry
meaningful variation. Boundary tokens (last-`t`, joint 4 / 6) are slightly
higher-rank. **(M)**

### 2.5 Per-frame residual cosine — joint redundancy is real, not just LayerNorm bias

For each frame `n` and token pair `(i, j)` we computed
`cos(h_n,i - μ_i, h_n,j - μ_j)` over 4000 random frames (4.9M token pairs):

| pair group | mean cos | median | p25 | p75 |
|---|---:|---:|---:|---:|
| same `t`, different `j` | **0.9959** | **0.9992** | 0.9972 | 0.9997 |
| same `j`, different `t` |   0.9105  |   0.9755  | 0.9322 | 0.9910 |
| cross                   |   0.9104  |   0.9745  | 0.9322 | 0.9906 |

→ **Same-`t` joint tokens carry essentially the same per-sample signal**, not
just the same marginal mean. The 7 joint copies per time step are
information-redundant *at the residual level*, not only at the bias level.
**(M)**

Cross-`t` pairs still show high cosine (0.91 mean / 0.97 median), but with
a much wider distribution (p1 = -0.34): each time step carries its own
per-sample variation that cannot be collapsed.

**Conclusion**: predicting 5 time-step tokens and broadcasting them across
the 7 joints is the correct inductive bias. The §3.3 caveat is closed.

---

## 3. Bottleneck Diagnosis

### 3.1 What is *not* the bottleneck

- **Latent capacity** (M+I): DreamerV3 RSSM default
  `deter=512 + stoch=32×32 = 1024+` is far above the 6-dim manifold (and
  even above the 139-dim 99% bound). Latent dim is not the limiter.
- **Decoder width on the output side** (I): producing 35×1024 outputs from
  ~6 effective directions is mostly broadcasting / repetition. The output
  side does not need 35 independent towers.
- **Joint-axis modeling capacity** (M): joint identity carries no marginal
  signal. Per-joint heads or joint-attention layers spend parameters on
  modeling a degenerate axis.

### 3.2 What *is* the bottleneck

**The bottleneck is the latent → low-D nonlinear manifold map**, *not*
output width or token-wise specialization.

Concretely:

1. **(I)** The reconstruction problem is "map an ~1500-dim latent to a
   ~6-dim curved manifold sitting inside R^35840". The expensive direction
   is the nonlinearity of that manifold, not the surface area you cover.
2. **(I)** The flat MLP baseline (memory obs 2890–2892) and ResNet-wide
   variant outperform the deep / token-based variants. This is consistent:
   width along the trunk lets the decoder approximate the curved 6-manifold,
   while extra depth and per-token specialization add parameters that the
   target geometry does not reward.
3. **(I)** `pi0_transformer`'s 35-token self-attention is paying for
   modeling a 35×35 relational matrix where 28 of those 35 axes are
   collapsed (5 effective × 7 redundant joints). Most attention rows
   duplicate each other.

### 3.3 Caveat resolved: residual cosine confirms joint redundancy **(M)**

Section 2.5 closed this caveat. Same-`t` residuals have cos = 0.996 (median
0.999) — joint tokens are genuinely redundant per-frame, not only on
average. A decoder that produces 5 time tokens and broadcasts them across
the 7 joints is structurally correct (HARD TIE).

---

## 4. Recommendations

Ordered by expected impact-to-effort ratio. Confidence tag in brackets.

### 4.1 Replace 35-token grid with 5-step time sequence (HARD TIE) **[confirmed]**

Implemented as variant **v4-F** (`pi0_time_broadcast`):

- New class `Pi0TimeBroadcastDecoder` in `dreamer_vla/models/world_model/dreamerv3_torch.py`.
- 5 learned time queries; transformer over `[8 memory; 5 queries]` (seq=13
  vs 43 in the old `pi0_transformer`).
- Output `[B, 5, 1024]` is repeated along the joint axis to
  `[B, 5, 7, 1024]`, then flattened to `[B, 35840]`.
- Hard tie verified at the network output (broadcast invariant: 7 joint
  slices are bitwise identical).
- Config: `configs/dreamer_vla_libero_goal_pi0_legacy_action_hidden_head_actor_v4f_time_broadcast.yaml`
- Launch: `scripts/wm_variants_v4_v4E/launch_wm_v4F_time_broadcast.sh`

Param count is unchanged from `pi0_transformer` (~134M for L4 d=1024 mem=8);
the real saving is attention compute (43² → 13², ~91% reduction in the
attention matmul). The structural benefit is removing the 30×30 degenerate
joint-pair attention block from the loss surface.

### 4.2 Stop using full 35×35 self-attention in `pi0_transformer` variant **[high confidence]**

The 35 tokens collapse to 5 effective slots along `t`. A 5-token transformer
over time + joint broadcast is the right inductive bias.

- Either: `[B, 35840] → [B, 5, 1024]`, transformer over T=5, then broadcast.
- Or: keep 35-token attention but mask attention to `(t_q, t_k)` only,
  treating joint as a feature channel.

### 4.3 ResNet-wide is the right default, *but* not deeper **[measured trend]**

Memory obs 2890–2892 already showed `wider > baseline`, `deeper < baseline`.
The structural analysis explains why: width helps fit the curvature of a
narrow manifold; depth without width adds composition that the low-D target
does not need.

Concrete:

- Default the WM decoder to `ResNetWide(L=4, width=16384)` (the v4d
  variant from mem 2864/2877/2884).
- Do **not** stack deeper unless paired with proportionally wider hidden.
- For ablation: try ResNet `L=2, width=32768` — same param budget but
  shallower — and see whether the manifold-curvature hypothesis holds.

### 4.4 Add a 5-component PC loss as an auxiliary signal **[medium confidence]**

Since the top 5 PCs explain 80% of variance, training the decoder to first
match those 5 directions (e.g. project both pred and target onto the top-5
PC basis, MSE on the projection, weighted ~1.0; raw MSE weighted ~0.1)
should accelerate convergence and stabilize the low-D backbone before the
high-frequency residuals come in.

Concrete: compute the top-5 PC basis once from a held-out subset of frames,
save as `data/diagnostics/hidden_token_structure/top5_basis.npy`, add a
config-toggleable `pc5_loss_weight` in the decoder's loss compute.

### 4.5 Latent dimension review **[low confidence; likely no-op]**

Default RSSM `deter=512 + stoch=32×32` gives 1024+ dim — far above the
139-dim 99% bound. There is no reason to enlarge the latent for
reconstruction fidelity.

The opposite question — can latent be **shrunk** without losing actor
performance? — is worth one ablation (`stoch=8×16=128, deter=256`) but
should be deferred until 4.1–4.3 land, since latent shrinkage interacts
strongly with actor learnability.

---

## 5. Follow-up Experiments

Ordered. Each unlocks the next.

### 5.1 Residual cosine distribution **[DONE — see §2.5]**

Result: same-`t` residuals have mean cos 0.9959 / median 0.9992. Hard tie
across joints is validated. Implementation landed as v4-F.

### 5.2 Inspect top-5 PC loadings (~1 min; CPU)

Compute PC vectors `v_k ∈ R^35840` for k=1..5, reshape to `[5, 7, 1024]`,
and visualize the `[5, 7]` energy heatmap. If each PC concentrates on one
`t` slot, 4.1 is justified. If PCs mix time steps, more attention to
sequence structure is needed.

### 5.3 Shared-vs-independent decoder ablation (Phase 2; ~15 min GPU)

Train two small decoders on a 1024-dim summary latent (per-frame mean across
35 tokens) to reconstruct the full `[35, 1024]` target:

- A: shared MLP + token positional embedding
- B: 35 independent MLPs (matched param budget)

If A ≈ B → recommendation 4.1 is confirmed in the wild.

### 5.4 Real WM decoder swap **[v4-F implemented, ready to train]**

Variant `v4-F` (`pi0_time_broadcast`) is implemented and registered.
To launch:

```bash
bash scripts/wm_variants_v4_v4E/launch_wm_v4F_time_broadcast.sh
```

Compare against `v4-D Pi0xform` and `v4-D ResNet wide`:
- final `hidden_rec` loss
- cosine to ground truth
- downstream actor task success on a fixed pi0 frozen-WM eval

---

## 6. TL;DR

- The 35-token, 35840-dim hidden target is **almost entirely a 5-step time
  sequence with 7 redundant joint copies**, living on a **~6-dim curved
  manifold** in R^35840.
- The bottleneck is *not* latent capacity, output width, or per-token
  specialization. It is the **latent → curved low-D manifold map**, which
  rewards width and trunk capacity but not depth or per-token towers.
- The clearest concrete wins:
  1. **DONE** — predict 5 time tokens, broadcast across 7 joints. Residual
     cosine (§2.5) confirms hard tie. Landed as variant **v4-F** /
     `Pi0TimeBroadcastDecoder` / `hidden_decoder_kind=pi0_time_broadcast`.
  2. **DONE** as a side effect of (1) — attention seq dropped from 43 to 13.
  3. Keep ResNet-wide as the default trunk; do not stack deeper without
     proportional width.
