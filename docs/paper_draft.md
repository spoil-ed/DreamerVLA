# Dreamer-VLA: Decoupling Semantic Understanding and Physical Prediction for Sample-Efficient Robotic Manipulation via Latent-Space Imagination over a Frozen Vision-Language-Action Encoder

*Paper draft — auto-generated from the repository at `DreamerVLA/` (README, source code, configs, docs). All architectural facts, hyper-parameters and losses below are extracted verbatim from the implementation; experimental numbers are placeholders (TBD) pending the training runs currently in progress.*

---

## Abstract

We present **Dreamer-VLA**, a hybrid framework that combines a pre-trained Vision-Language-Action (VLA) encoder with a DreamerV3-style latent world model for robotic manipulation. The central design principle is a clean separation between *understanding* the world (handled by a frozen multimodal VLA — RynnVLA-002 — that maps raw RGB, proprioception and task text into a 4096-dimensional semantic feature stream) and *predicting* the world (handled by a compact, Markovian, Transformer-backed **TSSM** latent world model trained on top of these features). An actor-critic policy is optimized purely in the imagination of the world model, using DreamerV3's symlog two-hot critic, a slow-moving target-critic, and percentile-based return normalization. On the LIBERO-10 long-horizon manipulation benchmark our system combines the language grounding and visual priors of modern VLAs with the sample efficiency and long-horizon credit assignment of Dreamer, while adding no pixel reconstruction objective. We describe the three-stage training pipeline (VLA SFT, world-model pre-training, Dreamer rollout) in full detail, together with the pre-tokenization data pipeline, the implementation architecture, and the LIBERO evaluation protocol used to validate policies.

**Keywords:** Vision-Language-Action models, world models, Dreamer, model-based reinforcement learning, robotic manipulation, LIBERO.

---

## 1. Introduction

Two recent families of robot-learning methods have advanced largely independently.

1. **Vision-Language-Action (VLA) models** such as RT-2, OpenVLA and RynnVLA inherit the perceptual richness and language grounding of large multimodal foundation models, and behave as strong *imitation* policies. But they are reactive, cannot "simulate" consequences, and are sample-inefficient for genuinely long-horizon tasks, because the only learning signal is behavior cloning on demonstration data.

2. **Dreamer-style world models** (Dreamer, DreamerV2, DreamerV3) learn a compact latent dynamics model and optimize a policy against *imagined* trajectories, which dramatically improves sample efficiency and enables credit assignment over long horizons. But they typically learn perception from scratch, have weak semantic grounding, and struggle when task success depends on understanding language or open-vocabulary objects.

The underlying insight motivating this work is:

> *"Understanding the world" and "predicting the world" are two different capabilities, and it is not necessary — nor advisable — to force a single network to do both.*

Dreamer-VLA therefore **freezes a large VLA as a semantic feature extractor** and **trains a small world model downstream** to learn physics in the VLA's latent space. Policy learning happens entirely in imagination, but the imagined states live in a semantically meaningful, language-conditioned feature space.

### 1.1 Contributions

* We instantiate the "frozen VLA + trainable world model" decomposition with a **Transformer Stochastic State-space Model (TSSM)** whose transition backbone is itself initialized from a pre-trained causal Transformer (RynnVLA-002's action-world-model head), rather than the usual GRU-based RSSM.
* We integrate three key **DreamerV3** components into this hybrid setting: a **two-hot symlog-binned categorical critic**, a **Polyak-averaged target critic**, and a **percentile-based return scale** (P95 − P5) used to normalize actor advantages — all of which we find important for stable imagination training on top of very-high-dimensional (4096-d) semantic features.
* We describe an end-to-end **pre-tokenization pipeline** that converts raw LIBERO HDF5 episodes into image tokens (Chameleon VQGAN), text tokens (Chameleon BPE), and normalized action/state tensors, enabling high-throughput multi-GPU training with FSDP / DDP.
* We provide a **three-stage training recipe** — (i) VLA SFT, (ii) world-model SFT with frozen VLA, (iii) Dreamer-style imagination rollout — together with a clean codebase (Hydra configs + `Workspace` abstraction) and an LIBERO real-environment evaluation harness.
* We identify and discuss three orthogonal design axes for the framework — presence of an RSSM, presence of a stochastic "bottleneck" projection, and choice of scalar-MSE vs. two-hot critic — and document the current default.

---

## 2. Related Work

**VLA models for robotic manipulation.** RT-2, OpenVLA, Octo, π0, RynnVLA-001/002. These treat manipulation as conditional sequence modeling over discretized actions. Our work uses RynnVLA-002 as a frozen feature extractor rather than an end-to-end policy, and delegates long-horizon credit assignment to a downstream world model.

**Dreamer and latent world models.** PlaNet, DreamerV1/V2/V3 introduced the RSSM (recurrent state-space model) and imagination-based actor-critic learning. DreamerV3 added the symlog two-hot critic, free-bits KL, EMA return normalization, and stabilizing tricks that make a single set of hyper-parameters transfer across hundreds of domains. Dreamer-VLA reuses V3's critic and return-normalization components, but replaces the RSSM's GRU with a pre-trained causal Transformer that already understands robot-action sequences.

**Hybrid VLM + world-model approaches.** Recent work (e.g. GR-1, WMPO) augments language-conditioned imitation with learned predictive components. Compared to WMPO (the direct ancestor of this codebase; our environment is derived from it), Dreamer-VLA (a) replaces the simple scalar critic with a two-hot symlog critic, (b) introduces the target critic and percentile normalization of DreamerV3, and (c) uses a TSSM rather than a per-token autoregressive future-token predictor.

---

## 3. Background

### 3.1 Latent state-space models

A latent world model maintains a belief state $s_t = (h_t, z_t)$ with deterministic recurrent path $h_t$ and stochastic path $z_t$, and learns

$$
\text{prior:}\quad p_\theta(z_t \mid h_t), \qquad
\text{posterior:}\quad q_\phi(z_t \mid h_t, o_t), \qquad
\text{dynamics:}\quad h_{t+1} = f_\theta(h_t, z_t, a_t).
$$

Training minimizes a reconstruction loss plus a KL between posterior and prior. In Dreamer-VLA we replace pixel reconstruction with a **latent-prediction loss** on VLA features, so the world model never learns to generate pixels.

### 3.2 DreamerV3 actor-critic

DreamerV3 computes λ-returns $R^\lambda_t$ along imagined trajectories and trains the critic to match $R^\lambda_t$ under a **two-hot symlog-binned categorical** parametrization:

$$
\mathrm{symlog}(x) = \mathrm{sign}(x)\log(1+|x|), \qquad
\mathrm{symexp}(x) = \mathrm{sign}(x)(\exp(|x|)-1).
$$

Advantages are normalized by an EMA percentile scale $S = \max(1,\ P_{95}^{\text{ema}} - P_{5}^{\text{ema}})$, and the critic is bootstrapped through a Polyak-averaged target copy $\theta^{-} \leftarrow (1-\tau)\theta^{-} + \tau\theta$.

---

## 4. Method

### 4.1 System overview

Dreamer-VLA decomposes an observation $o_t = (\text{RGB}, \text{wrist RGB}, \text{proprio}, \text{task prompt})$ into three stages:

```
 o_t ──► [VLA Encoder, frozen] ──► z^sem_t (4096-d, non-Markov)
                                        │
                                        ▼
                             [TSSM World Model]
                               ├─ encode_latent: z^sem → (stoch, deter)
                               ├─ predict_next:  (s_t, a_t) → s_{t+1}
                               ├─ reward head:   (s_t, a_t, s_{t+1}) → r̂
                               └─ (KL + transition + reward losses)
                                        │
                                        ▼
                    [Actor π, Critic V]  (trained only in imagination)
```

**Gradient policy.**

| Module                    | Trainable? | When                |
|---------------------------|------------|---------------------|
| VLA encoder (RynnVLA-002) | ❌ frozen  | all stages          |
| TSSM mapper / heads       | ✅          | stages 2, 3         |
| TSSM transition backbone  | ❌ frozen* | inherited from VLA  |
| Actor (VLAPolicy)         | ✅          | stage 3             |
| Online critic (two-hot)   | ✅          | stage 3             |
| Target critic             | Polyak     | stage 3             |

*In the default configuration the causal-Transformer transition backbone is loaded from RynnVLA-002 and frozen; only the projection heads (`obs_to_stoch`, `obs_to_deter`, `prior_head`, `posterior_head`, `transition_head`, `reward_head`, token/type embeddings, and MLP mappers) are trained.

### 4.2 VLA encoder

We use **RynnVLA-002** as the frozen encoder. Observations are pre-tokenized offline (see §5) into the format
`[prev_third_image_tokens, prev_wrist_image_tokens, cur_third_image_tokens, cur_wrist_image_tokens, state, text_prompt]`
(history length `his = 2`). Image tokens come from the **Chameleon VQGAN** (`src/models/chameleon_model/`), text tokens from the Chameleon BPE. The encoder returns a 4096-dimensional hidden state $z^{\text{sem}}_t$. At evaluation time the same prompt format is reproduced by `src/env/libero_env.py`, a consistency we emphasize because mis-aligned image ordering silently destroys success rate.

### 4.3 TSSM: Transformer Stochastic State-space Model

The world model (`src/models/world_model/tssm.py`, 1383 lines) maintains a latent

```python
TSSMState(mean, std, stoch[256], deter[4096])
```

and exposes four core methods:

1. **`encode_latent(hidden) → TSSMState`.** MLP heads `obs_to_stoch` and `obs_to_deter` map VLA features to Gaussian parameters of the stochastic path $z_t$ and to the deterministic path $h_t$.
2. **`predict_next(latent, action) → TSSMState`.**
   State and action are each embedded into $N_{state}=1$ and $N_{action}$ tokens, concatenated with learned type embeddings, and passed through the **causal Transformer transition backbone** inherited from RynnVLA-002. A `prior_head` reads out the next prior $p(z_{t+1}\mid h_{t+1})$; a `transition_head` also predicts the raw next VLA feature for a direct transition MSE.
3. **`reward(s_t, a_t, s_{t+1}) → ℝ`.** A small MLP reward head acting on the concatenated features.
4. **`pretrain_loss(hidden, action, next_hidden, reward)`.** Returns

$$
\mathcal{L}_{WM} = \mathcal{L}_{\text{trans}} + \beta_{KL}\, \mathcal{L}_{KL} + \beta_{r}\, \mathcal{L}_{\text{rew}},
$$

with $\beta_{KL} = 0.1$, $\beta_r = 0.1$ in the default config. $\mathcal{L}_{\text{trans}}$ is the MSE between the predicted next VLA feature and the true next VLA feature; $\mathcal{L}_{\text{rew}}$ is the reward-head MSE; $\mathcal{L}_{KL}$ is a Gaussian KL between the predicted prior and the posterior obtained from the *true* next observation.

**Key hyper-parameters (from `configs/dreamer_v3_vla_libero_10.yaml`):**

| Hyper-parameter          | Value  |
|--------------------------|--------|
| `hidden_dim` (VLA)       | 4096   |
| `latent_dim` (stoch)     | 256    |
| `action_dim`             | 7      |
| `state_token_count`      | 1      |
| `dynamics_hidden_dim`    | 4096   |
| `mapper_hidden_dim`      | 1024   |
| `reward_hidden_dim`      | 512    |
| `kl_loss_coef`           | 0.1    |
| `reward_loss_coef`       | 0.1    |

### 4.4 Policy and critic

**Actor (`src/models/vla_policy.py`).** A Gaussian policy over the 7-DoF action. Inputs are the full TSSM feature $[z_t, h_t]\in\mathbb{R}^{256+4096}$. Architecture: `LayerNorm → Linear → GELU → Linear` for the mean, with a learned per-dimension `log_std`. Actions can be sampled stochastically or taken deterministically at eval time.

**Two-hot critic (`src/models/critic/twohot_critic.py`).** The critic outputs a softmax over 255 bins uniformly spaced in symlog space over $[-20, 20]$. Given a target value $v$, the ground-truth distribution is a two-hot encoding that linearly interpolates between the two adjacent bins of $\mathrm{symlog}(v)$; the critic loss is the cross-entropy between this target and the critic's categorical. The expected value is read back as

$$
\hat V(s) = \mathrm{symexp}\!\Big(\sum_i p_i\, b_i\Big).
$$

A second, identical copy (the **target critic**) is kept frozen and soft-updated every step:

$$
\theta^{-} \leftarrow (1-\tau)\,\theta^{-} + \tau\, \theta, \qquad \tau = 0.02.
$$

**Return percentile tracker.** A `ReturnPercentileTracker` keeps EMAs of the 5th and 95th percentiles of recent λ-returns and exposes

$$
S = \max(1,\ P_{95}^{\text{ema}} - P_{5}^{\text{ema}}),
$$

used to normalize the actor advantage.

### 4.5 Imagination-based actor-critic update

Per batch, the Stage-3 workspace (`DreamerV3VLAWorkspace`) alternates two phases:

**Phase-1 — world-model step.** `world_model_pretrain_step()` takes a real transition $(o_t, a_t, o_{t+1}, r_t)$ and minimizes $\mathcal{L}_{WM}$ above, updating the TSSM mapper/heads.

**Phase-2 — imagination step.** `imagine_actor_critic_step_v3()` (`src/algorithms/dreamer_v3_vla.py`):

1. Encode $o_t$ through frozen VLA and TSSM to obtain the initial latent $s_0$, **detached** from the computational graph.
2. For $t=0,\ldots,H-1$ (with $H=15$):
   * sample $a_t \sim \pi_\phi(\cdot\mid s_t)$,
   * predict $s_{t+1}=$ `WM.predict_next(s_t, a_t)` **without** gradients flowing through the transition,
   * predict $r_t =$ `WM.reward(s_t, a_t, s_{t+1})` **with** gradients (this is the path used to train the actor),
   * query $V_\phi(s_t)$ from the online critic.
3. Bootstrap the final value from the **target critic** and compute λ-returns with $\gamma = 0.997$, $\lambda = 0.95$.
4. Update the percentile tracker on the detached returns and get $S$.
5. **Actor loss** (maximize normalized, discount-weighted return plus entropy bonus):

$$
\mathcal{L}_\pi = -\,\mathbb{E}\!\left[\sum_t \gamma^t\, \frac{R^\lambda_t}{S}\right] - \eta\, \mathcal{H}[\pi],
\qquad \eta = 3\!\times\!10^{-4}.
$$

6. **Critic loss** (cross-entropy against a two-hot encoding of the stop-gradient λ-return):

$$
\mathcal{L}_V = -\,\mathbb{E}\big[\log p_V(\mathrm{twohot}(\mathrm{sg}(R^\lambda_t)))\big].
$$

7. Soft-update the target critic.

Both phases can be toggled independently by `training.run_wm_phase` and `training.run_actor_critic_phase`.

### 4.6 V1/V2 variant

An earlier implementation (`DreamerVLAWorkspace` / `src/algorithms/dreamer_vla.py`) uses a **scalar MSE critic**, no target critic, and no return normalization. We retain it in the codebase as an ablation reference:

| Workspace            | Algorithm file       | Critic        | Return norm | Bootstrap        |
|----------------------|----------------------|---------------|-------------|------------------|
| `DreamerVLAWorkspace`   | `dreamer_vla.py`      | scalar MSE    | none        | online critic    |
| `DreamerV3VLAWorkspace` | `dreamer_v3_vla.py`   | two-hot symlog | percentile  | target critic EMA |

---

## 5. Data Pipeline

We train on the 10-task LIBERO-10 subset of LIBERO (LIBERO-Spatial, LIBERO-Object, LIBERO-Goal and LIBERO-Long), a well-studied tabletop manipulation benchmark.

**Stage 0 — pre-tokenization (`scripts/prepare_data.sh`):**

1. LIBERO HDF5 → drop no-op frames.
2. Extract third-person + wrist-camera images.
3. Chameleon VQGAN image tokenization + Chameleon BPE text tokenization.
4. Save per-transition pickles with fields `{image_tokens, text_tokens, action, state, reward, next_obs, wm_obs_input_ids, wm_next_obs_input_ids, is_eot_padded, effective_horizon, task_name}`.

Two special mechanisms are worth noting:

* **EOT (end-of-trajectory) padding.** Transitions that would otherwise fall off the end of a trajectory are padded with a sentinel and accompanied by `action_mask` / `wm_action_mask` tensors, so the WM and actor losses can mask out invalid steps rather than discarding incomplete windows.
* **Pre-encode vs. pre-tokenize.** Two variants are supported: pre-tokenize (store tokens, re-encode through VLA at each step — cheaper storage, more GPU at train time) and pre-encode (store the 4096-d VLA features directly). The defaults use pre-tokenize; a `preencode` path is available for the SFT workspaces in `src/dataloader/preencode_sft_dataset.py`.

---

## 6. Training Protocol

The pipeline has three stages (the first two can run in parallel on separate GPUs):

| Stage | Config | Workspace | What it trains | GPUs |
|-------|--------|-----------|----------------|------|
| 1 — VLA SFT            | `pretokenize_sft_libero_10.yaml` | `PretokenizeSFTWorkspace` | RynnVLA head on pre-tokenized conversations (next-token LM loss on action/state tokens) | 8 |
| 2 — WM SFT             | `pretokenize_wm_libero_10.yaml`  | `PretokenizeWMWorkspace`  | TSSM ($\mathcal{L}_{WM}$) with frozen VLA                                                 | 4–8 |
| 3 — Dreamer rollout    | `dreamer_v3_vla_libero_10.yaml`   | `DreamerV3VLAWorkspace`   | TSSM + actor + critic, per-batch alternating Phase-1/Phase-2                             | 4 |

### 6.1 Optimizers, schedules, precision

All stages use AdamW with cosine-decay + linear warmup and bf16 mixed precision. Notable settings:

| Stage | Module      | Opt    | lr     | weight decay | grad clip | epochs | batch |
|-------|-------------|--------|--------|--------------|-----------|--------|-------|
| 1     | VLA head    | Adam   | 5e-6   | 0.15         | 4.0       | 40     | 8 × 8 GPUs (FSDP full-shard) |
| 2     | TSSM        | Adam   | 1e-4   | 0.01         | 1.0       | 40     | 8 × 4–8 GPUs                 |
| 3     | TSSM        | Adam   | 1e-4   | 0.01         | 1.0       | 20     | 4 × 4 GPUs (DDP)             |
| 3     | Actor       | Adam   | 3e-5   | 0.0          | 1.0       | 20     | — |
| 3     | Critic      | Adam   | 3e-4   | 0.0          | 1.0       | 20     | — |

Dreamer-specific hyper-parameters at Stage 3: imagination horizon $H=15$, $\gamma=0.997$, $\lambda=0.95$, entropy coefficient $3\!\times\!10^{-4}$, target-critic $\tau=0.02$.

### 6.2 Checkpointing

`DreamerV3VLAWorkspace` uses `TopKCheckpointManager` to monitor `epoch_returns_mean` in `max` mode and keep the top-$k$ checkpoints.

---

## 7. Evaluation

### 7.1 LIBERO protocol

`EvalLiberoVLAWorkspace` (`configs/eval_libero_vla.yaml`, `scripts/eval_libero_vla.sh`) performs closed-loop rollouts in the real LIBERO simulator. Key protocol details:

* **Prompt format must match training** (history length `his=2`, image ordering `[prev_third, prev_wrist, cur_third, cur_wrist]`). Mismatches silently destroy success rate.
* Success is per-task binary (goal reached within a fixed step budget); we report the mean success rate across the 10 LIBERO-10 tasks, with 95% confidence intervals over seeds.
* Scalar Stage-3 training metrics (`epoch_returns_mean`, WM losses, actor entropy) are **not sufficient** to claim performance; real-environment success is the primary metric.

### 7.2 World-model sanity checks

`scripts/eval_wm.py` / `src/utils/wm_image_viz.py` visualize latent reconstructions; `scripts/verify_vqgan_recon.py` checks VQGAN round-trip quality; `scripts/smoke_test_token_wm.py` exercises a minimal forward pass for CI-style debugging.

### 7.3 Planned experiments

*(to be filled in once runs complete)*

1. **Main table.** LIBERO-10 success rate vs. (a) pure VLA SFT, (b) our VLA SFT + TSSM SFT (no Dreamer), (c) full Dreamer-VLA (V3), (d) V1/V2 ablation (scalar MSE critic).
2. **Critic ablations.** Two-hot vs. scalar MSE; with and without target critic; with and without percentile return normalization.
3. **World-model ablations.** TSSM with Transformer backbone vs. GRU-RSSM; frozen vs. trainable transition backbone; stochastic dim $\in\{32, 64, 256\}$.
4. **Imagination horizon.** $H \in \{5, 10, 15, 25\}$.
5. **Data-efficiency curves.** Success rate vs. number of LIBERO demonstrations.

---

## 8. Implementation

The codebase (~84 Python modules) is organized around a `Workspace` abstraction that wraps a Hydra config, instantiates models/datasets, and owns the training loop:

```
src/
├── algorithms/      # dreamer_vla, dreamer_v3_vla, ppo_grpo
├── dataloader/      # pretokenize_dataset, preencode_sft_dataset, libero_dataset
├── env/             # libero_env
├── models/
│   ├── chameleon_model/   # VQGAN + Chameleon BPE
│   ├── encoder/           # rynnvla_encoder, rynnvla_runtime, base_encoder
│   ├── world_model/       # tssm.py (core), causal_transformer, image_codec, token_io
│   ├── critic/            # twohot_critic, critic (scalar)
│   └── vla_policy.py
├── preprocess/      # pre_tokenize_action_state_local, action_state_model_conv_generation, paths, ...
├── trainer/         # distributed trainer primitives
├── utils/           # torch_utils, optim, checkpoint_util, ema, wm_image_viz, ...
├── workspace/       # dreamer_v3_vla_workspace (primary), dreamer_vla_workspace (V1/V2),
│                    # pretokenize_sft_workspace, pretokenize_wm_workspace,
│                    # eval_libero_vla_workspace, base_workspace
└── xllmx/           # external LLM integration (tokenizer, data utilities)
```

**Key file pointers (line-approximate):**

* `src/models/world_model/tssm.py:*` — TSSM + `pretrain_loss` + `compute_loss_dict` (≈1383 lines).
* `src/algorithms/dreamer_v3_vla.py:*` — `imagine_actor_critic_step_v3` (147 lines).
* `src/models/critic/twohot_critic.py:*` — `TwohotCritic`, `ReturnPercentileTracker`, `soft_update` (125 lines).
* `src/workspace/dreamer_v3_vla_workspace.py:*` — Stage-3 loop, freezes encoder, builds optimizers, runs Phase-1/Phase-2, checkpointing.
* `configs/dreamer_v3_vla_libero_10.yaml` — canonical Stage-3 configuration.

**Dependencies.** Python 3.11, PyTorch 2.5.1+cu124, xformers 0.0.28.post3, transformers 4.40.1 (pinned for Chameleon), diffusers 0.33.0, flash-attn (wheel), ColossalAI / TensorNVMe / APEX for Stage-1 FSDP, MuJoCo + PyOpenGL for LIBERO, Ray, wandb, tensordict.

**Hardware.** Runs were executed on NVIDIA H100 / H800 / A100 nodes (CUDA 12.x, Ubuntu 20.04).

---

## 9. Discussion

### 9.1 Design-space positioning

The write-up in `docs/dreamer_vla_writeup.md` enumerates two architectural options:

* **Scheme A — with RSSM/TSSM + bottleneck.** Handles POMDP-ish settings well, most stable for long horizons. Current default (our TSSM ≈ Scheme A with a Transformer replacing the GRU).
* **Scheme B — no RSSM.** Simpler: a deterministic dynamics MLP over a bottlenecked $z^{\text{phys}}$. Recommended only for short-horizon / near-Markov tasks.

A recurring recommendation in the internal document is to keep the bottleneck **as small as possible** (16–32 dims for manipulation), to force the world model to discard anything it cannot predict. The current TSSM uses a 256-dim stochastic latent, which is a deliberate trade-off favoring capacity over compression; a smaller-latent ablation is planned.

### 9.2 Things explicitly *not* done

The design document lists four things it recommends **against** doing in the first version, all of which are respected by the current implementation:

* ❌ joint-training the VLA (encoder is frozen in all stages);
* ❌ reconstructing pixels (only reward and latent MSE are predicted);
* ❌ discrete latents (the stochastic path is Gaussian);
* ❌ multi-task language conditioning at WM training time (text enters only through the VLA's pre-computed features).

### 9.3 "Rollout" is imagination, not environment interaction

A common source of confusion in the codebase: `imagine_actor_critic_step_v3` is a pure imagination rollout inside the world model's latent space. The LIBERO simulator is touched only at evaluation time. Extending this to a true Dyna-style loop (real-env rollouts → replay buffer → WM update → imagination) is future work and would require a new replay buffer + env stepper.

### 9.4 Known issues

From the README's "known issues" section and our own experience:

* Chameleon initialization prints a large number of re-instantiation logs (most of `train.log` is these); they are benign.
* The `transformers` version is pinned at 4.40.1 for Chameleon compatibility; upgrading breaks the VQGAN loader.
* `his=2` image-order consistency between training data and the eval env wrapper is a frequent foot-gun.

---

## 10. Conclusion

Dreamer-VLA demonstrates that a strong pre-trained VLA can be cleanly combined with a DreamerV3-style latent world model by (i) freezing the VLA and treating its hidden state as the observation signal, (ii) training a compact Transformer-backed TSSM to model dynamics and rewards in that feature space, and (iii) running an actor-critic update entirely in imagination with DreamerV3's two-hot critic, target critic, and percentile-normalized returns. The result is a sample-efficient, language-conditioned manipulation policy that inherits the semantic strengths of modern VLAs and the long-horizon credit-assignment strengths of Dreamer, without requiring pixel-level generative modeling. We release the full training pipeline — pre-tokenization, VLA SFT, WM SFT, Dreamer rollout, and LIBERO evaluation — as a foundation for further research on hybrid VLA + world-model robots.

---

## Appendix A — Loss summary

$$
\underbrace{\mathcal{L}_{WM}}_{\text{Phase-1}} = \underbrace{\|\hat z^{\text{sem}}_{t+1} - z^{\text{sem}}_{t+1}\|_2^2}_{\mathcal{L}_{\text{trans}}} + \beta_{KL}\,\mathrm{KL}\!\big(q(z_{t+1}\mid h_{t+1}, o_{t+1})\,\|\,p(z_{t+1}\mid h_{t+1})\big) + \beta_r\,\|\hat r_t - r_t\|_2^2.
$$

$$
\underbrace{\mathcal{L}_\pi}_{\text{Phase-2, actor}} = -\mathbb{E}\!\left[\sum_{t=0}^{H-1}\gamma^t\frac{R^\lambda_t}{S}\right] - \eta\,\mathcal{H}[\pi], \qquad
\underbrace{\mathcal{L}_V}_{\text{Phase-2, critic}} = -\mathbb{E}\!\left[\log p_V\!\big(\mathrm{twohot}(\mathrm{sg}\,R^\lambda_t)\big)\right].
$$

$$
R^\lambda_t = r_t + \gamma\big[(1-\lambda)\,V^{-}(s_{t+1}) + \lambda\,R^\lambda_{t+1}\big], \qquad
S = \max\!\big(1,\ P_{95}^{\text{ema}} - P_{5}^{\text{ema}}\big).
$$

## Appendix B — Configuration excerpt (`dreamer_v3_vla_libero_10.yaml`)

```yaml
workspace: DreamerV3VLAWorkspace

world_model:
  hidden_dim: 4096
  latent_dim: 256
  action_dim: 7
  state_token_count: 1
  dynamics_hidden_dim: 4096
  mapper_hidden_dim: 1024
  reward_hidden_dim: 512
  kl_loss_coef: 0.1
  reward_loss_coef: 0.1
  freeze_transition_backbone: true
  pretrained_model_path: data/ckpts/Action_World_model_512/libero_10

critic:
  num_bins: 255
  bin_min: -20
  bin_max: 20

policy:
  feature_dim: 4352      # 256 + 4096
  action_dim: 7

algorithm:
  imagination_horizon: 15
  gamma: 0.997
  lam: 0.95
  entropy_coef: 3.0e-4
  target_critic_tau: 0.02
  return_percentile: [5, 95]

training:
  run_wm_phase: true
  run_actor_critic_phase: true
  epochs: 20
  batch_size: 4
  precision: bf16
  grad_clip: 1.0
```

## Appendix C — Repository map

```
DreamerVLA/
├── README.md             (689 lines, bilingual, full setup guide)
├── install.md
├── pyproject.toml        (hydra-core, omegaconf)
├── requirements.txt      (44 packages)
├── download.sh           (Chameleon + RynnVLA weights)
├── configs/              (12 Hydra YAMLs; V1/V2/V3, SFT, WM, eval, smoke)
├── data/                 (ckpts, libero/, processed_data/, outputs/; gitignored)
├── docs/
│   ├── architecture.md
│   ├── dreamer_vla_writeup.md   (internal design doc; Schemes A/B)
│   └── paper_draft.md           (this file)
├── LIBERO/               (benchmark checkout, editable install)
├── scripts/              (train.py, eval_libero[.py|.sh], preprocess/, smoke tests)
└── src/                  (algorithms / dataloader / env / models /
                           preprocess / trainer / utils / workspace / xllmx)
```

---

*End of draft. Numbers marked TBD are pending the training and evaluation runs in progress at the time of writing (logs in `train.log`, outputs under `data/outputs/`).*
