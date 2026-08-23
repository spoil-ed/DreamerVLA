# DINO Token World Model Reproduction Plan

**Goal:** Reproduce DINO-WM's predictor, conditioning, shifted one-step MSE,
trajectory slicing, epoch loop, validation, and autoregressive rollout over
DreamerVLA's existing OpenVLA-OFT hidden tokens. The visual encoder/token geometry
is replaced by persisted `[256,4096]` tokens, while batch size and learning rate
are intentionally aligned with Dreamer-WM for comparison.

**Integration boundary:** This is a Hydra-selected model and experiment. Choose it
with `train.sh --config dino-wm`; `--config dreamer-wm` selects
`ChunkAwareWorldModel`. It is not yet selected by the online-cotrain recipe; that
promotion remains gated on beating the persistence baseline after a converged
training run.

Model construction parameters live in `configs/worldmodel/{dino-wm,dreamer-wm}.yaml`;
batch size, optimizer, data, and schedule parameters remain in the corresponding
Hydra experiment recipes.

## Implementation

- [x] Copy the DINO frame-block causal transformer, learned positional embedding,
      Conv1d action/proprio embeddings, concatenated conditioning, shifted one-step
      MSE, and rollout into `DinoTokenWorldModel`.
- [x] Preserve DINO math while registering the causal mask as a device-aware buffer
      instead of hard-coding CUDA.
- [x] Apply a fixed per-token LayerNorm at the DreamerVLA token boundary so the
      predictor receives normalized features like DINO's `x_norm_patchtokens`.
- [x] Attribute the MIT-licensed reference implementation and retain its license.
- [x] Add unit tests for the causal mask, positional initialization, shifted targets,
      action selection, conditioning layout, and closed-loop action replacement.
- [x] Numerically compare the copied predictor against the local upstream DINO-WM
      implementation with identical weights and inputs.

## Hydra training route

- [x] Add a dedicated trajectory dataset with the upstream trajectory-level 90/10
      split, seed 42, one-time slice permutation, `frameskip=5`, four model frames,
      concatenated five-step actions, and full-corpus action/proprio statistics.
- [x] Add an isolated official-data experiment with history 3, prediction offset 1,
      depth 6, heads 16, head width 64, MLP width 2048, dropout 0.1, action/proprio
      embedding width 10, AdamW at `3e-5`, FP32, 100 epochs, and per-rank batch 16
      to match the Dreamer-WM recipe.
- [x] Use a dedicated runner with separate predictor and action/proprio AdamW
      optimizers, fixed slice order, full train/valid epochs, and per-epoch resume
      checkpoints.
- [x] Validate the DINO model/data/precision relationship through Hydra and reject
      zero-variance normalization corpora before they can create NaNs.

## Evaluation and acceptance

- [x] Add a fixed-data diagnostic comparing the model's one-step token prediction
      against the last-observation persistence baseline on identical samples.
- [x] Run unit/config smoke tests in the `dreamervla` environment.
- [x] Run a bounded real-data GPU update to verify the complete training and
      evaluation path.
- [x] Verify the bounded checkpoint has finite model and optimizer states and load it
      through the fixed validation-window diagnostic.
- [ ] Run the configured reproduction to convergence. Promotion requires model MSE
      below persistence and model cosine above persistence on the fixed validation
      set.

## Chunk-WM follow-up

- [x] Add Hydra-selected, affine-free per-token LayerNorm to the retained Chunk-WM.
- [x] Apply it exactly once at replay/real-observation boundaries (`chunk_loss`,
      `observe_sequence`, `encode_latent`, `observe_next`, and imagination init).
- [x] Keep autoregressively predicted histories in model space without normalizing
      every predicted step again.
