# WM Training Routes

Active world-model and joint-training routes in this repo.

## Active Baselines

### 1. TransDreamer Token WM (main baseline)

Config:

- `configs/pretokenize_wm_libero_10.yaml`

Route:

```text
image BPE tokens
-> ImageTokenEmbedder
-> ConvEncoderStem
-> TSSMWorldModelTransDreamer
-> next-token decoder (with hseqce / static-vs-dynamic CE diagnostics)
```

This is the current main WM baseline. It supports image-token diagnostics such
as static CE, dynamic CE, static accuracy, and dynamic accuracy.

### 2. TransDreamer Token WM — sequence variant

Config:

- `configs/pretokenize_wm_libero_10_seq_t8.yaml`

Route: same as the main baseline, but trains on contiguous sequences of T=8
frames per sample (`PretokenizeSequenceDataset` +
`pretokenize_wm_sequence_workspace.PretokenizeWMWorkspace`) for sequence-aware
losses.

Built on top of the main baseline; pick this when sequence loss is needed.

### 3. DreamerVLA Joint Training

Configs:

- `configs/dreamer_vla_libero_10_transdreamer.yaml`
- `configs/dreamer_vla_libero_10_transdreamer_vlaactor.yaml`

Route:

```text
per batch:
  1. WM update
  2. DreamerV3-style actor-critic imagination update
     (twohot symlog critic, target critic EMA, percentile-normalised returns)
```

Use this as the downstream policy baseline after choosing a WM checkpoint.

## Deprecated Routes

Kept for old-checkpoint reproducibility:

- `configs/dreamer_vla_libero_10.yaml` — legacy joint training with the
  `TSSMWorldModel` (scalar reward / hidden-space) instead of TransDreamer.
