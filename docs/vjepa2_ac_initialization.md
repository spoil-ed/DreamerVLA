# PI0.5 / V-JEPA2-AC transition alignment

Baseline before this repair: `680013ba7c95be0fed358cc9773f133e7b670bc2`.
That commit repaired an exactly-zero visual residual head, not long-horizon motion
collapse. Changing initialization cannot repair an already trained checkpoint.

## Representation and architecture

The frozen PI0.5 encoder is unchanged: `[B,T,768,2048]` PaliGemma image-prefix
tokens, three genuine `16×16` image groups (512 valid tokens in this LIBERO route;
the padded third view is masked). Actions are `[B,T,7]`; raw proprio is `[B,T,8]`.
Vision tokens use the existing normalization. Raw state is padded into the existing
10 observation-conditioning slots, preserving the 2058-wide WM observation API.

The visual path is `2048 → Linear → 1024 → 24 AC blocks → norm → Linear → 2048`,
predicting a residual around the current visual tokens. Action/state tokens retain
frame-causal attention. RoPE uses time plus each real view's own spatial coordinates,
with a learned view embedding; it never treats 768 tokens as a fabricated single grid.
Nonspatial representations can still select temporal-only RoPE.

The official 24 blocks, predictor norm, action encoder and state encoder load
302,327,808 parameters exactly, without resizing QKV, MLP or LayerNorm tensors.
The current transition's transferred parameter ratio is **98.6105%**. Incompatible
1408-wide `predictor_embed` / `predictor_proj` are replaced atomically by new
2048↔1024 adapters (including their paired biases). An 8→7 state-input adapter,
identity-initialized action adapter, view embeddings and raw-state output head are
new. Loading prints every loaded, missing, mismatched and unused key and both
parameter counts; it does not silently ignore discrepancies.

## Repairs and why they are necessary

- **State feedback:** previously the latest state was broadcast over all historical
  frames. Now raw state history is aligned with the visual/action history and slid
  together at every step, including chunk boundaries and checkpoint recomputation.
- **State supervision:** the old learned proprio codec had no decoder gradient with
  per-step detachment and zero reconstruction-loss weight. All its tensors remained
  unchanged through checkpoints 1000, 2000, 4000 and the final checkpoint. New AC
  training uses invertible raw pad/slice and an explicitly supervised residual state
  head at every rollout step. It does not feed an untrained decoder back as state.
- **Credit assignment:** truncated BPTT now spans two predictions. Every step is
  supervised; gradients are truncated between segments/chunks, not described as
  equivalent to full 40-step backpropagation. Non-reentrant activation checkpointing
  preserves the entire latent dictionary, including masks and state history.
- **Internal feature collapse:** an unscaled real-data probe produced first-block
  MLP RMS 125 (attention RMS approximately 1); normalized spatial features reached
  mean pair cosine 0.9993. New per-channel LayerScale adapters multiply attention and
  MLP branch *activations*, initially by 0.01. These are 49,152 new trainable parameters;
  all pretrained weight tensors remain intact and all 24 blocks execute. This is a
  deliberate change to residual-branch scaling, not an assertion that the original
  pretrained function is unchanged. `null` restores the unscaled architecture;
  scale 1 is covered by exact functional-equivalence tests.
- **Initialization:** visual and raw-state residual output matrices use nonzero
  normal std `1e-3`, bias zero. First-backward upstream gradients remain nonzero.
- **Objectives:** retain existing latent/chunk/rollout losses. AC additionally gives
  the first clean-context prediction weight 1, first-chunk temporal-difference MSE
  weight 1, and raw-proprio prediction MSE weight 1. Loss scales are configurable.
  Temporal supervision uses actual consecutive target latents and valid-token masks,
  not an arbitrary encouragement to move. Total loss is therefore not directly
  comparable with the old total loss; visual error, persistence and motion remain
  separately logged.
- **Evaluation:** diagnostic and runtime rollout paths now agree on normalization,
  padding masks and per-frame proprio history. Decoder comparisons use the same
  frozen decoder for both rollout→decode and encode→decode, not raw RGB as reference.

The original transition keeps its legacy codec, objectives and transition behavior.
Random-AC and pretrained-AC use identical adapter/backbone parameter-group schedules
for a controlled initialization comparison. LayerScale parameters belong to the
adapter group. The historical `pretrained_backbone` group name denotes the backbone
role in the random-AC control too.

## Validation evidence (2026-09-08)

The full-checkpoint BF16 smoke verifies exact transferred tensors, forward shape,
current-action sensitivity, future-action isolation, nonzero first-update QKV/input/
action gradients, five finite updates and exact full-WM state-dict reload. A two-rank
Gloo smoke also exercised static-graph DDP, checkpointed multi-chunk loss and parameter
synchronization. These are correctness checks, not prediction-quality benchmarks.

Bounded real-data probes use episode 3 for optimization and episode 7 held out,
the same frozen encoder/decoder, and 40-step autoregressive predictions. They are
single-trajectory capacity probes, not estimates of suite-wide generalization.

| Probe | Updates | Train visual MSE | Held-out visual MSE | Decoded motion / reference (train; held-out) |
| --- | ---: | ---: | ---: | ---: |
| Persistence | — | 0.43603 | 0.41408 | 0%; 0% |
| State/history repair only | 20 | 0.44451 | 0.43093 | 3.2%; 3.3% |
| Plus one-step / temporal losses, no LayerScale | 40 | 0.46521 | 0.47061 | 7.6%; 8.0% |
| Plus nonzero LayerScale | 40 | 0.37756 | 0.40766 | 19.7%; 16.0% |

The final probe's normalized predictor spatial pair cosine was 0.67736, versus
0.99933 without LayerScale; centered feature RMS was 0.58548 versus 0.02049 on the
same real context. Raw 40-step proprio MSE was 0.000922 (train), 0.002671 (held-out).
Loss and gradients stayed finite after adapter alignment and backbone unfreezing.
This removes the measured internal collapse and improves the bounded motion/error
probe, but **does not establish realistic full-motion prediction or superiority to
the original WM at matched production steps**. Continue evaluating held-out decoded
motion and error during full training; motion amplitude alone is not accuracy.

Artifacts are under
`/jfs/oss-import/xinglei/pi05_outputs/wm_state_feedback_smoke/`, with subdirectories
`20260908_raw_state_v1`, `20260908_raw_state_v2`, and
`20260908_raw_state_v3_layerscale`. Each contains resolved config, checkpoint,
`results.json` and videos: top encode→decode, middle initial rollout→decode, bottom
trained rollout→decode (base/wrist views side by side). Probe schedules compress
warmup/alignment to 5 updates to exercise unfreezing; production retains its schedule.

## Run and reproduce

Start a fresh WM from official pretrained weights, not an old collapsed WM checkpoint.
Old runs remain readable with their own saved configs; raw-state/LayerScale training
must not silently resume a legacy architecture. Production keeps RGB streaming,
frozen PI0.5, global batch 128 on 8 GPUs, H=3, K=10, four chunks, BF16, and AdamW.
Adapter LR is `1e-4`, backbone LR `1e-5`; warmup is 250 updates, adapter alignment
500 updates, then backbone warmup 500 updates and cosine decay to 10% of peak.

```bash
# A: original transition
.venv-pi05/bin/python -m dreamervla.launchers.train --config wm_pi05_collected_train \
  task=pi05_libero_object profile=sinfra_pi05_object \
  world_model.transition_type=original world_model.transition_init=random

# B: matched AC architecture, random initialization
.venv-pi05/bin/python -m dreamervla.launchers.train --config wm_pi05_collected_train \
  task=pi05_libero_object profile=sinfra_pi05_object \
  world_model.transition_type=vjepa2_ac world_model.transition_init=random

# C: matched AC architecture, pretrained initialization
.venv-pi05/bin/python -m dreamervla.launchers.train --config wm_pi05_collected_train \
  task=pi05_libero_object profile=sinfra_pi05_object \
  world_model.transition_type=vjepa2_ac world_model.transition_init=pretrained \
  world_model.vjepa2_checkpoint_path=/jfs/oss-import/xinglei/DreamerVLA/data/checkpoints/vjepa2-ac-vitg.pt

# Full-checkpoint numerical regression (select an available BF16 GPU).
JAX_PLATFORMS=cpu RUN_VJEPA2_AC_SMOKE=1 VJEPA2_AC_CKPT=/path/to/vjepa2-ac-vitg.pt \
  .venv-pi05/bin/python -m pytest tests/e2e_tests/test_vjepa2_ac_pretrained.py -q -s

# Bounded real-data probe; output directory must not already exist.
JAX_PLATFORMS=cpu VJEPA2_AC_CKPT=/path/to/vjepa2-ac-vitg.pt \
  .venv-pi05/bin/python -m dreamervla.diagnostics.vjepa2_ac_state_smoke \
  --output-dir /path/to/new-probe --decoder-ckpt /path/to/frozen-decoder.ckpt \
  --steps 40 --override optim.world_model.lr_warmup_steps=5 \
  --override optim.world_model.adapter_alignment_steps=5 \
  --override optim.world_model.pretrained_backbone_warmup_steps=5
```

Online W&B is supplied by the user-selected credential file at launch; no secrets
are embedded in code, configuration, documentation or checkpoints.
