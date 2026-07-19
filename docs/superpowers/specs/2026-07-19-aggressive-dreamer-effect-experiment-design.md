# Aggressive Packed-PPO Dreamer Experiment Design

## Objective

Add one opt-in, short, aggressive Dreamer experiment for eight H100 GPUs. Improve
effect and actor efficiency by compacting valid PPO samples before policy forward,
while leaving the existing `openvla_libero` recipe and dense actor path unchanged.

## Scope

The change contains four isolated pieces:

1. an Actor-local packed-sample planner and packed PPO path;
2. explicit packed-path metrics and validation;
3. `configs/experiment/openvla_libero_aggressive_packed.yaml`;
4. focused unit, config-isolation, and gated distributed tests.

It does not change EnvGroup, RolloutGroup, ReplayGroup, channels, reward production,
classifier threshold, world-model behavior, checkpoint format, or existing recipes.

## Actor Data Flow

The existing trajectory boundary remains unchanged:

```text
TrajectoryShard
  -> load batch
  -> compute returns, group-relative advantages, and loss mask
  -> build PPO sample plan
  -> policy forward/backward
  -> KL transaction
  -> commit or rollback
```

The new `valid_only` planner operates after the existing loss mask and advantages
exist. It performs these steps per actor rank and PPO epoch:

1. flatten chunk samples exactly as the dense path does;
2. select indices whose current loss mask contains at least one valid token;
3. deterministically shuffle only those indices;
4. partition them as evenly as possible across the existing optimizer-step budget;
5. synchronize the required packed microbatch count across FSDP ranks;
6. pad only the final packed microbatch where collective alignment requires it;
7. exclude padding from loss, KL, entropy, clipping, and metric denominators.

All FSDP ranks execute the same number of forward/backward collectives. A rank with
no local valid sample in a globally active microbatch performs the existing
zero-connected policy forward, preserving collective safety. A microbatch that is
globally empty is not executed.

## Loss Semantics

`valid_only` uses the existing token-mean objective, but normalizes by actual valid
tokens rather than by a count that includes empty microbatches. For each optimizer
step, local loss numerators and valid-token counts are formed from unpadded samples;
the global valid-token denominator is reduced across ranks. The local backward loss
is scaled for FSDP's averaged gradients so the result is the global valid-token mean.

The dense path is not modified. Packing does not change rewards, returns,
advantages, PPO clipping, KL calculation, entropy calculation, or the configured
number of optimizer steps when valid samples exist for every planned step. An empty
planned optimizer step is skipped on every rank and reported explicitly.

## Configuration

Add an opt-in actor field:

```yaml
actor:
  train_cfg:
    ppo_sample_packing: dense  # dense | valid_only
```

The inherited default is `dense`. Validation rejects unknown modes. Only the new
experiment selects `valid_only`.

The aggressive experiment owns these values:

| Configuration path | Value |
|---|---:|
| `run.name` | `openvla_libero_aggressive_packed` |
| `manual_cotrain.global_steps` | `20` |
| `manual_cotrain.checkpoint_every` | `5` |
| `manual_cotrain.eval_interval_global_steps` | `5` |
| `manual_cotrain.eval_initial_global_step` | `true` |
| `manual_cotrain.eval_protocol.num_episodes_per_task` | `10` |
| `manual_cotrain.real_rollout_target_trajectories` | `64` |
| `manual_cotrain.max_policy_kl` | `0.03` |
| `algorithm.group_size` | `16` |
| `algorithm.entropy_bonus` | `1.0e-3` |
| `ray_actor_optimizer.lr` | `1.0e-6` |
| `actor.train_cfg.lr` | `1.0e-6` |
| `actor.train_cfg.algorithm_cfg.ppo_update_epochs` | `2` |
| `actor.train_cfg.ppo_sample_packing` | `valid_only` |

Imagined trajectories remain 1,024, global batch remains 16,384, microbatch remains
8, gradient clip remains 1, and reward/filter/clip/GAE settings remain inherited.

Training is selected explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero_aggressive_packed \
  --wm_ckpt /path/to/wm/checkpoints/latest.ckpt \
  --cls_ckpt /path/to/classifier/checkpoints/latest.ckpt
```

The original `--config openvla_libero` continues to resolve to the unmodified dense
route.

## Metrics

Both paths retain existing metric names. The packed path additionally emits:

- `actor/raw_ppo_samples`;
- `actor/packed_ppo_samples`;
- `actor/packing_keep_fraction`;
- `actor/packing_padding_samples`;
- `actor/packing_padding_fraction`;
- `actor/active_micro_batches`;
- `actor/skipped_empty_optimizer_steps`;
- `actor/global_valid_token_count`;
- `actor/active_ratio_mean`;
- `actor/active_entropy_mean`;
- `actor/active_clip_fraction`.

Existing `actor/ratio_mean`, entropy, KL, and clip metrics keep their dense-path
meaning. Packed-path active metrics use only real valid samples and cannot be
diluted by padding or globally empty microbatches.

## Failure Handling

- Zero global valid samples use the existing skip-update result.
- Non-finite packed loss or metric values fail before optimizer commit.
- Rank-asymmetric sample counts are padded only through the synchronized plan.
- A KL value above `manual_cotrain.max_policy_kl` uses the existing transaction
  rollback.
- Unknown packing modes fail configuration validation before workers launch.

## Verification

Tests must prove:

1. deterministic valid-index selection and partitioning;
2. padding never contributes to the objective or active metrics;
3. packed and dense objectives agree on an all-valid small batch;
4. zero-valid input skips without an optimizer step;
5. rank-asymmetric plans use identical collective counts in a gated two-rank test;
6. the aggressive experiment resolves every value in the configuration table;
7. `experiment=openvla_libero` retains its existing run name, budget, group size,
   learning rate, PPO epoch count, entropy coefficient, real trajectory target,
   KL limit, and dense packing mode;
8. both resolved configs pass `validate_cfg`.

## Evaluation Contract

Resident 100-episode evaluations at steps 0, 5, 10, 15, and 20 are screening
checks. A selected checkpoint is confirmed with 30 episodes per task through the
existing evaluation launcher. Effectiveness requires at least eight percentage
points paired improvement, one-sided paired `p < 0.05`, and no task losing more
than ten percentage points.

## Non-Goals

- No 2x2 experiment or sweep.
- No adaptive classifier threshold or reward reshaping.
- No async pipeline, streaming actor update, or replay redesign.
- No change to existing experiment YAML files.
- No new shell launcher or checkpoint layout.
