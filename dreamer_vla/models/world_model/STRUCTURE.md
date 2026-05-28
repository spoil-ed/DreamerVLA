# `dreamer_vla/models/world_model` — engineering structure

Inspired by `diffusion_policy/model/`: small, model-agnostic utilities live in `common.py`; shared WorldModel interfaces live in `base_world_model.py`; each concrete WorldModel lives in its own snake_case module.

```
dreamer_vla/models/world_model/
├── base_world_model.py        ← BaseWorldModel + shared latent/loss/actor-adapter API
├── common.py                  ← model-agnostic primitives (norms, MLP, ResMLP, BlockLinear, helpers)
├── dreamerv3_torch.py         ← DreamerV3 / RSSM family building blocks + legacy re-exports
├── tssm_torch.py              ← TSSM family building blocks + legacy re-exports
├── dreamer_v3_pixel_world_model.py
├── dreamer_v3_token_world_model.py
├── dreamer_v3_token_from_pixel_world_model.py
├── dreamer_v3_pixel_rynn_backbone_world_model.py
├── tssm_rynn_backbone_world_model.py
├── tssm_token_rynn_backbone_world_model.py
├── block_linear.py            ← BlockLinear (kept at top level — already imported widely)
├── chameleon_latent_action.py ← (legacy, unrelated to DreamerV3 path)
└── STRUCTURE.md
```

## `base_world_model.py` — shared WorldModel API

| Name | Purpose |
|---|---|
| `BaseWorldModel` | Common `nn.Module` base for concrete WorldModel wrappers |
| `DreamerV3Loss` | Loss + metrics return object used by DreamerV3/TSSM WMs |
| `DreamerV3LatentState` | Single-step RSSM latent state |
| `DreamerV3ActorAdapterMixin` | Shared `mode=...` actor/eval adapter (`encode_latent`, `predict_next`, `observe_sequence`, `actor_input`, etc.) |

## `common.py` — model-agnostic primitives

Truly generic pieces, reusable across any model:

| Name | Purpose |
|---|---|
| `RMSNorm`, `ChannelRMSNorm` | Normalization layers |
| `act` | Activation factory (`silu` / `gelu` / `elu` / `relu` / …) |
| `MLPHead`, `ResBlock`, `ResMLPHead` | MLP family (plain stacked / pre-norm residual) |
| `BlockLinear` | Block-diagonal linear (used by RSSM GRU, also reusable) |
| `_module_dtype`, `_module_device` | Tiny helpers to read a module's compute dtype/device |

All are instantiable via Hydra `_target_`, e.g.:

```yaml
hidden_decoder:
  _target_: dreamer_vla.models.world_model.common.ResMLPHead
  in_dim: 10240
  out_dim: 35840
  layers: 4
  units: 16384
```

## `dreamerv3_torch.py` — DreamerV3 / RSSM family

Everything that's part of the DreamerV3 lineage stays together:

* **Dynamics** — `DreamerV3RSSM` (GRU + block-diagonal)
* **Pixel I/O** — `DreamerV3PixelEncoder/Decoder`, `DreamerV3TokenEncoder/Decoder`
* **Hidden decoders** (pi0 35×1024 action_hidden) — `Pi0StyleHiddenDecoder`, `PerTokenMLPHead`, `FullHiddenSequenceDecoder`
* **Obs encoder** — `_RynnBackboneObsEncoder` (frozen RynnVLA backbone → embed)
* **Heads** — `BinaryRewardHead`, `SymexpTwoHotHead`, `_make_reward_head`, `_reward_loss`/`_pred`
* **Legacy exports** — concrete WMs are lazily re-exported so old Hydra targets under `dreamerv3_torch` still work

Concrete DreamerV3 WMs now live in:

* `dreamer_v3_pixel_world_model.py` — `DreamerV3PixelWorldModel`
* `dreamer_v3_token_world_model.py` — `DreamerV3TokenWorldModel`
* `dreamer_v3_token_from_pixel_world_model.py` — `DreamerV3TokenFromPixelWorldModel`
* `dreamer_v3_pixel_rynn_backbone_world_model.py` — `DreamerV3PixelRynnBackboneWorldModel`

## `tssm_torch.py` — TSSM family

Everything for the Transformer-State-Space-Model variants:

* **Transformer primitives** (faithful TransDreamer port, with TSSM-specific masking) — `_SinusoidalPosEmb`, `_MultiheadAttention`, `_PositionwiseFF`, `_GRUGating`, `_TransformerLayer`, `_Transformer` (= causal transformer cell)
* **Distribution** — `_onehot_st_sample` (manual straight-through OneHotCategorical, matches TransDreamer)
* **Dynamics** — `TSSMDynamic` (1 token/timestep, faithful), `TSSMTokenDynamic` (35 spatial tokens/timestep extension)
* **Latent state** — `TSSMLatentState`, `TSSMTokenLatentState`
* **Legacy exports** — concrete TSSM WMs are lazily re-exported so old Hydra targets under `tssm_torch` still work

Concrete TSSM WMs now live in:

* `tssm_rynn_backbone_world_model.py` — `TSSMRynnBackboneWorldModel`
* `tssm_token_rynn_backbone_world_model.py` — `TSSMTokenRynnBackboneWorldModel`

> Note: the transformer primitives in `tssm_torch.py` *could* in principle be promoted to `common.py`, but they came in 1:1 as a faithful TransDreamer port (with `pre_lnorm`, `dropatt`, `gating` knobs specific to that paper). Keeping them with TSSM keeps the port self-documenting; lifting them later if a third consumer appears is a one-line move.

## Two cfg styles for the WM (both supported)

### (A) Legacy kind-string dispatch — still works

```yaml
world_model:
  _target_: dreamer_vla.models.world_model.dreamerv3_torch.DreamerV3PixelRynnBackboneWorldModel
  obs_dim: 35840
  hidden_decoder_kind: per_token_mlp     # mlp | resnet | pi0_transformer | per_token_mlp
  hidden_decoder_layers: 2
  hidden_decoder_units: 2048
  hidden_decoder_n_tokens: 35
  hidden_decoder_token_dim: 1024
  hidden_decoder_query_dim: 128
```

### (B) LEGO `_target_` composition — diffusion_policy style (preferred for new variants)

```yaml
world_model:
  _target_: dreamer_vla.models.world_model.dreamerv3_torch.DreamerV3PixelRynnBackboneWorldModel
  obs_dim: 35840
  ...
  hidden_decoder:
    _target_: dreamer_vla.models.world_model.dreamerv3_torch.PerTokenMLPHead
    in_dim: 10240
    n_tokens: 35
    token_dim: 1024
    query_dim: 128
    layers: 2
    units: 2048
```

When the cfg supplies a pre-built `hidden_decoder: nn.Module`, the WM uses it directly and the legacy `hidden_decoder_kind` dispatch is bypassed.

**Verified equivalent**:

```
OLD kind-string path: WM total=196.8M decoder=27.54M kind=per_token_mlp
NEW _target_  path:   WM total=196.8M decoder=27.54M kind=PerTokenMLPHead
```

## Workflow: add a new component

* **Truly generic block** (e.g. a new norm layer) → add to `common.py`. Old names stay.
* **A new hidden decoder / encoder** (model-family-specific) → add to the relevant family building-block file (`dreamerv3_torch.py` or `tssm_torch.py`). Old classes stay.
* **A new WM wrapper** → add a new snake_case file named after the class, then lazily re-export it from the relevant legacy family module if old Hydra-style targets need it.
* **Use it** → cfg `_target_: dreamer_vla.models.world_model.<file>.<NewClass>`. No rewrites elsewhere.

## Variants history

See `scripts/wm_variants_v4_v4E/README.md` for the full per-variant index (launch script, config, code path, params). All v4-A through v4-E launch scripts and configs are preserved.

Example LEGO cfg: `configs/example_lego_per_token_mlp.yaml`.
