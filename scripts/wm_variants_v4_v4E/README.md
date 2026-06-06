# Archived WM hidden_decoder variants v4 → v4-E

All v4 / v4-D / v4-E world-model pretraining runs over the **legacy** RynnVLA action-head hidden reconstruction target. All start from the v3 ckpt `v3_e7_ft_perwindow.ckpt` except where noted.

These entries describe archived experiments; the referenced launch scripts and route configs live under `scripts/archive/` and `configs/archive/`.

## Variant index

| Tag | Launch script | Config | New code | Notes |
|-----|---|---|---|---|
| **v4-A** | `launch_wm_v4A.sh` | (override-only) | — | early variant, encoder 4096 |
| **v4-B** | `launch_wm_v4B.sh` | `..._v4b.yaml` | — | baseline: MLP 2×8192 |
| **v4-C wide** | `launch_wm_v4Cwide.sh` | `..._v4c_wide.yaml` | — | MLP 2×16384 |
| **v4-C deep** | `launch_wm_v4Cdeep.sh` | `..._v4c_deep.yaml` | — | MLP 3×12288 |
| **v4-D ResNet L4** | `launch_wm_v4D_resnet.sh` | `..._v4d_resnet.yaml` | `_ResBlock`, `ResMLPHead` in `dreamerv3_torch.py` | residual MLP, 4 layers × 8192 |
| **v4-D ResNet L6 deep** | `launch_wm_v4D_resnet6.sh` / `_L6.sh` | `..._v4d_resnet6.yaml` | (same as above) | deeper ResNet |
| **v4-D ResNet wide** | `launch_wm_v4D_resnet_wide.sh` | `..._v4d_resnet_wide.yaml` | (same) | L4 × 16384, ~2.9B decoder |
| **v4-D legacy xform L4** | `launch_wm_v4D_pi0xform.sh` | `..._v4d_pi0xform.yaml` | legacy transformer hidden decoder in `dreamerv3_torch.py` | transformer decoder with learned queries |
| **v4-D legacy xform deep** | `launch_wm_v4D_pi0xform_deep.sh` | `..._v4d_pi0xform_deep.yaml` | (same) | deeper transformer |
| **v4-D V2-mid** | `launch_wm_v4D_v2mid.sh` | (override-only) | — | encoder compresses flattened hidden to 8192 |
| **v4-D bigRSSM** | `launch_wm_v4D_bigRSSM.sh` | (override-only) | — | deter 12288, stoch 48 |
| **v4-D TSSM (flat)** | `launch_wm_v4D_tssm.sh` | `..._v4d_tssm.yaml` | `tssm_torch.py:TSSMRynnBackboneWorldModel` | 1 token / timestep, faithful TransDreamer port |
| **v4-D TSSM token** | `launch_wm_v4D_tssm_token.sh` | `..._v4d_tssm_token.yaml` | `tssm_torch.py:TSSMTokenRynnBackboneWorldModel` | 35 tokens / timestep, spatio-temporal mask |
| **v4-E Per-token MLP (Scenario C)** | `launch_wm_v4E_per_token_mlp.sh` | `..._v4e_per_token_mlp.yaml` | `PerTokenMLPHead` in `dreamerv3_torch.py` | 35 tokens × shared MLP, only 27M decoder params |
| **v4-E legacy xform resurrect** | `launch_wm_v4E_pi0xform_resurrect.sh` | (override-only) | (uses existing legacy transformer hidden decoder) | re-run after the v4-D legacy xform was killed |
| **v4-F Time-broadcast** | `launch_wm_v4F_time_broadcast.sh` | `..._v4f_time_broadcast.yaml` | legacy time-broadcast hidden decoder in `dreamerv3_torch.py` | 5 time queries + broadcast over 7 joints (hard tie); see `docs/archive/hidden_token_structure_report.md` |
| **v4-I overdim** | `launch_wm_v4I_overdim.sh` | `..._v4i_overdim.yaml` | — | keeps 35×1024 output; increases RSSM feature to 15360 and deepens ResMLP decoder to L8×8192; default per-GPU batch size 96 |

## Diff diagnostics

All under `scripts/wm_diff_v4/`. Loads a WM ckpt + the matching cfg, runs `step_with=wm + chunk_replay`, measures hidden cos / chunk_l1 / actor success.

## Hidden-decoder kinds (in `dreamerv3_torch.py`)

| `hidden_decoder_kind` | Class | Params (for feature → flattened hidden) |
|---|---|---|
| `mlp` | `MLPHead` (plain) | ~400M (L2×8192) — ~2.9B (L4×16384) |
| `resnet` | `ResMLPHead` (skip connections) | ~900M (L4×8192) — ~2.9B (L4×16384) |
| legacy transformer | learned queries + transformer | ~80M (L4 d_model=1024 mem=8) |
| `per_token_mlp` | `PerTokenMLPHead` (shared MLP across 35 tokens) | **~27M** (L2×2048 q=128) ⭐ |
| legacy time-broadcast | 5 time queries + broadcast over 7 joints | ~80M; same transformer scale but attention seq 13 vs 43 (-91% attn ops) |
