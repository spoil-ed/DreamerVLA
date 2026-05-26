# Findings: Train Action-Chunk WM

## Initial Context

- User asked to train the recently modified action-chunk WM.
- Existing worktree has unrelated deleted docs/untracked archive files; avoid reverting or modifying them.
- Need identify whether the correct path is the Rynn-DINO WM trainer or the `rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed` trainer.

## Entry Point Investigation

- `rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed.yaml` still configures `world_model.action_dim: 7` and uses RSSM-style per-step `actions`.
- `ChameleonLatentActionWMWorkspace` builds `action_seq` from pretokenized sequence batches and feeds it to the WM as a multi-step action chunk.
- `configs/chameleon_latent_action_wm_libero_goal.yaml` targets `src.workspace.ChameleonLatentWMWorkspace` and inherits the action-sequence model from the LIBERO-10 legacy config, adapted to LIBERO-goal.
- Correction: user clarified the desired route is DINO-WM-based. The Chameleon smoke path was aborted and no Chameleon training is running.

## DINO-WM Action-Chunk Route

- `/home/user01/liops/workspace/DreamerVLA` contains the source-complete DINO-WM implementation; this checkout has `src/models/world_model/rynn_dino_wm.py`, `src/models/world_model/rynn_dino_wm_chunk.py`, and `src/workspace/rynn_dino_wm_workspace.py`.
- `ChunkAwareRynnDinoWMWorldModel` adds `predict_next_chunk()` over a fixed `chunk_size` and does not add trainable parameters relative to `RynnDinoWMWorldModel`, so existing m1024/d6 checkpoints load directly.
- Reused the completed 10k m1024/d6 checkpoint:
  `/home/user01/liops/workspace/DreamerVLA/data/outputs/worldmodel/rynn_dino_wm_action_hidden/m1024_d6/resume_gpu7_20260524_213615/ckpt/latest.ckpt`.
- The first foreground debug run verified dataset construction, 4-rank DDP, strict resume at `global_step=10000`, and `ChunkAwareRynnDinoWMWorldModel(chunk_size=5)`.
