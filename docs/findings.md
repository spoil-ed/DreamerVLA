# Findings: Train Action-Chunk WM

## Initial Context

- User asked to train the recently modified action-chunk WM.
- Existing worktree has unrelated deleted docs/untracked archive files; avoid reverting or modifying them.
- Need identify whether the correct path is the Rynn-DINO WM trainer or the `rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed` trainer.

## Entry Point Investigation

- `rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed.yaml` still configures `world_model.action_dim: 7` and uses RSSM-style per-step `actions`.
- `ChameleonLatentActionWMRunner` builds `action_seq` from pretokenized sequence batches and feeds it to the WM as a multi-step action chunk.
- `configs/chameleon_latent_action_wm_libero_goal.yaml` targets `dreamer_vla.runners.ChameleonLatentWMRunner` and inherits the action-sequence model from the LIBERO-10 legacy config, adapted to LIBERO-goal.
- Correction: user clarified the desired route is DINO-WM-based. The Chameleon smoke path was aborted and no Chameleon training is running.

## DINO-WM Action-Chunk Route

- `/mnt/data/spoil/workspace/DreamerVLA` contains the source-complete DINO-WM implementation; this checkout has `dreamer_vla/models/world_model/rynn_dino_wm.py`, `dreamer_vla/models/world_model/rynn_dino_wm_chunk.py`, and `dreamer_vla/runners/rynn_dino_wm_runner.py`.
- `ChunkAwareRynnDinoWMWorldModel` adds `predict_next_chunk()` over a fixed `chunk_size` and does not add trainable parameters relative to `RynnDinoWMWorldModel`, so existing m1024/d6 checkpoints load directly.
- Reused the completed 10k m1024/d6 checkpoint:
  `/mnt/data/spoil/workspace/DreamerVLA/data/outputs/worldmodel/rynn_dino_wm_action_hidden/m1024_d6/resume_gpu7_20260524_213615/ckpt/latest.ckpt`.
- The first foreground debug run verified dataset construction, 4-rank DDP, strict resume at `global_step=10000`, and `ChunkAwareRynnDinoWMWorldModel(chunk_size=5)`.

## 2026-05-27 GPU45 One-Trajectory Pipeline

- User requested the LIBERO-goal single-trajectory VLA-SFT task on "45卡" and said WM and classifier need to be trained first.
- GPU check showed GPUs 4 and 5 free, while 6 and 7 were occupied; use `CUDA_VISIBLE_DEVICES=4,5`.
- WM config: `configs/world_model_dinowm_chunk.yaml`, target `dreamer_vla.runners.RynnDinoWMRunner`, `ChunkAwareRynnDinoWMWorldModel`, `chunk_rollout_chunks=4`, `chunk_rollout_loss_scale=1.0`.
- Existing WM checkpoint available for continuation:
  `data/outputs/worldmodel/dinowm_chunk/20260525_221114/ckpt/latest.ckpt` at approximately step 15000, with logs showing `rollout_chunks: 4.0`.
- Classifier config: `configs/latent_classifier_libero_goal_chunk.yaml`, target `dreamer_vla.runners.LatentClassifierRunner`, chunk granularity with `chunk_size=5`, `window=8`, and current native-chunk `episode_eval_min_steps=7`.
- The older strong classifier checkpoint is under `wmpo_aligned_small_tf_chunk_minsteps32`, but the current config's default `wmpo_aligned_small_tf_chunk` path has no matching completed run yet, so a fresh classifier stage is appropriate.
- VLA-SFT wrapper: `scripts/train_vla_one_traj_45.sh libero_goal`, config `vla_sft_one_trajectory`, one trajectory per task with `trajectory_offset=0`.
- The safe launch pattern for this multi-stage run is a small `run_pipeline.sh` under the run log directory. A direct nested `tmux new-session "bash -lc ..."` string failed because embedded quotes in the expanded script broke the outer shell command.
- Classifier training is intentionally a single-GPU Stage 2 and can show low GPU utilization during evaluation sweeps. The active config runs to `training.max_steps=8000`; the completed run reported `best_episode_f1=0.9704545454545455`, `best_window_f1=0.527363184079602`, and `total_steps=8000`.
- The ordered pipeline did advance into Stage 3 after classifier completion. The VLA log confirms `vla_sft_one_trajectory`, task `libero_goal`, `ngpu=2`, GPUs `4,5`, `trajectories_per_task=1`, and `trajectory_offset=0`.
- Early VLA metrics were written successfully after epoch 0 (`train_vla_loss` about `4.918`, `val_val_ind_loss` about `1.400`, `val_val_ood_loss` about `1.546`), followed by checkpoint writes and continued training into epoch 1.
- Final VLA metrics after 20 epochs improved to `train_vla_loss=3.2249`, `val_val_ind_loss=1.2873`, and `val_val_ood_loss=1.4886`. The final checkpoint set includes `ckpt/latest.ckpt` and `checkpoints/epoch=019-train_vla_loss=3.225.ckpt`.
- Correction from user: "train WM first" meant fresh WM training, not resume from the earlier complete-experiment checkpoint.
- The earlier complete-experiment checkpoint remains intact at `data/outputs/worldmodel/dinowm_chunk/20260525_221114/ckpt/latest.ckpt` with its paired `step_00015000.ckpt`.
- Fresh pipeline root cause/fix: the first fresh tmux launch used base Python and failed before training with `ModuleNotFoundError: No module named 'hydra'`; exporting the `dreamervla` env bin and `PYTHON` in `run_pipeline.sh` fixed the launcher.
- The corrected fresh WM launch is verified no-resume because the log shows `training.resume=false`, `training.resume_path=null`, and progress beginning at `step=0/20000`.
- Eval compatibility fix: VLA checkpoints saved under DDP can store encoder keys as `backbone.module.*`, while `EvalLiberoVLARunner` builds an unwrapped encoder. Normalizing those keys to `backbone.*` lets the completed VLA checkpoint load for single-process LIBERO eval.
- LIBERO-goal eval resource finding: GPUs 6 and 7 were not fully free; each had an existing preprocessing job using about 38GB. The VLA eval worker allocates about 13.7GB, so a safe concurrency limit on those GPUs is 2 eval workers per GPU, not 4.
- Completed VLA checkpoint eval result on LIBERO-goal: all tasks evaluated with 10 episodes/task and `eval.action_steps=5`; every task reported `0/10`, for an aggregate `0/100` success rate.
- Fresh no-resume pipeline result: WM completed from scratch to 20000 steps, classifier completed to 8000 steps with `best_episode_f1=0.9704545454545455`, and one-trajectory VLA-SFT completed 20 epochs with final logged `train_vla_loss=3.2240`, `val_val_ind_loss=1.2874`, and `val_val_ood_loss=1.4889`.
- Fresh no-resume VLA eval result on LIBERO-goal: all 10 tasks evaluated with 10 episodes/task and `eval.action_steps=5`; every task group reported zero successes, for an aggregate `0/100` success rate.
