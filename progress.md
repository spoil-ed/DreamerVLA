# Progress: Train Action-Chunk WM

## 2026-05-25

- Started investigation for the action-chunk WM training entry point.
- Created lightweight planning files for durable command/log tracking.
- Identified the likely action-chunk WM entry point as `configs/chameleon_latent_action_wm_libero_goal.yaml` via `ChameleonLatentActionWMWorkspace`.
- User clarified the target is the DINO-WM-based route; stopped the Chameleon smoke/debug path.
- Verified the DINO-WM chunk-aware route in `/home/user01/liops/workspace/DreamerVLA`:
  `world_model._target_=src.models.world_model.rynn_dino_wm_chunk.ChunkAwareRynnDinoWMWorldModel`
  with `+world_model.chunk_size=5`.
- Launched formal training in tmux session `dino_wm_chunk_20260525_133737` on GPUs `4,5,6,7`.
- Output directory:
  `/home/user01/liops/workspace/DreamerVLA/data/outputs/worldmodel/rynn_dino_wm_action_hidden/chunkaware_m1024_d6_resume10k_bs80_20k/20260525_133738`.
- Log file:
  `/home/user01/liops/workspace/DreamerVLA/data/outputs/logs/worldmodel/rynn_dino_wm/chunkaware_m1024_d6_resume10k_bs80_20k_gpu4567_20260525_133737.log`.
- Early log check: resumed from the existing m1024/d6 checkpoint at `global_step=10000`, started 4-rank DDP training toward `training.max_steps=20000`, and reached at least `step=10020` with all GPUs 4-7 active.
- Updated the DINO-WM tqdm metrics so the progress bar shows `wm`, `next`, `roll`, `rew`, and `acc` instead of the pixel-WM-only `dyn/mse/psnr/rec/rep` zeros.
- Restarted the run with patched metrics in tmux session `dino_wm_chunk_metrics_20260525_135027`.
- New log file:
  `/home/user01/liops/workspace/DreamerVLA/data/outputs/logs/worldmodel/rynn_dino_wm/chunkaware_m1024_d6_resume10k_bs80_20k_metrics_gpu4567_20260525_135027.log`.
- New output directory:
  `/home/user01/liops/workspace/DreamerVLA/data/outputs/worldmodel/rynn_dino_wm_action_hidden/chunkaware_m1024_d6_resume10k_bs80_20k_metrics/20260525_135027`.
