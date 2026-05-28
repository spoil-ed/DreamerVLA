# Progress: Train Action-Chunk WM

## 2026-05-25

- Started investigation for the action-chunk WM training entry point.
- Created lightweight planning files for durable command/log tracking.
- Identified the likely action-chunk WM entry point as `configs/chameleon_latent_action_wm_libero_goal.yaml` via `ChameleonLatentActionWMWorkspace`.
- User clarified the target is the DINO-WM-based route; stopped the Chameleon smoke/debug path.
- Verified the DINO-WM chunk-aware route in `/mnt/data/spoil/workspace/DreamerVLA`:
  `world_model._target_=src.models.world_model.rynn_dino_wm_chunk.ChunkAwareRynnDinoWMWorldModel`
  with `+world_model.chunk_size=5`.
- Launched formal training in tmux session `dino_wm_chunk_20260525_133737` on GPUs `4,5,6,7`.
- Output directory:
  `/mnt/data/spoil/workspace/DreamerVLA/data/outputs/worldmodel/rynn_dino_wm_action_hidden/chunkaware_m1024_d6_resume10k_bs80_20k/20260525_133738`.
- Log file:
  `/mnt/data/spoil/workspace/DreamerVLA/data/outputs/logs/worldmodel/rynn_dino_wm/chunkaware_m1024_d6_resume10k_bs80_20k_gpu4567_20260525_133737.log`.
- Early log check: resumed from the existing m1024/d6 checkpoint at `global_step=10000`, started 4-rank DDP training toward `training.max_steps=20000`, and reached at least `step=10020` with all GPUs 4-7 active.
- Updated the DINO-WM tqdm metrics so the progress bar shows `wm`, `next`, `roll`, `rew`, and `acc` instead of the pixel-WM-only `dyn/mse/psnr/rec/rep` zeros.
- Restarted the run with patched metrics in tmux session `dino_wm_chunk_metrics_20260525_135027`.
- New log file:
  `/mnt/data/spoil/workspace/DreamerVLA/data/outputs/logs/worldmodel/rynn_dino_wm/chunkaware_m1024_d6_resume10k_bs80_20k_metrics_gpu4567_20260525_135027.log`.
- New output directory:
  `/mnt/data/spoil/workspace/DreamerVLA/data/outputs/worldmodel/rynn_dino_wm_action_hidden/chunkaware_m1024_d6_resume10k_bs80_20k_metrics/20260525_135027`.

## 2026-05-27

- New request: run LIBERO-goal one-trajectory VLA-SFT on GPUs 4,5, with WM and classifier training first.
- Confirmed `nvidia-smi`: GPUs 4 and 5 were idle; GPUs 6 and 7 already had memory allocated.
- Confirmed relevant entries:
  - WM: `CONFIG=world_model_dinowm_chunk bash scripts/train_wm.sh`
  - Classifier: `CONFIG=latent_classifier_libero_goal_chunk bash scripts/train_wm.sh`
  - VLA-SFT: `bash scripts/train_vla_one_traj_45.sh libero_goal`
- Existing WM continuation point chosen:
  `data/outputs/worldmodel/dinowm_chunk/20260525_221114/ckpt/latest.ckpt`.
- The classifier will be trained into a timestamped `wmpo_aligned_small_tf_chunk_gpu45_*` directory so the old `wmpo_aligned_small_tf_chunk_minsteps32` result remains untouched.
- First launch attempt `goal_one_traj_gpu45_20260527_031309` failed before training because the nested `tmux new-session "bash -lc ..."` command had unsafe shell quoting. No training process remained from that attempt.
- Relaunched via script:
  `data/outputs/logs/pipeline_goal_one_traj_gpu45_20260527_031359/run_pipeline.sh`.
- Active tmux session:
  `goal_one_traj_gpu45_20260527_031359`.
- Pipeline manifest:
  `data/outputs/logs/pipeline_goal_one_traj_gpu45_20260527_031359/manifest.txt`.
- Stage 1 WM command is running on GPUs 4,5:
  `CONFIG=world_model_dinowm_chunk NGPU=2 CUDA_VISIBLE_DEVICES=4,5 bash scripts/train_wm.sh task=libero_goal training.resume=true training.resume_path=data/outputs/worldmodel/dinowm_chunk/20260525_221114/ckpt/latest.ckpt training.max_steps=20000`.
- WM output:
  `data/outputs/worldmodel/dinowm_chunk/libero_goal_gpu45_resume15k_to20k_20260527_031359`.
- WM log:
  `data/outputs/logs/pipeline_goal_one_traj_gpu45_20260527_031359/01_wm.log`.
- Early WM check: resumed at `global_step=15000`, `epoch=39`; DDP detected 2 ranks; train target is 20000 steps; reached at least `global_step=15180` with `rollout_chunks=4.0`.
- WM stage completed and wrote:
  `data/outputs/worldmodel/dinowm_chunk/libero_goal_gpu45_resume15k_to20k_20260527_031359/ckpt/step_00020000.ckpt`.
- Classifier stage started automatically after WM.
- Early classifier check: setup loaded 433 success demos and 67 failure demos, `granularity=chunk`, `W=8`, `K=5`, `pool=last`; first eval at step 200 reached episode F1 `0.9317` at threshold `0.48`.
- Classifier continued through at least step 3400; best episode F1 observed so far is `0.9521` at threshold `0.52`.
- Later classifier check: reached at least step 5000; best episode F1 observed so far is `0.9617` at threshold `0.41`. VLA stage had not started yet at that check.
- Follow-up classifier check: config target is `training.max_steps=8000`; reached at least step 5600. Best episode F1 observed so far is `0.9705` at threshold `0.67`, with `latest.ckpt` refreshed at `03:44:54`.
- Classifier stage completed at step 8000 and wrote `summary.json` with `best_episode_f1=0.9704545454545455`, `best_window_f1=0.527363184079602`, and `total_steps=8000`.
- VLA stage started automatically after classifier completion. The log confirms `config=vla_sft_one_trajectory`, `ngpu=2`, `gpus=4,5`, task `libero_goal`, `dataset.trajectories_per_task=1`, `dataset.trajectory_offset=0`, and output directory `data/outputs/vla/pi0_query_one_trajectory/libero_goal_one_traj_o0_gpu45_after_wmclf_20260527_031359`.
- Early VLA check: epoch 0 completed with validation metrics written to `vla_logs.json.txt`, `ckpt/latest.ckpt` and `checkpoints/epoch=000-train_vla_loss=4.918.ckpt` were saved, and epoch 1 is running. GPUs 4 and 5 were both at 100% utilization during this check.
- VLA stage completed all 20 epochs (`epoch` 0 through 19, final `global_step=79`). Final logged metrics: `train_vla_loss=3.2249`, `val_val_ind_loss=1.2873`, `val_val_ood_loss=1.4886`.
- Final VLA artifacts include `ckpt/latest.ckpt`, `checkpoints/epoch=019-train_vla_loss=3.225.ckpt`, and `vla_logs.json.txt`.
- The tmux session `goal_one_traj_gpu45_20260527_031359` exited after completion, and GPUs 4/5 were idle in the final `nvidia-smi` check.
- User corrected that WM should not have resumed; all stages need to be trained fresh.
- Confirmed the original complete-experiment WM checkpoint remains untouched:
  `data/outputs/worldmodel/dinowm_chunk/20260525_221114/ckpt/latest.ckpt` and `step_00015000.ckpt`.
- Created fresh no-resume pipeline script:
  `data/outputs/logs/pipeline_goal_one_traj_fresh_gpu45_20260527_050937/run_pipeline.sh`.
- First fresh launch failed before training because `tmux` did not activate the `dreamervla` conda environment; `python` resolved to base and missed `hydra`. Preserved that log as `01_wm.failed_env.log`.
- Patched the fresh pipeline script to export `/home/user01/miniconda3/envs/dreamervla/bin` and `PYTHON=/home/user01/miniconda3/envs/dreamervla/bin/python`.
- Relaunched tmux session `goal_one_traj_fresh_gpu45_20260527_050937`.
- Fresh WM is now running on GPUs 4,5 with `training.resume=false`, `training.resume_path=null`, output:
  `data/outputs/worldmodel/dinowm_chunk/libero_goal_gpu45_fresh_20k_20260527_050937`.
- Fresh WM log confirms training started from `step=0/20000`.
- Started LIBERO-goal eval for the previously completed VLA checkpoint:
  `data/outputs/vla/pi0_query_one_trajectory/libero_goal_one_traj_o0_gpu45_after_wmclf_20260527_031359/ckpt/latest.ckpt`.
- Patched eval loading to normalize DDP-saved VLA encoder keys (`backbone.module.*` -> `backbone.*`) for single-process eval, verified with a 1-episode smoke run on task 0.
- Launched multi-process LIBERO-goal eval under:
  `data/outputs/eval/eval_libero_vla/completed_vla_goal_8proc_20260527_052728`.
- Initial 8-worker launch hit OOM on tasks 0,2,5,7 because GPUs 6 and 7 already had ~38GB preprocessing jobs; tasks 1,3/4,6,8/9 continued successfully at 2 eval workers per GPU.
- Added and launched retry queue:
  `data/outputs/logs/eval_completed_vla_goal_8proc_20260527_052728/run_retry_queue.sh`,
  which waits for task1/task6 slots to free before retrying tasks 0,2,5,7.
- Fresh WM finished at `2026-05-27T06:06:00-04:00` and wrote:
  `data/outputs/worldmodel/dinowm_chunk/libero_goal_gpu45_fresh_20k_20260527_050937/ckpt/latest.ckpt`.
- Fresh classifier started immediately after WM, output:
  `data/outputs/dreamervla/outcome_classifier/libero_goal/wmpo_aligned_small_tf_chunk_gpu45_fresh_20260527_050937`.
- Eval task1 and task6 completed first, both `0/10`; retry workers for task0 and task5 started automatically at `06:09:56 EDT`.
- LIBERO-goal eval completed for the previously completed VLA checkpoint. Aggregated result:
  `0/100` successes, success rate `0.0%`; all 10 tasks have task-level results and none are missing.
- Fresh classifier completed at `training.max_steps=8000` with `best_episode_f1=0.9704545454545455` and `best_window_f1=0.527363184079602`.
- Fresh VLA-SFT started after the fresh classifier, output:
  `data/outputs/vla/pi0_query_one_trajectory/libero_goal_one_traj_o0_gpu45_fresh_after_wmclf_20260527_050937`.
- Fresh VLA-SFT completed all 20 epochs (`epoch` 0 through 19, final `global_step=79`). Final logged metrics:
  `train_vla_loss=3.2239904403686523`, `val_val_ind_loss=1.2873804569244385`, `val_val_ood_loss=1.488869309425354`.
- Fresh VLA artifacts include:
  `data/outputs/vla/pi0_query_one_trajectory/libero_goal_one_traj_o0_gpu45_fresh_after_wmclf_20260527_050937/ckpt/latest.ckpt`,
  `data/outputs/vla/pi0_query_one_trajectory/libero_goal_one_traj_o0_gpu45_fresh_after_wmclf_20260527_050937/checkpoints/epoch=019-train_vla_loss=3.224.ckpt`,
  and `data/outputs/vla/pi0_query_one_trajectory/libero_goal_one_traj_o0_gpu45_fresh_after_wmclf_20260527_050937/vla_logs.json.txt`.
- Final verification: tmux session `goal_one_traj_fresh_gpu45_20260527_050937` exited after checkpoint save, and GPUs 4/5 were released.
- Launched an 8-worker LIBERO-goal eval for the fresh VLA checkpoint on GPUs 4/5:
  `data/outputs/logs/eval_fresh_vla_goal_8proc_20260527_071628/run_eval.sh`.
- Fresh VLA LIBERO-goal eval completed for all 10 tasks with 10 episodes/task. Aggregated result:
  `0/100` successes, success rate `0.0%`; all task IDs 0-9 are covered and none are missing.
- Fresh VLA eval output:
  `data/outputs/eval/eval_libero_vla/fresh_vla_goal_8proc_20260527_071628`.
- Planned Stage 2 classifier output/log:
  `data/outputs/dreamervla/outcome_classifier/libero_goal/wmpo_aligned_small_tf_chunk_gpu45_20260527_031359`,
  `data/outputs/logs/pipeline_goal_one_traj_gpu45_20260527_031359/02_classifier.log`.
- Planned Stage 3 VLA output/log:
  `data/outputs/vla/pi0_query_one_trajectory/libero_goal_one_traj_o0_gpu45_after_wmclf_20260527_031359`,
  `data/outputs/logs/pipeline_goal_one_traj_gpu45_20260527_031359/03_vla_one_traj.log`.
