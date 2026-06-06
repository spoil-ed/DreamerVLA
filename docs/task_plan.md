# Task Plan: LIBERO Goal One-Trajectory Pipeline on GPU45

## Goal

Run the LIBERO-goal single-trajectory VLA-SFT pipeline on GPUs 4 and 5, with the requested prerequisite WM and classifier training stages first.

## Phases

- [completed] Identify the correct configs and wrappers for WM, classifier, and one-trajectory VLA-SFT.
- [completed] Start an ordered GPU45 pipeline: WM -> classifier -> LIBERO-goal one-trajectory VLA-SFT.
- [completed] Check early WM logs for resume/config/runtime failures and GPU placement.
- [completed] Record session, log paths, and output directories.
- [completed] Re-check after the pipeline advances to classifier/VLA stages.
- [completed] Monitor early VLA logs for startup/runtime failures.
- [completed] Let the VLA training stage continue to completion in tmux.
- [completed] Multi-process test-eval the completed VLA checkpoint on LIBERO-goal only.
- [completed] Aggregate per-task LIBERO-goal eval metrics after all retry workers finish.
- [completed] Monitor the fresh no-resume VLA-SFT stage to completion.
- [completed] Multi-process test-eval the fresh no-resume VLA checkpoint on LIBERO-goal only.

## Notes

- Do not touch unrelated dirty worktree changes.
- Prefer an existing training script/config over inventing a new command.
- User clarified this is the DINO-WM-based action-chunk route, not the Chameleon WM route.
- Current request: "单轨迹VLA-SFT在libero goal上面的任务，需要先训练WM和classifier" on "45卡".
- Interpret "45卡" as `CUDA_VISIBLE_DEVICES=4,5`; GPUs 4 and 5 were free at session start.
- Run from `$DVLA_ROOT`, the source-complete checkout with current DINO-WM and classifier code.
- Resume-based pipeline session: `goal_one_traj_gpu45_20260527_031359`.
- Pipeline manifest: `data/outputs/logs/pipeline_goal_one_traj_gpu45_20260527_031359/manifest.txt`.
- WM stage reached 20000 steps.
- Classifier stage completed at 8000 steps; summary best episode F1 `0.9704545454545455`.
- VLA stage started with `vla_sft_one_trajectory` on GPUs 4,5 for `libero_goal`, one trajectory per task with offset 0.
- Early VLA verification: epoch 0 completed, validation metrics and checkpoint were written, and epoch 1 is running.
- Final VLA verification: completed 20 epochs, final logged `global_step=79`, `train_vla_loss=3.2249`, `val_val_ind_loss=1.2873`, `val_val_ood_loss=1.4886`; tmux session exited and GPUs 4/5 are idle.
- Correction: user requires fresh training for all stages; the prior resume-based run is not the requested final run.
- Original complete-experiment checkpoint under `data/outputs/worldmodel/dinowm_chunk/20260525_221114/ckpt/` is still present and was not overwritten.
- Fresh no-resume pipeline session: `goal_one_traj_fresh_gpu45_20260527_050937`.
- Fresh WM log confirms `training.resume=false`, `training.resume_path=null`, and progress from `step=0/20000`.
- Completed VLA checkpoint under eval:
  `data/outputs/vla/rynnvla_action_head_one_trajectory/libero_goal_one_traj_o0_gpu45_after_wmclf_20260527_031359/ckpt/latest.ckpt`.
- Eval base directory:
  `data/outputs/eval/eval_libero_vla/completed_vla_goal_8proc_20260527_052728`.
- Retry queue session:
  `eval_vla_goal_retry_queue_20260527_052728`.
- Eval aggregate for the completed VLA checkpoint: `0/100` successes (`0.0%`) over all 10 LIBERO-goal tasks.
- Fresh classifier completed with best episode F1 `0.9704545454545455`.
- Fresh VLA verification: completed 20 epochs, final logged `global_step=79`, `train_vla_loss=3.2240`, `val_val_ind_loss=1.2874`, `val_val_ood_loss=1.4889`; tmux session `goal_one_traj_fresh_gpu45_20260527_050937` exited and GPUs 4/5 were released.
- Fresh VLA artifacts:
  `data/outputs/vla/rynnvla_action_head_one_trajectory/libero_goal_one_traj_o0_gpu45_fresh_after_wmclf_20260527_050937/ckpt/latest.ckpt`,
  `data/outputs/vla/rynnvla_action_head_one_trajectory/libero_goal_one_traj_o0_gpu45_fresh_after_wmclf_20260527_050937/checkpoints/epoch=019-train_vla_loss=3.224.ckpt`,
  `data/outputs/vla/rynnvla_action_head_one_trajectory/libero_goal_one_traj_o0_gpu45_fresh_after_wmclf_20260527_050937/vla_logs.json.txt`.
- Fresh VLA eval base directory:
  `data/outputs/eval/eval_libero_vla/fresh_vla_goal_8proc_20260527_071628`.
- Fresh VLA eval aggregate: `0/100` successes (`0.0%`) over all 10 LIBERO-goal tasks.
