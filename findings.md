# Findings — RLinf reference contract & DreamerVLA divergences

> Data only. Reference = RLinf (verified ~0.50 success_once on libero-goal traj1).
> Target = DreamerVLA. All paths absolute on this host.

## Environment / infra facts
- `dreamervla` conda env HAS gym+libero+robosuite → DreamerVLA rollout runs natively on host.
- RLinf eval runs in docker; container **already running**: name `rlinf`,
  image `rlinf/rlinf:agentic-rlinf0.2-maniskill_libero`. Host
  `/mnt/data/spoil/workspace/DreamerVLA` → docker `/workspace/RLinf/DreamerVLA`.
  Host `/mnt/data/spoil/workspace/RLinf` → docker `/workspace/RLinf/RLinf`.
- `MUJOCO_GL=osmesa` (EGL crashes in robosuite read_pixels on this host).
- Checkpoints exist: libero10 / libero-goal / libero-object / libero-spatial traj1.
- RLinf launcher default MODEL_PATH points at stale `data/ckpts/...`; REAL path is
  `data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1`. Override it.
- RLinf eval entry: `examples/embodiment/eval_embodied_agent.py`,
  config `wan_libero_goal_grpo_openvlaoft_4567`, `unnorm_key=libero_goal_no_noops`.

## RLinf reference I/O contract (file:line in /mnt/data/spoil/workspace/RLinf)
- Env: `rlinf/envs/libero/libero_env.py` LiberoEnv; 256x256; cameras agentview + eye_in_hand.
- Image to policy: **num_images_in_input=1** (`config/model/openvla_oft.yaml:30`) → 1 agentview frame.
  - 180° rotate at env level: `img[::-1, ::-1]` (`rlinf/envs/libero/utils.py:73-103`).
  - Preprocess (`models/.../prismatic/processing_prismatic.py:37-68`): TVF.resize→224,
    TVF.center_crop (center_crop:True), /255, ImageNet normalize. RGB. NO temporal stack.
- Proprio: `libero_env.py:548-559` builds eef_pos(3)+axisangle(3)+gripper_qpos → states.
  Passed as proprio_states. **VERIFY** whether discrete traj1 model uses it (openvla_oft.yaml use_proprio).
- Prompt: `f"In: What action should the robot take to {t.lower()}?\nOut: "`
  (`models/.../openvla_oft/rlinf/openvla_oft_action_model.py:219-222`) — TRAILING SPACE after "Out:".
- Action decode: discrete argmax → vocab_size - id → clip → bin_centers → `_unnormalize_actions`
  (model.py:382-404, 167-203). chunk = 8 (`num_action_chunks: 8`). action_dim 7.
- Gripper post-process: `rlinf/envs/action_utils.py:66-77` prepare_actions_for_libero:
  `chunk[...,-1] = 2*g - 1` then `= sign(...)*-1.0`. (standard OpenVLA-LIBERO).
- Action application: `rlinf/workers/env/env_worker.py:403-415` executes **all 8** chunk actions
  sequentially (open-loop), then re-queries. NO temporal ensemble.
- Initial settle: `libero_env.py:633-638` **15 no-op steps**, gripper held -1 (reset_gripper_open).
- Model load: from_pretrained local dir, bf16; dataset_statistics.json from model_path.
- Success: success_once = ever-terminated within ep; success_at_end = terminated at final step.
- Constants: eval total_num_envs (launcher 64; cfg default 496), max_episode_steps eval 512,
  eval_rollout_epoch (cfg 1; launcher default 8).

## DreamerVLA current pipeline (file:line in /mnt/data/spoil/workspace/DreamerVLA)
- Env: `dreamervla/envs/libero_env.py` LIBERODreamerEnv; 256x256; returns third_image+wrist_image,
  both 180°-rotated; proprio 8-dim (3+3+2 gripper_qpos).
- Extractor: `dreamervla/runners/rollout_hidden_extractor.py` prepare()/step(); history deque;
  camera select `oft_collect_common.py select_vla_image_keys` n_views=num_images/history.
- Preprocess: TF lanczos3 resize + TF crop_and_resize (openvla_utils.prepare_images_for_vla).
- Prompt: `f"In: What action should the robot take to {task_description.lower()}?\nOut:"` (NO trailing space).
- Action decode: same discrete formula (rollout_hidden_extractor.py:521-546); chunk 8.
- Action apply: `collect_parallel_rollouts.py:180,217` executes only **chunk[0]** (receding horizon).
- Gripper: NOT explicitly transformed (no 2g-1 / sign flip found) — SUSPECT.
- Settle steps: none found — SUSPECT.
- Ray path: ColdStartRayCollectRunner + RolloutInferenceWorker.forward_batch.
- Non-ray path: CollectRolloutsRunner / collect_parallel_rollouts (torchrun shard + vectorized).
- Tests: most are stub/fake; the only real-GPU test compares extractor hidden vs offline sidecar
  (numeric parity), NOT success rate. → success rate never verified anywhere.

## ★★★ FINAL ROOT CAUSE (2026-06-18): transformers must be the openvla-oft FORK, not vanilla
THE bug: openvla-oft requires the CUSTOM transformers fork `moojink/transformers-openvla-oft` (v4.40.1),
which patches the Llama forward for OFT parallel/bidirectional action attention. The host conda env had
**VANILLA transformers 4.40.1**; docker had the FORK. BOTH report `__version__ == "4.40.1"`, so the
hard version check (modeling_prismatic.py:331) passed and the difference stayed invisible.
- Proof: transformers/models/llama/modeling_llama.py — vanilla 1566 lines (md5 f8588efe) vs fork 1620 lines
  (md5 752d33b6). Swapping the fork into dvla_oft made golden action[0] flip from garbage
  [0.0003,0.2876,...] to coherent [0.601,0.0113,...] == docker exactly.
- Mechanism: vanilla Llama gives a tiny layer-0 numerical difference, amplified by Llama massive activations
  (L01 outlier ~3486) into structurally wrong action tokens => 0% success. Fork's modified attention is correct.
- FIX: install the fork transformers into the OFT env (copied docker's into `dvla_oft`).
- Everything else (torch 2.5/2.6, timm, cuBLAS, weights, code, inputs, GPU) was IDENTICAL & ruled out — the
  exhaustive isolation below was correct that it's "the Llama forward", just the wrong sub-cause until the fork.

## (superseded mid-investigation) torch 2.5.1 vs 2.6.0 hypothesis — WRONG (dvla_oft torch 2.6.0 still garbage w/ vanilla)
Layer-by-layer isolation (identical saved input_ids + pixel_values fed both envs):
- vision_backbone(pv): host==docker BYTE-IDENTICAL (max abs diff 0.0).
- inputs_embeds into Llama (1,336,4096) + attention_mask: host==docker BYTE-IDENTICAL (0 positions differ).
- Llama OUTPUT (action_hidden_states 1,56,4096): DIFFER max abs diff 11.25.
- Per-layer Llama hidden divergence: L00 mean 0.005 → grows monotonically → L31 mean 1.57; plus a
  massive-activation dim differing ~3486 from L01 onward.
- transformers ruled out: host output BYTE-IDENTICAL across 4.43.0 and 4.40.1.
- timm ruled out: host output BYTE-IDENTICAL across 0.9.16 and 0.9.10.
- attention impl ruled out: both default to sdpa; forcing eager on host does NOT fix (still 11.19 vs docker).
- PRECISION ruled out: host fp32 predict_action == host bf16 EXACTLY ([0.0003,0.2876,...]) != docker [0.601,...].
  (If precision, fp32 would change host output; it doesn't => deterministic structural torch difference.)
=> Only remaining env diff: torch 2.5.1 (host) vs 2.6.0 (docker). torch 2.5.1 computes the Llama forward
   structurally differently (precision-independent) for identical inputs/weights/code. FIX = match torch 2.6.0.
   Env-strategy decision pending (upgrade host torch — risks flash_attn recompile etc. / dedicated env / docker).
Probe scripts: data/_io_probe.py, _llm_input_probe.py, _layer_probe.py, _attn_probe.py (clean up later).

## (superseded) earlier hypothesis: transformers version
DreamerVLA rollout = 0% in host `dreamervla` env; RLinf docker = 50%. SAME ckpt/image/env-render.
- HOST `dreamervla` env: **transformers 4.43.0** (tokenizers 0.19.1).
- DOCKER openvla-oft venv: **transformers 4.40.1** (the version openvla-oft REQUIRES; modeling_prismatic.py warns on mismatch).
- Golden test (same image data/_dbg_model_third.png + same third_party/openvla-oft code + same ckpt,
  ONLY transformers differs): 4.40.1 -> smooth coherent action chunk; 4.43.0 -> erratic garbage
  (max abs diff 1.59, gripper dim nearly inverted). => discrete predict_action/generate is broken on 4.43.
- Image render host==docker pixel-identical (ruled out). All 28 canonical eval mp4s = success=False (systematic).
- FIX = run OFT rollout/collector with transformers==4.40.1. Decision pending: downgrade shared dreamervla env
  vs dedicated env vs docker. (tokenizers 0.19.1 already correct.)

## Divergence ranking (most → least likely to cause 0%) — SUPERSEDED by root cause above
NOTE: divergences 1-6 below were the pre-root-cause hypotheses; the ACTUAL 0% cause is the transformers
version. The alignment items (gripper/chunk/single-frame/settle) are still required for a faithful port and
are implemented in dreamervla/runners/rlinf_libero_rollout.py, but they are NOT what caused 0%.
1. **Gripper post-process missing** (2g-1 + sign flip). Grasp fails ⇒ ~0%.
2. **Execute all 8 vs only chunk[0]**. Trained for chunked open-loop ⇒ big drop.
3. **Image frame count** 1 vs stacked-2 (depends on active history config).
4. **15-step initial settle** missing.
5. **proprio parity** for discrete ckpt.
6. **prompt trailing space**.
7. image preprocess TF vs torchvision (numerically close, low risk for success).

## RESOLVED config (from live eval resolved-config dump, console log lines 240-269)
CONFIRMED for libero-goal traj1 discrete eval (`wan_libero_goal_grpo_openvlaoft_4567`):
- `use_proprio: false`  → discrete traj1 does NOT feed proprio. ✅ (matches DreamerVLA discrete)
- `num_images_in_input: 1` → single agentview frame (no wrist, no temporal stack). ✅
- `use_film: false`, `center_crop: true`, `image_size: [224,224]`, `vocab_size: 32000`, `hidden_size: 4096`.
- `action_dim: 7`, `num_action_chunks: 8`, `unnorm_key: libero_goal_no_noops`, `precision: bf16`,
  `max_prompt_length: 128`, `policy_setup: widowx_bridge`, `attn: flash_attention_2`.
- env.eval: `max_episode_steps: 512`, `max_steps_per_rollout_epoch: 512`, `eval_rollout_epoch: 1`,
  `total_num_envs: 16` (smoke override), `auto_reset: true`, `ignore_terminations: true`,
  `reset_gripper_open: true`, `is_eval: true`, `seed: 0`, `group_size: 1`,
  `init_params.camera_heights/widths: 256`.

## Phase 1 RESOLVED (all open questions closed)
- **proprio = FALSE**: use_proprio False; discrete `_build_embedding` only does vision features,
  proprio_states passed to processor but ignored. (model.py _build_embedding ~80-127)
- **gripper**: value before prepare_actions_for_libero is in [0,1]; transform (action_utils.py:75-76):
  `g = 2g - 1` ([0,1]→[-1,1]) then `g = sign(g) * -1.0` (BINARIZE @ 0.5 + INVERT → ±1.0).
  This == standard OpenVLA `normalize_gripper_action(binarize=True)` + `invert_gripper_action`.
  PORT MUST replicate exactly.
- **history = 1**: num_images_in_input=1, agentview only, no wrist, no temporal stack.
- **action chunk = ALL 8 open-loop**: `libero_env.py:681-735 chunk_step` `for i in range(chunk_size=8)`
  executes all 8 actions before re-query. (DreamerVLA executes only chunk[0] → divergence #2.)
- **num_steps_wait = 15** initial settle steps, gripper held -1 (libero_env.py:633).
- max_prompt_length 128 (config override), max_episode_steps eval 512.

→ Phase 1 contract is FROZEN. New code (§4 strategy) implements exactly the above.
