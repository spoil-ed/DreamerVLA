# RLinf / DreamerVLA OpenVLA-OFT eval alignment findings

Date: 2026-06-18

## Scope

Compare the RLinf LIBERO OpenVLA-OFT eval path against DreamerVLA's non-Ray
collector path for the one-trajectory SFT checkpoint:

- Checkpoint:
  `data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1`
- Suite / task:
  `libero_goal`
- Dataset statistics key:
  `libero_goal_no_noops`

## What has been verified

- The earlier failed run was the RLinf Ray eval path, not DreamerVLA's non-Ray
  collector.
- RLinf call chain:
  `examples/embodiment/eval_embodied_agent.py` ->
  `EmbodiedEvalRunner` ->
  `MultiStepRolloutWorker` ->
  `EnvWorker` ->
  `rlinf/envs/libero/libero_env.py`.
- DreamerVLA should keep the non-Ray Runner / torchrun design. Do not port
  RLinf's Ray worker topology into DreamerVLA.
- Missing package found in the `dreamervla` conda env:
  `gymnasium`. It was installed with `uv pip install --python
  /home/user01/miniconda3/envs/dreamervla/bin/python gymnasium`.
- RLinf import then reached model/env setup, but the small Ray eval attempt later
  aborted inside Ray worker / Gloo collective teardown. This does not prove a
  model or checkpoint issue.

## Current alignment points

- Both RLinf and DreamerVLA rotate LIBERO RGB images by 180 degrees before VLA
  preprocessing.
- For the one-trajectory discrete checkpoint, both intended eval paths use one
  input image:
  - RLinf config: `num_images_in_input: 1`
  - DreamerVLA `experiment=collect_rollouts_onetraj`: `collect.num_images_in_input: 1`
- DreamerVLA collector now applies the OpenVLA gripper convention before
  stepping LIBERO:
  dataset / OpenVLA `0=close, 1=open` -> LIBERO env `+1=close, -1=open`.
- DreamerVLA collector now executes the open-loop action chunk queue instead of
  repeatedly using only the first action in each predicted chunk.

## Confirmed root cause

The low DreamerVLA non-Ray rollout success rate was caused by the collector
stepping LIBERO with raw OpenVLA-OFT gripper values.

OpenVLA-OFT emits the gripper channel in the dataset convention:

- `0` means close
- `1` means open

LIBERO execution expects:

- `+1` means close
- `-1` means open

RLinf applies this conversion in `prepare_actions_for_libero` before
`env.step()`:

```python
chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
chunk_actions[..., -1] = np.sign(chunk_actions[..., -1]) * -1.0
```

DreamerVLA's collector previously skipped the conversion in both
`collect_parallel_rollouts._run_episode` and
`vectorized_collect.collect_vectorized`. The failure mode is severe: an intended
open command (`1`) is executed as close (`+1`), and an intended close command
(`0`) becomes neutral-ish (`0`) rather than a firm close. This explains
"reaches the object but cannot grasp/release reliably" and can collapse LIBERO
success even when the pose dimensions are correct.

The local fix is `dreamervla.envs.libero_env.process_openvla_libero_action`,
used before stepping both the single-env and vectorized collector paths.

## Secondary alignment point

The same local diff also switches rollout execution from repeatedly taking only
`action_chunk[0]` to queuing and replaying the predicted action chunk for
`chunk_size` env steps. This matches the OpenVLA-OFT/RLinf chunked eval
protocol. It is an alignment improvement, but the gripper conversion is the
primary success-rate fix.

## If the next small eval still fails

Then run the layer-cos diagnostic. The likely remaining divergence points would
be:

1. Input image preprocessing:
   RLinf uses its `MultiInputPrismaticProcessor` and torch image processor on
   tensor images. DreamerVLA uses the OpenVLA-OFT utility preprocessing path in
   `rollout_hidden_extractor.py`.
2. Prompt/token padding:
   RLinf pads prompts to `max_prompt_length=128` and normalizes BOS placement.
   DreamerVLA builds shorter single-sample prompts and the decoder appends the
   trailing separator token when needed.
3. Action-token logits:
   RLinf masks logits to the action-token range before argmax/sample.
   DreamerVLA's discrete decode currently takes argmax over the raw action-span
   logits.
4. Action head / action decode:
   The one-trajectory checkpoint appears to be the discrete-token mode, so the
   first diagnostic should confirm whether divergence happens before any action
   head would be involved.

## Diagnostic plan

Add a non-Ray diagnostic CLI that:

- builds one shared LIBERO reset observation from the same task/reset state;
- loads the same checkpoint through RLinf and DreamerVLA code paths in the same
  Python environment;
- runs deterministic forward inference for both paths;
- records input ids, attention masks, processed pixels, action-token logits,
  decoded token ids/actions, and per-language-layer cosine similarity over the
  action-token hidden span.

The first useful failure boundary is:

- low pixel / token cosine: input pipeline mismatch;
- high early-layer cosine but low action-logit cosine: model forward or masking
  mismatch;
- high logits but different final actions: decode / unnormalization mismatch.
