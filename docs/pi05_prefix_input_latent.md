# π0.5 Prefix-Input Latent

`prefix_input_latent` is the native OpenPI π0.5 prefix immediately before the
PaliGemma transformer. It is neither the PaliGemma output hidden state nor a KV
cache. Its fixed per-frame shape is `[968,2048]`:

- slots `0:768`: three 256-token camera blocks produced by native
  `paligemma_with_expert.embed_image()`;
- slots `768:968`: 200 language embeddings produced by the original OpenPI
  tokenizer, padding policy, and language embedding layer.

The existing `pi05_prefix_output [768,2048]` route remains the post-transformer,
image-only representation used by `wm_pi05_collected_train`. The two sources have
different names and sidecar schema versions so they cannot be silently mixed.

## Text modes

`text_mode=exact` retains the native language embeddings and validity mask.
`text_mode=masked` preserves `[968,2048]` but zeros slots `768:968` and marks their
attention mask false. `wm_use_text` controls the WM input independently from
`policy_use_text`; exact language embeddings remain available as a static sidecar
when the WM is text-disabled. A text-enabled policy recomposes them with imagined
visual tokens before prefill, while a text-disabled policy uses the masked form.

## Dynamics and action path

`Pi05PrefixInputWorldModel` conditions visual dynamics on current visual tokens,
proprioception, and continuous actions. Only slots `0:768` contribute to visual
reconstruction loss. Language slots are copied as static condition metadata and
are never prediction targets; `predict_text_tokens` must be false.

For action inference, `policy_prefix_input()` combines predicted visual tokens
with exact or masked language slots. `Pi05Policy.sample_actions_from_prefix_input()`
then constructs prefix attention and position ids, runs native PaliGemma prefill
with caching enabled, and passes the layer KV cache to the original π0.5 action
expert and flow-matching denoising loop. No independent action head is introduced.

## Configuration and launch

The task-owned contract is `task.prefix_input_latent`; construction is selected
through `worldmodel=pi05-prefix-input-wm`. The complete collected-rollout warmup
recipe is:

```bash
torchrun --standalone --nproc-per-node=8 -m dreamervla.train \
  experiment=wm_pi05_prefix_input_train
```

Useful Hydra overrides include:

```text
task.prefix_input_latent.text_mode=masked
task.prefix_input_latent.wm_use_text=false
task.prefix_input_latent.policy_use_text=false
```

The selected world-model and online encoder settings resolve from this task-owned
contract, so each override stays aligned across extraction, training, and policy
prefill.
