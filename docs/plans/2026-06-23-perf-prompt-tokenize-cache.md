# PERF — cache the invariant prompt tokenization in the OFT rollout extractor

> Audit ref: `docs/plans/performance_optimization_audit.md` §3.1 (`rollout_hidden_extractor.py:230-267`)
> and §F (`__getitem__`/hot-path repeated re-work). Roadmap: Phase 1 "prompt-tokenize cache"
> (`docs/plans/2026-06-23-perf-audit-execution-roadmap.md`). Plan STYLE follows
> `docs/plans/2026-06-22-mem-rl-01-microbatch-wmpo.md` (problem → key facts → design → TDD →
> equivalence gate). All tests/ruff run in the **dreamervla** conda env.

## Problem
`OFTRolloutHiddenExtractor.prepare` (`rollout_hidden_extractor.py:230-267`) builds, **every rollout
step**, the prompt

```
In: What action should the robot take to {task_description.lower()}?\nOut:
```

then calls `processor(prompt, img)` **once per camera view**. `PrismaticProcessor.__call__`
(`models/embodiment/openvla_oft/processing_prismatic.py:68-95`; upstream
`prismatic/extern/hf/processing_prismatic.py:187-216`) does TWO independent things:

1. `pixel_values = self.image_processor(images, return_tensors=...)["pixel_values"]` — depends ONLY
   on the image, must run each step.
2. `text_inputs = self.tokenizer(text, ...)` (+ `_normalize_left_padded_bos` in the dreamervla
   subclass) — depends ONLY on `text` (the prompt), i.e. ONLY on `task_description`. Invariant within
   an episode.

So for a fixed `task_description` the tokenizer runs `len(views)` times per step (e.g. 2×) producing
the SAME `input_ids`/`attention_mask` every step — pure repeated work. The extra views additionally
discard the text part entirely (only `pixel_values` is kept and `cat`-ed), so their tokenization is
100% wasted today.

## Key facts that make caching numerically identical
- In BOTH processor variants the text branch and image branch are **independent**: they touch no
  shared state and only meet at a batch-size equality check. `_normalize_left_padded_bos` mutates the
  tokenizer output in place but is a pure function of `text`. Hence `processor(prompt, img)`'s
  `input_ids`/`attention_mask` are a pure function of `prompt`, and `prompt` is a pure function of
  `task_description`.
- `processor.image_processor(img, return_tensors="pt")["pixel_values"]` is **byte-identical** to
  `processor(prompt, img)["pixel_values"]` — it is the exact line `__call__` uses (default
  `return_tensors=TensorType.PYTORCH` == `"pt"`). So pulling `pixel_values` straight from
  `image_processor` does not change the image numerics at all.
- Therefore: caching `(input_ids, attention_mask)` per `task_description` and getting `pixel_values`
  per-view from `image_processor` reproduces today's `input_ids`/`attention_mask`/`pixel_values`
  exactly (atol=0) for any `(prompt, img)`.

## Design (minimal, surgical)
In `prepare`:
- Populate the prompt cache on a MISS by calling the FULL `processor(prompt, processed_images[0])`
  exactly as today (same call path → canonical text tensors), and store CPU clones of its
  `input_ids` / `attention_mask` keyed by `task_description`. Using the full call (not the bare
  `processor.tokenizer(...)`) preserves any processor-specific text post-processing — the bundled OFT
  checkpoints register the upstream `PrismaticProcessor` (no extra step), but the dreamervla subclass
  applies `_normalize_left_padded_bos` in place; the full call captures whichever is loaded. The miss
  therefore does one extra (discarded) image pass, once per task — acceptable.
- On a HIT, reuse the cached text tensors; do NOT re-call the processor.
- For EVERY view (primary + extra) obtain `pixel_values` from
  `processor.image_processor(img, return_tensors="pt")["pixel_values"]` — the same image branch the
  processor uses internally, so it is numerically identical and also drops the wasted extra-view
  tokenization.
- Move the cached text tensors + the per-view pixel_values to `device`/`bfloat16` exactly as today.
  `BatchFeature.to(device, dtype=bf16)` casts only floating tensors, so the old code already moved
  `input_ids`/`attention_mask` device-only (no dtype change) and bf16-cast `pixel_values`; the new
  `input_ids.to(device)` / `attention_mask.to(device)` + `cat(...).to(device, bf16)` reproduce that.
  `cat` is a pure copy, so `cat(fp32 views).to(bf16) == cat(bf16 views)` element-wise.
- The cache is a single-slot `{task_description: (input_ids, attention_mask)}` on the extractor
  instance; it auto-refreshes when `task_description` changes (different key). `reset()` is unchanged
  (the prompt is task-keyed, not episode-keyed; same task across episodes legitimately reuses it).

Helpers `_prompt_text_inputs(processor, prompt, task_description, fallback_image)` (the cache) and
`_view_pixel_values(processor, image)` (the image branch) are factored out so they are unit-testable
WITHOUT the vendored OFT tree / a real model (`prepare` itself needs `prepare_images_for_vla`).

No new config knob: the cache is always-on and behavior-equivalent (Phase-1 quick win).

## TDD (RED → GREEN), dreamervla env, no GPU/model needed
Add `tests/unit_tests/test_rollout_hidden_extractor_prompt_cache.py`:
1. A fake processor: `image_processor(img)` returns a deterministic tensor derived from the image
   (so per-image variation is visible); `tokenizer(text)` returns deterministic tensors derived from
   `text` and increments a call counter; the in-place text normalisation is a no-op (the dreamervla
   mutation is text-pure, irrelevant to the cache contract).
2. **Equivalence**: for two different images under the SAME `task_description`, the cached path returns
   `input_ids`/`attention_mask` equal (atol=0) to a direct `processor(prompt, img)` call, and
   `pixel_values` equal to `processor(prompt, img)["pixel_values"]` for each image.
3. **Cache hit**: calling the prepare/tokenize path N times with the SAME `task_description` invokes
   the tokenizer exactly ONCE (counter == 1). RED first (no cache → counter == N).
4. **Cache refresh**: a different `task_description` re-tokenizes (counter increments), and the new
   text tensors differ from the first task's.

Gate: `conda run -n dreamervla python -m pytest tests/unit_tests/test_rollout_hidden_extractor_prompt_cache.py
tests/unit_tests/test_rollout_hidden_extractor.py -q` green; `conda run -n dreamervla ruff check` clean
on the touched files. Behavior-equivalent → no GPU smoke required (the real-model consistency gate in
`test_rollout_hidden_extractor.py` already guards the numerics end-to-end and is unchanged).

## Out of scope
Batched / multi-task tokenization sharing across envs, and the OFT `prepare_inputs` per-sample
processor loop (§3.5 H) — separate items.
