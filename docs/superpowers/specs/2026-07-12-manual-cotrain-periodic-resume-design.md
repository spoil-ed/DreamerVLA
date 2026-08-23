# Manual Cotrain Periodic Resume Design

## Problem

Periodic VLA evaluation splits one manual-cotrain job into subprocess segments. The first WM/CLS segment succeeds, but the second segment fails during Hydra composition because the launcher emits `manual_cotrain.resume_ckpt=...` while `dreamervla_wmcls_cotrain_ray_eval` does not declare that optional key under `manual_cotrain`.

The same segmenter also serves the frozen-model recipe, which does declare `manual_cotrain.resume_ckpt`. A schema-only fix in either recipe would therefore leave the launcher dependent on recipe-specific structure.

## Chosen Design

Treat the resume checkpoint as a launcher-injected optional Hydra value and emit it with `++manual_cotrain.resume_ckpt`. Hydra force-add semantics work whether the composed recipe already declares the key or not.

Update `_replace_overrides()` to compare normalized Hydra keys, stripping append/delete prefixes before replacement while preserving the requested prefix in the emitted override. This prevents an existing `manual_cotrain.resume_ckpt`, `+manual_cotrain.resume_ckpt`, or `++manual_cotrain.resume_ckpt` argument from surviving beside the newly emitted value.

## Scope

- Modify `dreamervla/launchers/manual_cotrain_vla_eval.py` only.
- Add a real composition regression test to `tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py`.
- Verify both the WM/CLS eval recipe and frozen-model eval recipe can compose a resumed segment.
- Do not alter model state, checkpoint payloads, training parameters, or YAML schemas.

## Error Handling

Hydra remains responsible for validating all other override keys. The launcher only makes the optional resume key schema-independent; invalid checkpoint paths remain runner-level errors.

## Verification

The regression test builds the production launch command, creates its second-segment command with `segment_train_command()`, and calls the production Hydra composition helper. On the current code it fails with the reported `ConfigCompositionException`; after the fix it must resolve the expected resume path for both recipe families.
