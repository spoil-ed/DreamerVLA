# PERF-W2 (H8) — atomic checkpoint save + single serialization

## Problem (audit H8 / roadmap W2 / §1.2 / §3.2 / §I)
`BaseRunner.save_checkpoint` at `dreamervla/runners/base_runner.py:800-821`
builds the resume payload (all modules + optimizers deep-copied to CPU via
`_copy_to_cpu` at `:1054`, plus `global_step` / `classifier_threshold` and other
pickled `include_keys`), then writes it with a **blocking, non-atomic**
`torch.save(payload, path)`. Two issues:

1. **Non-atomic write.** `torch.save` writes the destination file in place. A
   crash (or `kill -9`, OOM, preemption) mid-write leaves a half-written /
   truncated `.ckpt` at the canonical path — i.e. a corrupt resume artifact that
   overwrote the last-good one.
2. **Double serialization.** Every checkpoint event issues two separate
   `save_checkpoint` calls — once for `latest` and once for the top-k path
   (`openvla_oft_runner.py:224,236`; `pretokenize_vla_runner.py:945,957`). Each
   call **rebuilds the entire payload** (re-running `_copy_to_cpu` over every
   module + optimizer) and **re-runs `torch.save`** on the SAME bytes. The
   identical payload is serialized to disk twice.

## Existing pattern to REUSE (do NOT invent)
`dreamervla/runners/_dreamer_runner_common.py:85-88` already does the
temp-then-rename atomic write for the Dreamer checkpoint:

```python
path.parent.mkdir(parents=True, exist_ok=True)
tmp_path = path.with_suffix(path.suffix + ".tmp")
torch.save(payload, tmp_path)
tmp_path.replace(path)
```

`Path.replace` is an atomic `os.replace` on the same filesystem: the destination
only ever flips to the fully-written file or stays the previous one — never a
partial. Apply the identical pattern in `BaseRunner`.

## Exact change (scope: `base_runner.py` only)
1. **Atomic write helper.** Add a module-level `_atomic_torch_save(payload, path)`
   in `base_runner.py` that mirrors the Dreamer pattern (`.tmp` sibling →
   `torch.save` → `Path.replace`), and route `save_checkpoint`'s write through
   it instead of the in-place `torch.save`.
2. **Single serialization for latest + top-k.** Give `save_checkpoint` an
   optional `extra_paths` argument. When set, the payload is built **once**,
   serialized **once** (atomically to the primary `path` via the helper above),
   and then the resulting file is materialized at each extra destination by
   **hardlink (fall back to file copy on `OSError`, e.g. cross-filesystem)** —
   never a second `torch.save` of the same bytes. The link/copy is itself
   temp-then-rename atomic. Sidecars are emitted per destination from the same
   in-memory `payload` (sidecars are cheap relative to the full torch payload and
   are derived from the payload, not by re-reading the file).

   This collapses the latest + top-k double write into one serialization: a
   caller passes `save_checkpoint(extra_paths=(topk_path,))` instead of the
   back-to-back `save_checkpoint()` + `save_checkpoint(path=topk_path)` pair at
   `openvla_oft_runner.py:224,236` / `pretokenize_vla_runner.py:945,957`.

   **Scope note:** per the task's strict scope, this change lands the
   single-serialize *capability* in `base_runner.py` (exercised by the test) and
   leaves the two callers untouched in this commit; rewiring them to pass
   `extra_paths` is a follow-up so each call site can be re-verified separately.

Content preserved EXACTLY: payload keys, `_copy_to_cpu` semantics, the
`is_main_process` / `requires_collective` gating, and the sidecar hooks are
unchanged — only HOW the bytes reach disk (atomic temp→rename, serialize once,
link the rest) changes. A load of any resulting ckpt yields identical state.

## TDD
RED→GREEN on a fake runner (reuse the `_HookRunner` + `_FakeDistributed` shape
from `test_base_runner_shared_helpers.py`) with `tmp_path`:

- **(a) atomic**: after `save_checkpoint`, no leftover `*.tmp` sits next to the
  final file, and the final file loads back to the exact input payload.
- **(b) single serialize**: monkeypatch `torch.save` to a counting wrapper; a
  latest + top-k save via `extra_paths=(topk,)` calls `torch.save` **once**
  (RED driver — pre-change two separate calls would count twice).
- **(c) round-trip**: both the primary and the top-k file load to byte-identical
  state dicts / pickles as the saved content.

## Verify
- `conda run -n dreamervla python -m pytest tests/unit_tests/test_atomic_checkpoint_save.py -q`
- `conda run -n dreamervla ruff check dreamervla/runners/base_runner.py <test>`
- Full suite stays green (ignore only `.claude` hygiene + vendored
  `openvla_oft`/`prismatic` import artifacts).
