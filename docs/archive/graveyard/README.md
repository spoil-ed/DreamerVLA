# Graveyard

This directory holds files that were once part of the active codebase but are
no longer referenced by any working entry point (mainline configs, formal
shell launchers, tests, or `_target_` resolutions).

Layout mirrors the original repository path so each file remains traceable to
where it used to live (e.g. `docs/archive/graveyard/scripts/eval/foo.py` was previously at
`scripts/eval/foo.py`).

Files are kept here rather than deleted so the implementation can be revived
if a future ablation needs it. Nothing under `docs/archive/graveyard/` is
imported by the live tree.

For the rationale and the orphan-criteria used during the 2026-05-27 sweep,
see [docs/graveyard_candidates_2026-05-27.md](../../graveyard_candidates_2026-05-27.md).
