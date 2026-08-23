"""Shared checkpoint-loading helpers for actor variants.

These helpers factor out the duplicated candidate-discovery / prefix-strip /
dtype-cast / expected-key filtering loop that several actor checkpoint loaders
share.  Each helper preserves the exact behaviour of the original per-actor
code so callers remain behaviour-preserving.
"""

from __future__ import annotations

import torch


def strip_prefixes(key: str, prefixes: tuple[str, ...]) -> str:
    """Repeatedly strip any of ``prefixes`` from the front of ``key``."""
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def normalize_candidate(
    candidate: dict,
    prefixes: tuple[str, ...],
    expected_keys: set[str],
    *,
    require_all_valid: bool = False,
) -> dict[str, torch.Tensor]:
    """Normalize one candidate dict to tensors matching ``expected_keys``.

    Strips ``prefixes`` from each key, casts matching tensors to float32, and
    keeps only keys that land in ``expected_keys``.

    ``require_all_valid`` reproduces the original VLA whole-candidate guard:
    when True, a candidate is skipped entirely (returns ``{}``) unless *every*
    entry is a ``(str, tensor)`` pair.  When False (latent / discrete-token
    behaviour) non-conforming entries are skipped individually.
    """
    if require_all_valid:
        if not candidate or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in candidate.items()
        ):
            return {}
    normalized: dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        stripped = strip_prefixes(key, prefixes)
        if stripped in expected_keys:
            normalized[stripped] = value.to(dtype=torch.float32)
    return normalized


def extract_state_dict(
    candidates: list[dict],
    prefixes: tuple[str, ...],
    expected_keys: set[str],
    *,
    require_all_valid: bool = False,
) -> dict[str, torch.Tensor]:
    """Return the first candidate that normalizes to a non-empty state dict."""
    for candidate in candidates:
        normalized = normalize_candidate(
            candidate, prefixes, expected_keys, require_all_valid=require_all_valid
        )
        if normalized:
            return normalized
    return {}
