"""Protocol + typed output for the WMPO imagination (the World Model layer seam)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class ImaginedRollout:
    """One group-aligned imagined slice.

    Per-chunk policy inputs are kept as host buffers (``actor_feats`` on CPU; the
    rest on device) so the multi-epoch PPO loss can stream them back one chunk at a
    time. ``complete`` / ``finish_step`` (env-step units) are the verifier's sparse
    threshold outcome. ``score`` / ``score_step`` are an optional continuous value
    source from the same verifier, also in env-step units.
    """

    actor_feats: list[torch.Tensor]
    actions: list[torch.Tensor]
    action_token_ids: list[torch.Tensor | None]
    old_log_probs: list[torch.Tensor]
    ref_kls: list[torch.Tensor] | None
    complete: torch.Tensor
    finish_step: torch.Tensor
    score: torch.Tensor | None = None
    score_step: torch.Tensor | None = None


@runtime_checkable
class Imaginer(Protocol):
    """Imagines + scores ONE group-aligned start slice in the world model.

    Default = ``WMPOImaginer`` (chunk-WM rollout scored by the success verifier).
    A different rollout/scoring strategy can be swapped in as long as it returns an
    ``ImaginedRollout`` with the same contract.
    """

    def imagine_slice(self, **kwargs: Any) -> ImaginedRollout:
        """Return the imagined + scored slice for the given group-aligned start."""
        ...
