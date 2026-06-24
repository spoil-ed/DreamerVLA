"""Phase 3 characterization gate: extracting the Imaginer seam must not change the
WMPO outcome-step numerics. Golden values were captured from the pre-refactor
``dino_wmpo_outcome_step`` via the deterministic harness in
``test_wmpo_microbatch_equivalence`` (same mocks).
"""

import torch
from test_wmpo_microbatch_equivalence import _run_update

from dreamervla.algorithms.imagine import ImaginedRollout, Imaginer

# Captured from the current (pre-Imaginer-extraction) outcome step.
_GOLDEN_GRAD_1EPOCH = 0.19900251924991608
_GOLDEN_PARAM_3EPOCH = 0.4701790511608124


def test_outcome_step_gradient_matches_golden():
    grad, _ = _run_update(micro_batch_starts=0)
    assert torch.allclose(
        grad, torch.tensor(_GOLDEN_GRAD_1EPOCH), atol=1e-7
    ), grad.item()


def test_outcome_step_multiepoch_param_matches_golden():
    _, param = _run_update(micro_batch_starts=0, update_epochs=3, lr=0.05)
    assert torch.allclose(
        param, torch.tensor(_GOLDEN_PARAM_3EPOCH), atol=1e-7
    ), param.item()


def test_wmpo_imaginer_satisfies_protocol():
    from dreamervla.algorithms.ppo.outcome import WMPOImaginer

    assert isinstance(WMPOImaginer(), Imaginer)


def test_imagined_rollout_fields():
    fields = ImaginedRollout.__dataclass_fields__
    assert set(fields) == {
        "actor_feats",
        "actions",
        "action_token_ids",
        "old_log_probs",
        "ref_kls",
        "complete",
        "finish_step",
    }
