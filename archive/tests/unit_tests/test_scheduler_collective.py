from __future__ import annotations

import torch


def test_async_result_completed_waits_immediately() -> None:
    from dreamervla.scheduler.collective import AsyncResult

    work = AsyncResult.completed({"ok": True})

    assert work.done()
    assert work.wait() == {"ok": True}


def test_torch_collective_group_noops_when_dist_uninitialized() -> None:
    from dreamervla.scheduler.collective import TorchCollectiveGroup

    group = TorchCollectiveGroup()
    state = {"weight": torch.ones(2)}
    out = group.broadcast_state_dict(state, src=0)

    assert out is not state
    assert torch.allclose(out["weight"], state["weight"])
