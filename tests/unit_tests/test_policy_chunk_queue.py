from __future__ import annotations

import torch

from dreamer_vla.utils.policy_chunk_queue import PolicyChunkActionQueue


def test_policy_chunk_queue_reuses_chunk_actions_before_resampling() -> None:
    calls: list[torch.Tensor] = []

    def policy(batch: dict[str, object]):
        calls.append(batch["hidden"])
        base = 10 * len(calls)
        chunk = torch.tensor([[[base + 0, 0], [base + 1, 1], [base + 2, 2]]], dtype=torch.float32)
        return chunk, None, {}

    queue = PolicyChunkActionQueue(collect_chunk_steps=2)
    first = queue.next_action(policy, hidden=torch.tensor([[1.0]]), deterministic=True)
    second = queue.next_action(policy, hidden=torch.tensor([[2.0]]), deterministic=True)
    third = queue.next_action(policy, hidden=torch.tensor([[3.0]]), deterministic=True)

    assert len(calls) == 2
    assert first.tolist() == [10.0, 0.0]
    assert second.tolist() == [11.0, 1.0]
    assert third.tolist() == [20.0, 0.0]
