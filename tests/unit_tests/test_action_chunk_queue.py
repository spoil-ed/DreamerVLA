from __future__ import annotations

import numpy as np
import pytest

from dreamervla.runtime.action_chunk_queue import ActionChunkQueue


def test_action_chunk_queue_executes_full_chunk_before_refill() -> None:
    queue = ActionChunkQueue(action_dim=7, action_steps=3)
    first_chunk = np.arange(28, dtype=np.float32).reshape(4, 7)

    queue.refill(first_chunk)

    np.testing.assert_array_equal(queue.pop(), first_chunk[0])
    np.testing.assert_array_equal(queue.pop(), first_chunk[1])
    np.testing.assert_array_equal(queue.pop(), first_chunk[2])
    assert queue.has_pending is False


def test_action_chunk_queue_clear_drops_pending_actions() -> None:
    queue = ActionChunkQueue(action_dim=7, action_steps=2)
    queue.refill(np.ones((2, 7), dtype=np.float32))

    queue.clear()

    assert queue.has_pending is False
    with pytest.raises(IndexError, match="empty action chunk queue"):
        queue.pop()


def test_action_chunk_queue_rejects_short_chunk() -> None:
    queue = ActionChunkQueue(action_dim=7, action_steps=4)

    with pytest.raises(ValueError, match="need action_steps=4"):
        queue.refill(np.zeros((3, 7), dtype=np.float32))
