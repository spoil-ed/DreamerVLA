"""Adapt env records and hidden vectors to RolloutDumpWriter step dicts."""

from __future__ import annotations

from typing import Any

import numpy as np

from dreamervla.runtime.vectorized_collect import build_step_record, proprio_from_record


def build_dump_step(
    *,
    full_record: dict[str, Any],
    obs_embedding: Any,
    lang_emb: Any | None = None,
    action: Any,
    reward: float,
    sparse_reward: int,
    done: bool,
) -> dict[str, Any]:
    """Build one ``RolloutDumpWriter.write_demo`` step from Ray env output."""
    step = build_step_record(
        full_record,
        np.asarray(obs_embedding, dtype=np.float16),
        action,
        lang_emb=lang_emb,
    )
    step["actions"] = np.asarray(action, dtype=np.float64).reshape(-1)[:7]
    step["rewards"] = np.float32(reward)
    step["sparse_rewards"] = np.uint8(int(sparse_reward))
    step["dones"] = np.uint8(1 if done else 0)
    step["obs_embedding"] = np.asarray(obs_embedding, dtype=np.float16)
    step["proprio"] = proprio_from_record(full_record)
    step["success"] = bool(done and sparse_reward)
    step["wm_action"] = np.asarray(action, dtype=np.float32).reshape(-1)
    step["reward"] = np.float32(reward)
    step["done"] = np.float32(1.0 if done else 0.0)
    step["is_terminal"] = np.float32(1.0 if sparse_reward else 0.0)
    step["is_last"] = np.float32(1.0 if done else 0.0)
    return step
