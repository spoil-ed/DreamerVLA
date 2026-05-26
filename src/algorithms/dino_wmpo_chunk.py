"""Back-compat shim — the outcome-reward PPO route now lives in
``src.algorithms.ppo``. Prefer the new name:

    from src.algorithms.ppo import dino_wmpo_outcome_step

The legacy names ``dino_wmpo_chunk_step`` and ``dino_wmpo_window_step`` are
retained here as aliases for back-compat. The distinguishing axis between
the two PPO routes is the **reward form** — this route uses a sparse
outcome reward from ``LatentSuccessClassifier`` scoring the imagined latent
video — not chunk-vs-step and not single-frame-vs-window WM input.

``build_valid_chunk_count`` and ``_build_reward_tensor`` are also re-exported
here because existing tests import them by name from this module path.
"""
from src.algorithms.ppo import dino_wmpo_outcome_step
from src.algorithms.ppo.outcome import _build_reward_tensor, build_valid_chunk_count

# Legacy aliases.
dino_wmpo_window_step = dino_wmpo_outcome_step
dino_wmpo_chunk_step = dino_wmpo_outcome_step

__all__ = [
    "dino_wmpo_outcome_step",
    "dino_wmpo_window_step",
    "dino_wmpo_chunk_step",
    "build_valid_chunk_count",
    "_build_reward_tensor",
]
