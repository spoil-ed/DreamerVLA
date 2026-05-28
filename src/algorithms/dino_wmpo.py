"""Back-compat shim — the dense-reward PPO route now lives in
``src.algorithms.ppo``. Prefer the new name:

    from src.algorithms.ppo import dino_wmpo_dense_step

The legacy names ``dino_wmpo_ppo_step`` and ``dino_wmpo_frame_step`` are
retained here as aliases for back-compat. The distinguishing axis between
the two PPO routes is the **reward form** (dense per-step state-reward vs
sparse outcome reward from LatentSuccessClassifier), not step vs chunk and
not single-frame vs window WM input.
"""

from src.algorithms.ppo import dino_wmpo_dense_step

# Legacy aliases.
dino_wmpo_frame_step = dino_wmpo_dense_step
dino_wmpo_ppo_step = dino_wmpo_dense_step

__all__ = ["dino_wmpo_dense_step", "dino_wmpo_frame_step", "dino_wmpo_ppo_step"]
