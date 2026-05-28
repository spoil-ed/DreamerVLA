"""DreamerVLA PPO routes.

Three mature end-to-end PPO routes are exposed. The two primary routes
differ in the **reward form** seen by the actor:

  • ``dino_wmpo_dense_step``       (dense per-step state-reward)
      A scalar reward is decoded from the world-model hidden at every
      imagined env-step; the γ-discounted sum becomes the rollout return.
      Per-step actor + per-step WM (``predict_next``).
      Implementation: ``src.algorithms.ppo.dense``.

  • ``dino_wmpo_outcome_step``     (sparse outcome reward)
      A single ``LatentSuccessClassifier`` score over the imagined latent
      video produces ``(complete, finish_step)``. Per-chunk actor +
      per-chunk WM (``predict_next_chunk``), eos_mask truncation,
      KL-into-reward, zero-variance group filtering.
      Implementation: ``src.algorithms.ppo.outcome``.

  • ``dino_wmpo_dense_chunk_step`` (dense reward + chunk rollout)
      Same dense per-step state-reward as ``dense_step``, but driven by
      the chunk WM so each actor decision produces a full K-step action
      chunk (one PPO log_prob per chunk). Bridges dense and outcome:
      aligns actor decision granularity with its natural chunk output
      without changing the reward form. TD-MPC critic and
      ``real_rollout_relabel`` side-losses not yet wired.
      Implementation: ``src.algorithms.ppo.dense_chunk``.

Submodules (``grpo``, ``relabel``, ``tdmpc_critic``, ``dense``,
``dense_chunk``, ``outcome``) are implementation detail.

Legacy aliases — kept for back-compat with workspace / tests / scripts:
  ``dino_wmpo_ppo_step``   → ``dino_wmpo_dense_step``
  ``dino_wmpo_chunk_step`` → ``dino_wmpo_outcome_step``
  ``dino_wmpo_frame_step`` → ``dino_wmpo_dense_step``
  ``dino_wmpo_window_step``→ ``dino_wmpo_outcome_step``
"""

from src.algorithms.ppo.dense import dino_wmpo_dense_step
from src.algorithms.ppo.dense_chunk import dino_wmpo_dense_chunk_step
from src.algorithms.ppo.outcome import dino_wmpo_outcome_step

# Legacy aliases — DO NOT USE in new code.
dino_wmpo_frame_step = dino_wmpo_dense_step
dino_wmpo_window_step = dino_wmpo_outcome_step
dino_wmpo_ppo_step = dino_wmpo_dense_step
dino_wmpo_chunk_step = dino_wmpo_outcome_step

__all__ = [
    "dino_wmpo_dense_step",
    "dino_wmpo_dense_chunk_step",
    "dino_wmpo_outcome_step",
    # Legacy aliases.
    "dino_wmpo_frame_step",
    "dino_wmpo_window_step",
    "dino_wmpo_ppo_step",
    "dino_wmpo_chunk_step",
]
