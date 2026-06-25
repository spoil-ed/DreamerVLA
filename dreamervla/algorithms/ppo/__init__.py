"""DreamerVLA PPO routes.

Three mature end-to-end PPO routes are exposed. The two primary routes
differ in the **reward form** seen by the actor:

  • ``dino_lumos_dense_step``       (dense per-step state-reward)
      A scalar reward is decoded from the world-model hidden at every
      imagined env-step; the γ-discounted sum becomes the rollout return.
      Per-step actor + per-step WM (``predict_next``).
      Implementation: ``dreamervla.algorithms.ppo.dense``.

  • ``dino_lumos_step``     (verifier outcome reward)
      A success verifier scores the imagined latent video and emits sparse
      ``(complete, finish_step)`` plus optional continuous score tensors.
      ``algorithm.lumos.reward_model`` selects how that verifier output becomes
      reward. Per-chunk actor + per-chunk WM (``predict_next_chunk``),
      eos_mask truncation, KL-into-reward, zero-variance group filtering.
      Implementation: ``dreamervla.algorithms.ppo.outcome``.

  • ``dino_lumos_dense_chunk_step`` (dense reward + chunk rollout)
      Same dense per-step state-reward as ``dense_step``, but driven by
      the chunk WM so each actor decision produces a full K-step action
      chunk (one PPO log_prob per chunk). Bridges dense and outcome:
      aligns actor decision granularity with its natural chunk output
      without changing the reward form. TD-MPC critic and
      ``real_rollout_relabel`` side-losses not yet wired.
      Implementation: ``dreamervla.algorithms.ppo.dense_chunk``.

Submodules (``grpo``, ``relabel``, ``tdmpc_critic``, ``dense``,
``dense_chunk``, ``outcome``) are implementation detail.
"""

from dreamervla.algorithms.ppo.dense import dino_lumos_dense_step
from dreamervla.algorithms.ppo.dense_chunk import dino_lumos_dense_chunk_step
from dreamervla.algorithms.ppo.outcome import dino_lumos_step

__all__ = [
    "dino_lumos_dense_step",
    "dino_lumos_dense_chunk_step",
    "dino_lumos_step",
]
