"""DreamerVLA algorithms — mature end-to-end training routes.

Routes (public interface):

  • DreamerV3 actor-critic over WM imagination:
        ``imagine_actor_critic_step``, ``world_model_pretrain_step``
        (helpers: ``compute_lambda_returns``, ``compute_replay_lambda_returns``)
        from ``src.algorithms.dreamer_vla``

  • PPO routes — distinguished by **reward form**:
        ``dino_wmpo_dense_step``    (dense per-step state-reward)
        ``dino_wmpo_outcome_step``  (sparse outcome reward from
                                     LatentSuccessClassifier, WMPO/verl)
        from ``src.algorithms.ppo``
        (legacy aliases ``dino_wmpo_ppo_step`` / ``dino_wmpo_chunk_step`` /
        ``dino_wmpo_frame_step`` / ``dino_wmpo_window_step`` are retained
        for back-compat but discouraged in new code)

  • TD-MPC MPC planner (eval-time): ``src.algorithms.tdmpc_mpc``

Internal modules used to build these routes (e.g. anything under
``src.algorithms.ppo.{grpo,relabel,tdmpc_critic,dense,outcome}``) are
implementation detail; depend on the route entrypoints above.
"""
from .dreamer_vla import (
    compute_lambda_returns,
    compute_replay_lambda_returns,
    imagine_actor_critic_step,
    world_model_pretrain_step,
)
from .ppo import (
    dino_wmpo_chunk_step,  # legacy alias
    dino_wmpo_dense_step,
    dino_wmpo_frame_step,  # legacy alias
    dino_wmpo_outcome_step,
    dino_wmpo_ppo_step,    # legacy alias
    dino_wmpo_window_step, # legacy alias
)

__all__ = [
    # DreamerV3 actor-critic route.
    "imagine_actor_critic_step",
    "world_model_pretrain_step",
    "compute_lambda_returns",
    "compute_replay_lambda_returns",
    # PPO routes (new names, distinguished by reward form).
    "dino_wmpo_dense_step",
    "dino_wmpo_outcome_step",
    # Legacy aliases — back-compat only.
    "dino_wmpo_frame_step",
    "dino_wmpo_window_step",
    "dino_wmpo_ppo_step",
    "dino_wmpo_chunk_step",
]
