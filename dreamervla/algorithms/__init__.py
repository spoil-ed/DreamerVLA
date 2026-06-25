"""DreamerVLA algorithms.

Routes (public interface):

  • DreamerV3 actor-critic over WM imagination:
        ``imagine_actor_critic_step``, ``world_model_pretrain_step``
        (helpers: ``compute_lambda_returns``, ``compute_replay_lambda_returns``)
        from ``dreamervla.algorithms.dreamervla``

  • PPO routes — distinguished by **reward form**:
        ``dino_lumos_dense_step``    (dense per-step state-reward)
        ``dino_lumos_step``  (verifier outcome reward selected by
                                     algorithm.lumos.reward_model, LUMOS/verl)
        from ``dreamervla.algorithms.ppo``

  • TD-MPC MPC planner (eval-time): ``dreamervla.algorithms.tdmpc_mpc``

Internal modules used to build these routes (e.g. anything under
``dreamervla.algorithms.ppo.{grpo,relabel,tdmpc_critic,dense,outcome}``) are
implementation detail; depend on the route entrypoints above.
"""

from .dreamervla import (
    compute_lambda_returns,
    compute_replay_lambda_returns,
    imagine_actor_critic_step,
    world_model_pretrain_step,
)
from .ppo import (
    dino_lumos_dense_chunk_step,
    dino_lumos_dense_step,
    dino_lumos_step,
)
from .registry import ActorUpdateRoute, actor_update_names, get_actor_update_route

__all__ = [
    # DreamerV3 actor-critic route.
    "imagine_actor_critic_step",
    "world_model_pretrain_step",
    "compute_lambda_returns",
    "compute_replay_lambda_returns",
    # PPO routes (distinguished by reward form).
    "dino_lumos_dense_step",
    "dino_lumos_dense_chunk_step",
    "dino_lumos_step",
    "ActorUpdateRoute",
    "actor_update_names",
    "get_actor_update_route",
]
