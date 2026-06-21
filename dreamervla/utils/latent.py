from __future__ import annotations

from dreamervla.models.world_model.dreamerv3_torch import DreamerV3LatentState


def slice_latent(latent: DreamerV3LatentState, t: int) -> DreamerV3LatentState:
    return DreamerV3LatentState(
        deter=latent.deter[:, t],
        stoch=latent.stoch[:, t],
        logits=latent.logits[:, t],
    )


def reward_of(world_model, latent: DreamerV3LatentState) -> float:
    return float(world_model.state_reward(latent).float().cpu().item())
