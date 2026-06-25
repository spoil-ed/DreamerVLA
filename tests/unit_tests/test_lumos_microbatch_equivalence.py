"""MEM-RL-01: micro-batching the LUMOS update must not change the math.

Safety gate: running `dino_lumos_step` with the whole effective batch in one
slice vs split into group-aligned micro-batches must produce the SAME policy gradient.
Uses deterministic mocks (no RNG) and a parity classifier so GRPO groups have non-zero
within-group variance deterministically, independent of slicing.
"""

import torch
from omegaconf import OmegaConf

from dreamervla.algorithms.ppo.outcome import dino_lumos_step


class _DetWM(torch.nn.Module):
    def forward(self, batch):
        mode = batch["mode"]
        if mode == "observe_sequence":
            return {"latent": batch["obs_embedding"]}
        if mode == "actor_input":
            latent = batch["latent"]
            return latent["hidden"] if isinstance(latent, dict) else latent
        if mode == "predict_next_chunk":
            latent = batch["latent"]
            hidden = latent["hidden"] if isinstance(latent, dict) else latent
            chunk = hidden.unsqueeze(1).repeat(1, batch["actions"].shape[1], 1)
            return {"hidden_seq": chunk, "history": hidden, "actions": batch["actions"], "hidden": hidden}
        raise ValueError(mode)


class _ParityClassifier(torch.nn.Module):
    """complete = (rollout index % 2 == 0) → within each group_size=2 block one
    success + one failure → non-zero, deterministic, slice-invariant advantage."""

    def predict_success(self, video, threshold, stride=1, min_steps=1):
        del threshold, stride, min_steps
        b = video.shape[0]
        idx = torch.arange(b, device=video.device)
        return {
            "complete": (idx % 2 == 0),
            "finish_step": torch.zeros(b, dtype=torch.long, device=video.device),
        }


class _DetPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_value = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, batch):
        hidden = batch["hidden"]
        b = int(hidden.shape[0])
        # Policy MEAN action (the learnable param), shared by sample + evaluate.
        mean_chunk = self.action_value.expand(b, 2, 1)
        if batch["mode"] == "sample":
            # The SAMPLED action is the mean plus a per-rollout offset, so
            # ``action != mean`` and ``new_lp`` has a NON-ZERO derivative w.r.t.
            # ``action_value`` at the eval point. The offset MUST vary WITHIN a
            # GRPO group: within-group advantages are zero-mean, so any quantity
            # that is constant within a group (e.g. derived from the shared start
            # latent) gets cancelled by the advantage and the gradient is
            # identically zero — a vacuous gate. We use ``index % group_size``
            # (here group_size=2), which a real stochastic policy emulates via
            # per-rollout sampling RNG. Crucially this is SLICE-INVARIANT: group
            # alignment forces every slice to begin at a multiple of group_size,
            # so the local ``arange(b) % 2`` equals the global within-group parity
            # — and the gate would FAIL if a buggy micro-batch broke alignment.
            parity = torch.arange(b, device=hidden.device) % 2
            offset = (parity - 0.5).reshape(b, 1, 1) * 0.1  # even -> -0.05, odd -> +0.05
            action_chunk = mean_chunk + offset
            return action_chunk, torch.zeros(b, device=hidden.device), {"action_chunk": action_chunk}
        if batch["mode"] == "evaluate":
            action = batch["action"]
            target = mean_chunk if action.ndim == 3 else mean_chunk[:, 0, :]
            lp = -((action - target) ** 2).reshape(b, -1).sum(dim=-1)
            return lp, torch.zeros(b, device=hidden.device), {"action_chunk": mean_chunk}
        raise ValueError(batch["mode"])


def _cfg(micro_batch_starts, update_epochs=1):
    return OmegaConf.create({
        "lumos": {
            "chunk_size": 2, "episode_max_steps": 4, "classifier_min_steps": 1,
            "classifier_granularity": "action", "filter_zero_variance_groups": False,
            "update_micro_batch_starts": micro_batch_starts,
        },
        "ppo_rollouts_per_start": 2, "ppo_update_epochs": update_epochs,
        "kl_coef": 0.0, "actor_bc_to_ref_scale": 0.0,
        "clip_ratio_low": 0.2, "clip_ratio_high": 0.28, "advantage_eps": 1.0e-6,
        "imag_last": 2,
    })


def _run_update(micro_batch_starts, update_epochs=1, lr=0.0):
    """Run one LUMOS step; return (policy .grad, final param) after it.

    lr=0 freezes params so ``.grad`` is read directly; lr>0 lets params move
    between PPO epochs so the final param value exercises the multi-epoch path.
    """
    torch.manual_seed(0)
    policy = _DetPolicy()
    opt = torch.optim.SGD(policy.parameters(), lr=lr)
    obs = {"obs_embedding": torch.arange(4 * 4 * 1, dtype=torch.float32).reshape(4, 4, 1)}
    dino_lumos_step(
        policy=policy, chunk_world_model=_DetWM(), classifier=_ParityClassifier(),
        classifier_threshold=0.5, actor_optimizer=opt, obs=obs, device=torch.device("cpu"),
        algorithm_cfg=_cfg(micro_batch_starts, update_epochs),
        optim_cfg=OmegaConf.create({"grad_clip_norm": 1e9, "zero_grad_set_to_none": True}),
        ref_policy=None,
    )
    return policy.action_value.grad.clone(), policy.action_value.detach().clone()


def test_predict_next_chunk_mb_matches_full_batch():
    from dreamervla.algorithms.ppo.outcome import _predict_next_chunk_mb

    wm = _DetWM()
    current = {"hidden": torch.randn(6, 3)}
    action = torch.randn(6, 2, 1)
    full = _predict_next_chunk_mb(wm, current, action, 0)
    micro = _predict_next_chunk_mb(wm, current, action, 2)
    assert full.keys() == micro.keys()
    for key in full:
        assert torch.allclose(full[key], micro[key]), key


def test_microbatch_matches_full_batch_gradient():
    g_full, _ = _run_update(micro_batch_starts=0)   # 0 / unset -> one slice (whole batch)
    g_micro, _ = _run_update(micro_batch_starts=1)  # 1 start (=group_size rollouts) per slice
    # Guard against the fixture regressing to a vacuous (identically-zero) gate.
    assert g_full.abs().item() > 1e-6, f"fixture must produce a non-zero gradient, got {g_full}"
    assert torch.allclose(g_full, g_micro, atol=1e-6), (g_full, g_micro)


def test_microbatch_matches_full_batch_multiepoch():
    # lr>0 so params move between PPO epochs; the micro-batch host buffer must
    # reproduce the full-batch multi-epoch trajectory exactly (each epoch
    # re-evaluates the same stored trajectory under the just-stepped params).
    _, p_full = _run_update(micro_batch_starts=0, update_epochs=3, lr=0.05)
    _, p_micro = _run_update(micro_batch_starts=1, update_epochs=3, lr=0.05)
    assert torch.allclose(p_full, p_micro, atol=1e-6), (p_full, p_micro)
