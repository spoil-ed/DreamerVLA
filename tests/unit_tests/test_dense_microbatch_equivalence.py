"""PERF-W6: micro-batching the dense / dense-chunk PPO update must not change the math.

Safety gate: running ``dino_lumos_dense_step`` / ``dino_lumos_dense_chunk_step`` with the
whole effective batch in one slice vs split into group-aligned micro-batches must produce
the SAME policy gradient (and the same multi-epoch param trajectory). Uses deterministic
mocks (no RNG); the action-dependent reward gives each rollout a distinct return so the
GRPO within-group advantage variance is non-zero and the gradient is non-trivial.

Modeled on ``tests/unit_tests/test_lumos_microbatch_equivalence.py``.
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from dreamervla.algorithms.ppo.dense import dino_lumos_dense_step
from dreamervla.algorithms.ppo.dense_chunk import dino_lumos_dense_chunk_step

GROUP_SIZE = 2
N_STARTS = 3  # B_eff = 6 rollouts; mb_starts=1 -> three group-aligned slices
HORIZON = 2
K = 2
ACTION_DIM = 1


def test_dense_ppo_messages_use_role_based_wm_wording():
    source = (
        Path(__file__).resolve().parents[2] / "dreamervla" / "algorithms" / "ppo" / "dense.py"
    ).read_text(encoding="utf-8")
    assert ("DINO" + "-WM") not in source
    assert ("dino" + "_wm") not in source.lower()
    assert ("dino" + "wm") not in source.lower()


class _DetWM(torch.nn.Module):
    """Deterministic frozen imagination env.

    ``reward`` decodes a scalar from the hidden so the imagined return depends on the
    executed action (the mocks make the hidden carry the action), giving per-rollout
    return variation -> non-zero within-group advantage.
    """

    def forward(self, batch):
        mode = batch["mode"]
        if mode == "observe_sequence":
            return {"latent": batch["obs_embedding"]}
        if mode == "actor_input":
            latent = batch["latent"]
            return latent["hidden"] if isinstance(latent, dict) else latent
        if mode == "predict_next":
            latent = batch["latent"]
            hidden = latent["hidden"] if isinstance(latent, dict) else latent
            # Fold the (mean of the) executed action into the next hidden so the
            # downstream reward varies with the action — deterministic, no RNG.
            act = batch["actions"]
            act_scalar = act.reshape(act.shape[0], -1).mean(dim=1, keepdim=True)
            return {"hidden": hidden + act_scalar}
        if mode == "predict_next_chunk":
            latent = batch["latent"]
            hidden = latent["hidden"] if isinstance(latent, dict) else latent
            act = batch["actions"]  # [B, K, A]
            chunk = hidden.unsqueeze(1).repeat(1, act.shape[1], 1)
            chunk = chunk + act.reshape(act.shape[0], act.shape[1], -1).mean(dim=2, keepdim=True)
            return {
                "hidden_seq": chunk,
                "history": hidden,
                "actions": act,
                "hidden": hidden,
            }
        if mode == "reward":
            latent = batch["latent"]
            hidden = latent["hidden"] if isinstance(latent, dict) else latent
            return hidden.reshape(hidden.shape[0], -1).sum(dim=1)
        raise ValueError(mode)


class _DetPolicy(torch.nn.Module):
    """One learnable scalar mean action; per-rollout deterministic sampled offset.

    The offset varies WITHIN a GRPO group via ``arange(b) % group_size`` (slice-invariant
    under group alignment), so within-group advantages are not all equal and the PPO
    gradient w.r.t. ``action_value`` is non-zero. ``evaluate`` returns a differentiable
    log-prob; ``sample`` returns the action chunk (used for BC anchor / WM execution).
    """

    def __init__(self):
        super().__init__()
        self.action_value = torch.nn.Parameter(torch.tensor(0.5))
        self.evaluate_batch_sizes: list[int] = []

    def forward(self, batch):
        hidden = batch["hidden"]
        b = int(hidden.shape[0])
        mean_chunk = self.action_value.expand(b, K, ACTION_DIM)
        if batch["mode"] == "evaluate":
            self.evaluate_batch_sizes.append(b)
        if batch["mode"] == "sample":
            parity = torch.arange(b, device=hidden.device) % GROUP_SIZE
            offset = (parity.float() - 0.5).reshape(b, 1, 1) * 0.1
            action_chunk = mean_chunk + offset
            lp = torch.zeros(b, device=hidden.device)
            if bool(batch.get("return_chunk", False)):
                return action_chunk, lp, {"action_chunk": action_chunk}
            return action_chunk[:, 0, :], lp, {"action_chunk": action_chunk}
        if batch["mode"] == "evaluate":
            action = batch["action"]
            target = mean_chunk if action.ndim == 3 else mean_chunk[:, 0, :]
            lp = -((action - target) ** 2).reshape(b, -1).sum(dim=-1)
            return lp, torch.zeros(b, device=hidden.device), {"action_chunk": mean_chunk}
        raise ValueError(batch["mode"])


def _base_cfg(micro_batch_starts, update_epochs, *, bc_scale=0.0):
    return OmegaConf.create(
        {
            "lumos": {
                "chunk_size": K,
                "update_micro_batch_starts": micro_batch_starts,
            },
            "imagination_horizon": HORIZON,
            "imag_last": N_STARTS,
            "ppo_rollouts_per_start": GROUP_SIZE,
            "ppo_update_epochs": update_epochs,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "advantage_eps": 1.0e-6,
            "kl_coef": 0.0,
            "ppo_gamma": 0.99,
            "actor_bc_to_ref_scale": bc_scale,
            "rssm_action_scale": "policy",
        }
    )


def _obs():
    # observe_sequence returns this verbatim as the latent [B, T, D]; with imag_last=N_STARTS
    # and group_size=GROUP_SIZE the effective batch is N_STARTS * GROUP_SIZE.
    return {"obs_embedding": torch.zeros(1, N_STARTS, 1)}


def _run_dense(micro_batch_starts, *, update_epochs=1, lr=0.0, bc_scale=0.0, with_ref=False):
    torch.manual_seed(0)
    policy = _DetPolicy()
    ref_policy = None
    if with_ref:
        ref_policy = _DetPolicy()
        ref_policy.action_value = torch.nn.Parameter(torch.tensor(0.2))
        for p in ref_policy.parameters():
            p.requires_grad = False
    opt = torch.optim.SGD(policy.parameters(), lr=lr)
    dino_lumos_dense_step(
        policy=policy,
        world_model=_DetWM(),
        actor_optimizer=opt,
        obs=_obs(),
        device=torch.device("cpu"),
        algorithm_cfg=_base_cfg(micro_batch_starts, update_epochs, bc_scale=bc_scale),
        optim_cfg=OmegaConf.create({"grad_clip_norm": 1e9, "zero_grad_set_to_none": True}),
        ref_policy=ref_policy,
    )
    return policy.action_value.grad.clone(), policy.action_value.detach().clone(), policy


def _run_dense_chunk(micro_batch_starts, *, update_epochs=1, lr=0.0):
    torch.manual_seed(0)
    policy = _DetPolicy()
    opt = torch.optim.SGD(policy.parameters(), lr=lr)
    dino_lumos_dense_chunk_step(
        policy=policy,
        chunk_world_model=_DetWM(),
        actor_optimizer=opt,
        obs=_obs(),
        device=torch.device("cpu"),
        algorithm_cfg=_base_cfg(micro_batch_starts, update_epochs),
        optim_cfg=OmegaConf.create({"grad_clip_norm": 1e9, "zero_grad_set_to_none": True}),
        ref_policy=None,
    )
    return policy.action_value.grad.clone(), policy.action_value.detach().clone(), policy


def test_dense_microbatch_actually_slices_the_batch():
    # RED guard: with the knob ON, the policy must see SMALLER evaluate batches than the
    # full B_eff (proving the loop partitioned). Full path always evaluates the whole batch.
    _, _, p_full = _run_dense(0)
    _, _, p_micro = _run_dense(1)
    b_eff = N_STARTS * GROUP_SIZE
    assert max(p_full.evaluate_batch_sizes) == b_eff
    assert max(p_micro.evaluate_batch_sizes) == GROUP_SIZE, p_micro.evaluate_batch_sizes


def test_dense_microbatch_matches_full_batch_gradient():
    g_full, _, _ = _run_dense(0)  # 0 -> one slice (whole batch)
    g_micro, _, _ = _run_dense(1)  # 1 start (=group_size rollouts) per slice
    assert g_full.abs().item() > 1e-6, f"fixture must give a non-zero gradient, got {g_full}"
    assert torch.allclose(g_full, g_micro, atol=1e-6, rtol=1e-4), (g_full, g_micro)


def test_dense_microbatch_matches_full_batch_multiepoch():
    _, p_full, _ = _run_dense(0, update_epochs=3, lr=0.05)
    _, p_micro, _ = _run_dense(1, update_epochs=3, lr=0.05)
    assert torch.allclose(p_full, p_micro, atol=1e-6, rtol=1e-4), (p_full, p_micro)


def test_dense_microbatch_matches_with_bc_anchor():
    # BC ref loss is the non-clean-mean term (mean over horizon of mean over (B,K,A));
    # its micro-batch accumulation must use the global-B_eff normalizer to stay equivalent.
    g_full, _, _ = _run_dense(0, bc_scale=1.0, with_ref=True)
    g_micro, _, _ = _run_dense(1, bc_scale=1.0, with_ref=True)
    assert g_full.abs().item() > 1e-6, f"fixture must give a non-zero gradient, got {g_full}"
    assert torch.allclose(g_full, g_micro, atol=1e-6, rtol=1e-4), (g_full, g_micro)


def test_dense_chunk_microbatch_actually_slices_the_batch():
    _, _, p_full = _run_dense_chunk(0)
    _, _, p_micro = _run_dense_chunk(1)
    b_eff = N_STARTS * GROUP_SIZE
    assert max(p_full.evaluate_batch_sizes) == b_eff
    assert max(p_micro.evaluate_batch_sizes) == GROUP_SIZE, p_micro.evaluate_batch_sizes


def test_dense_chunk_microbatch_matches_full_batch_gradient():
    g_full, _, _ = _run_dense_chunk(0)
    g_micro, _, _ = _run_dense_chunk(1)
    assert g_full.abs().item() > 1e-6, f"fixture must give a non-zero gradient, got {g_full}"
    assert torch.allclose(g_full, g_micro, atol=1e-6, rtol=1e-4), (g_full, g_micro)


def test_dense_chunk_microbatch_matches_full_batch_multiepoch():
    _, p_full, _ = _run_dense_chunk(0, update_epochs=3, lr=0.05)
    _, p_micro, _ = _run_dense_chunk(1, update_epochs=3, lr=0.05)
    assert torch.allclose(p_full, p_micro, atol=1e-6, rtol=1e-4), (p_full, p_micro)
