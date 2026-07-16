"""PERF-H4: per-step actor grad-norm / cosine diagnostics gate.

The diagnostics in ``imagine_actor_critic_step`` (per-component ``_flat_grad``
extra backward passes with ``retain_graph=True`` + five ``_named_grad_norm``
full-parameter traversals) are pure instrumentation. They must be gated behind
``optim_cfg.grad_diagnostics`` (default OFF) so they are not paid every step,
and turning them OFF must leave the training math (post-step params) identical.

CPU-only, tiny stub modules.
"""

from __future__ import annotations

import copy

import torch
from omegaconf import OmegaConf
from torch import nn

import dreamervla.algorithms.dreamervla as dvla
from dreamervla.algorithms.critic.twohot_critic import ReturnPercentileTracker, TwohotCritic
from dreamervla.algorithms.dreamervla import imagine_actor_critic_step

D_LAT = 4  # world-model latent / actor-critic feature dim
A_DIM = 3  # action dim
B = 2  # batch
T = 3  # observed sequence length
HORIZON = 2


class _StubWorldModel(nn.Module):
    """Minimal mode-dispatched WM exposing exactly the entry points the
    actor-critic step uses. No trainable params (WM is frozen during the step)."""

    def forward(self, batch):
        mode = batch["mode"]
        if mode == "observe_sequence":
            emb = batch["obs_embedding"]  # [B,T,D_LAT]
            return {"latent": emb}
        if mode in ("actor_input", "critic_input"):
            return batch["latent"]  # latent IS the feature [N,D_LAT]
        if mode == "predict_next":
            latent = batch["latent"]  # [N,D_LAT]
            actions = batch["actions"]  # [N,A_DIM]
            # Deterministic, no-grad transition (caller wraps in no_grad anyway).
            pad = torch.zeros(latent.shape[0], D_LAT, device=latent.device)
            pad[:, :A_DIM] = actions[:, :A_DIM]
            return latent + 0.1 * pad
        if mode == "reward":
            return batch["latent"].sum(dim=-1)  # [N]
        # continue / success_return fall back via the algorithm helpers.
        raise ValueError(f"Unknown mode: {mode}")


class _StubPolicy(nn.Module):
    """Gaussian policy: mean = Linear(feat), std = exp(log_std)."""

    def __init__(self) -> None:
        super().__init__()
        self.mean_head = nn.Linear(D_LAT, A_DIM)
        self.log_std = nn.Parameter(torch.zeros(A_DIM))

    def forward(self, batch):
        mode = batch["mode"]
        hidden = batch["hidden"]  # [N,D_LAT]
        mean = self.mean_head(hidden)
        std = self.log_std.exp().expand_as(mean)
        if mode == "sample":
            with torch.no_grad():
                noise = torch.randn_like(mean)
                action = (mean + std * noise).clamp(-1.0, 1.0)
            return action, None, {"mean": mean, "std": std}
        if mode == "evaluate":
            from torch.distributions import Normal

            lp = Normal(mean, std).log_prob(batch["action"]).sum(dim=-1)
            return lp, None, {}
        raise ValueError(f"Unknown policy mode: {mode}")


def _make_components():
    torch.manual_seed(0)
    world_model = _StubWorldModel()
    policy = _StubPolicy()
    critic = TwohotCritic(hidden_dim=D_LAT, num_bins=5, bin_min=-5.0, bin_max=5.0, critic_layers=0)
    target_critic = TwohotCritic(
        hidden_dim=D_LAT, num_bins=5, bin_min=-5.0, bin_max=5.0, critic_layers=0
    )
    target_critic.load_state_dict(critic.state_dict())
    return world_model, policy, critic, target_critic


def _algorithm_cfg():
    return OmegaConf.create(
        {
            "imagination_horizon": HORIZON,
            "lam": 0.95,
            "actent": 1.0e-3,
            "rssm_action_scale": "policy",  # identity action mapping for the stub WM
            "repval_loss": False,
            "slowreg": 0.0,
            "return_normalization": {"mode": "none"},
        }
    )


def _run_step(world_model, policy, critic, target_critic, grad_diagnostics, seed):
    torch.manual_seed(seed)
    actor_opt = torch.optim.SGD(policy.parameters(), lr=0.1)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=0.1)
    tracker = ReturnPercentileTracker(decay=0.99, low=0.05, high=0.95)
    obs = {"obs_embedding": torch.randn(B, T, D_LAT)}
    optim_cfg = OmegaConf.create(
        {
            "grad_clip_norm": 1.0,
            "zero_grad_set_to_none": True,
            "grad_diagnostics": grad_diagnostics,
        }
    )
    return imagine_actor_critic_step(
        policy=policy,
        world_model=world_model,
        critic=critic,
        target_critic=target_critic,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        return_tracker=tracker,
        obs=obs,
        device=torch.device("cpu"),
        algorithm_cfg=_algorithm_cfg(),
        optim_cfg=optim_cfg,
    )


def test_diagnostics_off_skips_extra_backward_and_named_grad_norm(monkeypatch):
    world_model, policy, critic, target_critic = _make_components()
    init_params = [p.detach().clone() for p in policy.parameters()]

    # Spy: extra autograd.grad calls come ONLY from the diagnostic _flat_grad
    # (the closure is not directly patchable). _named_grad_norm is module-level.
    grad_calls = {"n": 0}
    real_grad = torch.autograd.grad

    def _counting_grad(*args, **kwargs):
        grad_calls["n"] += 1
        return real_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", _counting_grad)

    named_calls = {"n": 0}
    real_named = dvla._named_grad_norm

    def _counting_named(*args, **kwargs):
        named_calls["n"] += 1
        return real_named(*args, **kwargs)

    monkeypatch.setattr(dvla, "_named_grad_norm", _counting_named)

    metrics = _run_step(world_model, policy, critic, target_critic, grad_diagnostics=False, seed=1)

    assert grad_calls["n"] == 0, "diagnostics OFF must not run extra autograd.grad"
    assert named_calls["n"] == 0, "diagnostics OFF must not call _named_grad_norm"

    # Step still updated the policy params.
    moved = any(
        not torch.equal(p.detach(), q)
        for p, q in zip(policy.parameters(), init_params, strict=True)
    )
    assert moved, "actor optimizer.step() must still update params when diagnostics OFF"

    # Gated metrics default to 0.0 when OFF.
    for key in (
        "actor_grad_norm_pg",
        "actor_grad_norm_bc_ref",
        "actor_grad_norm_entropy",
        "actor_grad_norm_bc_vla",
        "actor_grad_cos_pg_bcref",
        "actor_grad_norm_adapter",
        "actor_grad_norm_action_head",
        "actor_grad_norm_output_projection",
        "actor_grad_norm_policy_head",
        "actor_grad_norm_log_std",
    ):
        assert metrics[key] == 0.0


def test_diagnostics_off_params_identical_to_on_reference():
    # CRITICAL: turning diagnostics OFF must not change the training math. The
    # ON run (== today's behavior) and the OFF run start from identical clones
    # and the same seed, so post-step policy params must be bit-identical
    # (atol=0). The gated code reads grads / computes norms but never mutates
    # params, the loss, or the optimizer state.
    wm_on, pol_on, cr_on, tc_on = _make_components()
    wm_off = copy.deepcopy(wm_on)
    pol_off = copy.deepcopy(pol_on)
    cr_off = copy.deepcopy(cr_on)
    tc_off = copy.deepcopy(tc_on)

    _run_step(wm_on, pol_on, cr_on, tc_on, grad_diagnostics=True, seed=7)
    _run_step(wm_off, pol_off, cr_off, tc_off, grad_diagnostics=False, seed=7)

    for p, q in zip(pol_on.parameters(), pol_off.parameters(), strict=True):
        assert torch.equal(p, q)
    for p, q in zip(cr_on.parameters(), cr_off.parameters(), strict=True):
        assert torch.equal(p, q)


def test_diagnostics_on_runs_extra_backward(monkeypatch):
    world_model, policy, critic, target_critic = _make_components()

    grad_calls = {"n": 0}
    real_grad = torch.autograd.grad

    def _counting_grad(*args, **kwargs):
        grad_calls["n"] += 1
        return real_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", _counting_grad)

    metrics = _run_step(world_model, policy, critic, target_critic, grad_diagnostics=True, seed=1)

    # PG and entropy components are present (actent != 0) → at least 2 extra grads.
    assert grad_calls["n"] >= 1, "diagnostics ON must run the extra autograd.grad"
    # PG grad norm is populated (non-default) on the ON path.
    assert metrics["actor_grad_norm_pg"] > 0.0
