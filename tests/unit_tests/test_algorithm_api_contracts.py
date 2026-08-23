from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from dreamervla.algorithms.actor.vla_policy import VLAPolicy
from dreamervla.algorithms.critic.critic import Critic
from dreamervla.algorithms.dreamervla import world_model_pretrain_step
from dreamervla.algorithms.ppo.grpo import (
    _entropy_coef,
    _repeat_latent,
    _slice_latent,
    masked_mean_ratio_chunk_term,
)
from dreamervla.algorithms.ppo.outcome import (
    _adaptive_group_advantage_and_mask,
    build_valid_chunk_count,
)
from dreamervla.algorithms.ppo.relabel import _real_relabel_anchor_loss
from dreamervla.algorithms.ppo.tdmpc_critic import (
    _sequence_field,
    _tdmpc_action_dim,
    _tdmpc_prepare_action,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"action_dim": 0}, "action_dim"),
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"policy_head_hidden_dim": 0}, "policy_head_hidden_dim"),
        ({"num_layers": 0}, "num_layers"),
        ({"act": "mystery"}, "act"),
        ({"initial_log_std": math.nan}, "initial_log_std"),
        ({"min_log_std": 2.0, "max_log_std": -2.0}, "min_log_std"),
    ],
)
def test_vla_policy_rejects_invalid_constructor_contracts(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        VLAPolicy(**kwargs)


def test_vla_policy_requires_a_supported_observation() -> None:
    policy = VLAPolicy(hidden_dim=4)

    with pytest.raises(KeyError, match="obs_embedding.*proprio.*state.*image"):
        policy.encode({"task_id": torch.tensor([1, 2])})


def test_vla_policy_rejects_hidden_width_mismatch() -> None:
    policy = VLAPolicy(hidden_dim=4)

    with pytest.raises(ValueError, match="hidden feature dim"):
        policy.sample_action_from_embedding(torch.zeros(2, 3))


def test_vla_policy_sample_evaluate_log_probability_roundtrip() -> None:
    policy = VLAPolicy(action_dim=3, hidden_dim=4)
    hidden = torch.randn(5, 4)

    action, sampled_log_prob, _ = policy.sample_action_from_embedding(hidden)
    evaluated_log_prob, entropy, _ = policy.evaluate_action_from_embedding(hidden, action)

    assert action.shape == (5, 3)
    assert torch.allclose(sampled_log_prob, evaluated_log_prob)
    assert torch.isfinite(entropy).all()


@pytest.mark.parametrize("kwargs", [{"hidden_dim": 0}, {"critic_hidden_dim": 0}])
def test_critic_rejects_nonpositive_dimensions(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        Critic(**kwargs)


@pytest.mark.parametrize("repeats", [0, -1])
def test_repeat_latent_rejects_nonpositive_repeats(repeats: int) -> None:
    with pytest.raises(ValueError, match="repeats"):
        _repeat_latent(torch.ones(2, 3), repeats)


@pytest.mark.parametrize(("lo", "hi"), [(-1, 1), (1, 1), (2, 1), (0, 4)])
def test_slice_latent_rejects_invalid_bounds(lo: int, hi: int) -> None:
    with pytest.raises(ValueError, match="slice bounds"):
        _slice_latent(torch.ones(3, 2), lo, hi)


@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_entropy_coefficient_must_be_nonnegative_and_finite(value: float) -> None:
    with pytest.raises(ValueError, match="entropy coefficient"):
        _entropy_coef({"actent": value})


def test_masked_mean_ratio_requires_aligned_vectors_and_batch_size() -> None:
    values = torch.ones(2)
    mask = torch.ones(2)
    counts = torch.ones(2)

    with pytest.raises(ValueError, match="b_eff"):
        masked_mean_ratio_chunk_term(values, mask, counts, 0)
    with pytest.raises(ValueError, match="matching 1D shapes"):
        masked_mean_ratio_chunk_term(values, mask[:1], counts, 2)
    with pytest.raises(ValueError, match="global b_eff=1"):
        masked_mean_ratio_chunk_term(values, mask, counts, 1)


@pytest.mark.parametrize(
    ("finish_step", "chunk_size", "num_chunks", "message"),
    [
        (torch.tensor([0]), 0, 2, "chunk_size"),
        (torch.tensor([0]), 1, 0, "num_chunks"),
        (torch.tensor([], dtype=torch.long), 1, 2, "non-empty"),
        (torch.tensor([[0]]), 1, 2, "1D"),
        (torch.tensor([0.5]), 1, 2, "integer"),
        (torch.tensor([-1]), 1, 2, "finish_step"),
        (torch.tensor([2]), 1, 2, "finish_step"),
    ],
)
def test_valid_chunk_count_rejects_invalid_geometry(
    finish_step: torch.Tensor,
    chunk_size: int,
    num_chunks: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_valid_chunk_count(finish_step, chunk_size, num_chunks)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"group_size_min": 0, "group_size_max": 2, "eps": 1e-6}, "group_size"),
        ({"group_size_min": 3, "group_size_max": 2, "eps": 1e-6}, "group_size"),
        ({"group_size_min": 1, "group_size_max": 2, "eps": 0.0}, "eps"),
    ],
)
def test_adaptive_group_advantage_rejects_invalid_geometry(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _adaptive_group_advantage_and_mask(torch.tensor([0.0, 1.0]), **kwargs)


def test_sequence_field_rejects_present_non_tensor_values() -> None:
    with pytest.raises(TypeError, match="obs.rewards must be a Tensor"):
        _sequence_field(
            {"rewards": [1.0, 0.0]},
            ("rewards", "reward"),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_tdmpc_action_helpers_reject_invalid_geometry_and_values() -> None:
    with pytest.raises(ValueError, match="action_dim"):
        _tdmpc_action_dim({"action_dim": 0}, fallback=7)
    with pytest.raises(ValueError, match="action_dim"):
        _tdmpc_prepare_action(torch.ones(2, 3), action_dim=0)
    with pytest.raises(ValueError, match="finite"):
        _tdmpc_prepare_action(torch.tensor([[0.0, math.nan]]), action_dim=2)
    with pytest.raises(ValueError, match=r"\[B,A\].*\[B,T,A\]"):
        _tdmpc_prepare_action(torch.ones(1, 2, 3, 4), action_dim=2)


def test_masked_chunk_mean_rejects_invalid_counts_and_values() -> None:
    values = torch.ones(2)
    mask = torch.ones(2)

    with pytest.raises(ValueError, match="positive"):
        masked_mean_ratio_chunk_term(values, mask, torch.tensor([1.0, 0.0]), 2)
    values[0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        masked_mean_ratio_chunk_term(values, mask, torch.ones(2), 2)


class _GaussianPolicy(nn.Module):
    def forward(self, batch: dict[str, object]):
        hidden = batch["hidden"]
        action = batch["action"]
        assert isinstance(hidden, torch.Tensor)
        assert isinstance(action, torch.Tensor)
        mean = hidden[:, : action.shape[-1]]
        dist = torch.distributions.Normal(mean, torch.ones_like(mean))
        return (
            dist.log_prob(action).sum(dim=-1),
            dist.entropy().sum(dim=-1),
            {},
        )


def _valid_relabel_batch() -> dict[str, torch.Tensor]:
    return {
        "hidden": torch.tensor([[0.0, 0.5], [1.0, -0.5]]),
        "action": torch.tensor([[0.1, 0.2], [0.8, -0.2]]),
        "old_log_prob": torch.tensor([-2.0, -2.0]),
        "advantage": torch.tensor([1.0, -1.0]),
        "weight": torch.tensor([1.0, 0.5]),
    }


@pytest.mark.parametrize("missing", ["hidden", "action", "old_log_prob", "advantage", "weight"])
def test_real_relabel_rejects_missing_required_fields(missing: str) -> None:
    batch = _valid_relabel_batch()
    del batch[missing]

    with pytest.raises(KeyError, match=missing):
        _real_relabel_anchor_loss(_GaussianPolicy(), batch, 0.2, 0.2)


def test_real_relabel_rejects_misaligned_or_invalid_tensors() -> None:
    batch = _valid_relabel_batch()
    batch["advantage"] = torch.ones(3)
    with pytest.raises(ValueError, match="advantage"):
        _real_relabel_anchor_loss(_GaussianPolicy(), batch, 0.2, 0.2)

    batch = _valid_relabel_batch()
    batch["weight"][0] = -1.0
    with pytest.raises(ValueError, match="weight.*non-negative"):
        _real_relabel_anchor_loss(_GaussianPolicy(), batch, 0.2, 0.2)

    batch = _valid_relabel_batch()
    batch["action"][0, 0] = math.nan
    with pytest.raises(ValueError, match="action.*finite"):
        _real_relabel_anchor_loss(_GaussianPolicy(), batch, 0.2, 0.2)


def test_real_relabel_zero_weight_batch_is_not_reported_as_applied() -> None:
    batch = _valid_relabel_batch()
    batch["weight"].zero_()

    loss, metrics = _real_relabel_anchor_loss(_GaussianPolicy(), batch, 0.2, 0.2)

    assert loss is None
    assert metrics["real_relabel_applied"] == 0.0


def test_real_relabel_valid_batch_returns_finite_loss_and_metrics() -> None:
    loss, metrics = _real_relabel_anchor_loss(_GaussianPolicy(), _valid_relabel_batch(), 0.2, 0.2)

    assert loss is not None and torch.isfinite(loss)
    assert metrics["real_relabel_applied"] == 1.0
    assert math.isfinite(metrics["real_relabel_ratio_mean"])


class _PretrainContractWorldModel(nn.Module):
    def __init__(self, output: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.output = output

    def forward(self, batch: dict[str, object]):
        del batch
        if self.output == "mapping":
            return {"_loss": self.weight.square()}
        if self.output == "nan":
            return {"_loss": self.weight * torch.tensor(float("nan"))}
        return self.weight.square()


def _run_pretrain_contract_step(model: nn.Module, *, grad_clip_norm: float = 1.0) -> None:
    world_model_pretrain_step(
        policy=nn.Identity(),
        world_model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        batch={},
        device=torch.device("cpu"),
        optim_cfg=OmegaConf.create({"precision": "fp32", "grad_clip_norm": grad_clip_norm}),
    )


def test_world_model_pretrain_requires_mapping_output() -> None:
    with pytest.raises(TypeError, match="mapping"):
        _run_pretrain_contract_step(_PretrainContractWorldModel("tensor"))


def test_world_model_pretrain_rejects_nonfinite_loss_before_update() -> None:
    model = _PretrainContractWorldModel("nan")

    with pytest.raises(ValueError, match="finite"):
        _run_pretrain_contract_step(model)

    assert model.weight.item() == 1.0


@pytest.mark.parametrize("grad_clip_norm", [0.0, -1.0, float("nan")])
def test_world_model_pretrain_requires_positive_grad_clip(grad_clip_norm: float) -> None:
    with pytest.raises(ValueError, match="grad_clip_norm"):
        _run_pretrain_contract_step(
            _PretrainContractWorldModel("mapping"), grad_clip_norm=grad_clip_norm
        )
