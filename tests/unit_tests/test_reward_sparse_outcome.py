import torch

from dreamervla.algorithms.ppo.outcome import _build_reward_tensor
from dreamervla.algorithms.reward import get_reward_model


def test_sparse_outcome_matches_build_reward_tensor_bitforbit():
    batch, max_steps, K = 5, 12, 3
    finish_step = torch.tensor([0, 4, 11, 7, 3])
    complete = torch.tensor([True, False, True, True, False])

    model = get_reward_model("sparse_outcome")
    got = model.build_reward(
        batch=batch,
        max_steps=max_steps,
        chunk_size=K,
        finish_step=finish_step,
        complete=complete,
        device=torch.device("cpu"),
    )
    expected = _build_reward_tensor(
        batch=batch,
        max_steps=max_steps,
        chunk_size=K,
        finish_step=finish_step,
        complete=complete,
    )
    assert torch.equal(got, expected)
    # sparse 0/1: one positive per complete row at its finish column
    assert got.sum().item() == 3.0
    assert got[0, 0].item() == 1.0 and got[1].sum().item() == 0.0


def test_sparse_outcome_aliases_resolve():
    assert get_reward_model("outcome").name == "sparse_outcome"
    assert get_reward_model("sparse").name == "sparse_outcome"
