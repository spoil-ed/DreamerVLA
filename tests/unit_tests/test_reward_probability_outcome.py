import torch

from dreamervla.algorithms.reward import get_reward_model


def test_probability_outcome_places_score_at_score_step():
    model = get_reward_model("probability_outcome")
    reward = model.build_reward(
        batch=3,
        max_steps=6,
        chunk_size=2,
        finish_step=torch.tensor([0, 1, 2]),
        complete=torch.tensor([False, False, True]),
        score=torch.tensor([0.2, 0.8, 1.2]),
        score_step=torch.tensor([1, 4, 99]),
        device=torch.device("cpu"),
    )

    expected = torch.zeros(3, 6)
    expected[0, 1] = 0.2
    expected[1, 4] = 0.8
    expected[2, 5] = 1.0
    assert torch.equal(reward, expected)


def test_probability_outcome_aliases_resolve():
    assert get_reward_model("probability").name == "probability_outcome"
    assert get_reward_model("success-probability").name == "probability_outcome"
