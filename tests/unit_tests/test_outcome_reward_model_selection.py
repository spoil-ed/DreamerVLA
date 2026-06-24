import torch

import dreamervla.algorithms.reward as reward_pkg


def test_outcome_step_resolves_reward_model_from_cfg(monkeypatch):
    seen = {}

    class _Spy:
        name = "spy"

        def build_reward(self, *, batch, max_steps, chunk_size, finish_step, complete, device):
            seen["called_with"] = (batch, max_steps, chunk_size)
            return torch.zeros((batch, max_steps), device=device)

    def _fake_get(name):
        seen["name"] = name
        return _Spy()

    monkeypatch.setattr(reward_pkg, "get_reward_model", _fake_get)

    from dreamervla.algorithms.ppo.outcome import _resolve_reward_tensor

    finish_step = torch.tensor([0, 1])
    complete = torch.tensor([True, False])
    out = _resolve_reward_tensor(
        wmpo_cfg={"reward_model": "spy"},
        batch=2,
        max_steps=4,
        chunk_size=2,
        finish_step=finish_step,
        complete=complete,
        device=torch.device("cpu"),
    )
    assert seen["name"] == "spy"
    assert seen["called_with"] == (2, 4, 2)
    assert out.shape == (2, 4)


def test_outcome_default_reward_model_is_sparse_outcome(monkeypatch):
    captured = {}
    real_get = reward_pkg.get_reward_model

    def _capturing_get(name):
        captured["name"] = name
        return real_get(name)

    monkeypatch.setattr(reward_pkg, "get_reward_model", _capturing_get)

    from dreamervla.algorithms.ppo.outcome import _resolve_reward_tensor

    _resolve_reward_tensor(
        wmpo_cfg={},
        batch=2,
        max_steps=4,
        chunk_size=2,
        finish_step=torch.tensor([0, 1]),
        complete=torch.tensor([True, False]),
        device=torch.device("cpu"),
    )
    assert captured["name"] == "sparse_outcome"
