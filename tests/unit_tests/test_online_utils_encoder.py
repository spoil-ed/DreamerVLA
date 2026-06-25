from __future__ import annotations

from types import SimpleNamespace

import torch


def test_build_encoder_uses_configured_hydra_target(monkeypatch):
    import dreamervla.runners.online_utils as mod

    captured: dict[str, object] = {}

    class FakeEncoder(torch.nn.Module):
        pass

    def fake_instantiate(cfg):
        captured["cfg"] = dict(cfg)
        return FakeEncoder()

    monkeypatch.setattr(mod.hydra.utils, "instantiate", fake_instantiate)
    monkeypatch.setattr(mod, "freeze_module", lambda encoder: captured.setdefault("frozen", encoder))
    monkeypatch.setattr(mod, "checkpoints_path", lambda *parts: "/tmp/" + "/".join(parts))

    args = SimpleNamespace(
        vla_ckpt_path="/tmp/vla",
        encoder_state_ckpt=None,
        action_head_type="oft_discrete_token",
        encoder_target="tests.fake.Encoder",
    )

    encoder = mod.build_encoder(args, torch.device("cpu"))

    assert isinstance(encoder, FakeEncoder)
    assert captured["frozen"] is encoder
    cfg = captured["cfg"]
    assert cfg["_target_"] == "tests.fake.Encoder"
    assert cfg["model_path"] == "/tmp/vla"
    assert cfg["action_head_type"] == "oft_discrete_token"
