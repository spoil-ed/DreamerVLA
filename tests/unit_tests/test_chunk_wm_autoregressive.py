from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from dreamervla.models.embodiment.world_model.wm_chunk import ChunkAwareWorldModel


def _tiny_chunk_wm(**overrides) -> ChunkAwareWorldModel:
    cfg = {
        "obs_dim": 8,
        "action_dim": 2,
        "token_count": 2,
        "token_dim": 4,
        "time_horizon": 2,
        "latent_stage": "query_after",
        "latent_source": "tiny tokenized test input",
        "action_emb_dim": 2,
        "num_action_repeat": 1,
        "model_dim": 6,
        "depth": 1,
        "heads": 2,
        "dim_head": 4,
        "mlp_dim": 16,
        "dropout": 0.0,
        "num_hist": 3,
        "chunk_size": 3,
        "max_seq_len": 16,
        "reward_head_type": "none",
    }
    cfg.update(overrides)
    return ChunkAwareWorldModel(**cfg)


def _run_chunk_rollout(wm, seed: int = 0):
    torch.manual_seed(seed)
    b, h, n, d = 1, wm.num_hist, wm.token_count, wm.token_dim
    history = torch.randn(b, h, n, d)
    latent = {
        "hidden": history[:, -1],
        "history": history,
        "actions": torch.zeros(b, h, wm.action_dim),
    }
    action_chunk = torch.randn(b, wm.chunk_size, wm.action_dim)
    out = wm.predict_next_chunk(latent, action_chunk)
    loss = out["hidden_seq"].pow(2).sum()
    wm.zero_grad(set_to_none=True)
    loss.backward()
    grads = {k: v.grad.detach().clone() for k, v in wm.named_parameters() if v.grad is not None}
    return out["hidden_seq"].detach().clone(), grads


def test_grad_checkpoint_defaults_off() -> None:
    assert _tiny_chunk_wm().grad_checkpoint is False


def test_chunk_wm_layer_normalizes_only_raw_vision_tokens() -> None:
    wm = _tiny_chunk_wm(
        proprio_dim=3,
        proprio_emb_dim=2,
        model_dim=8,
        token_normalization="layer_norm",
        token_norm_eps=1.0e-6,
    )
    raw = torch.arange(1, 1 + 1 * 3 * 2 * 4, dtype=torch.float32).reshape(
        1, 3, 2, 4
    )

    normalized = wm._normalize_raw_vision_tokens(raw)

    assert torch.allclose(normalized.mean(dim=-1), torch.zeros(1, 3, 2), atol=1e-6)
    assert torch.allclose(
        normalized.var(dim=-1, unbiased=False),
        torch.ones(1, 3, 2),
        atol=1e-5,
    )

    predicted_with_proprio = torch.randn(1, 3, 2, wm.obs_token_dim)
    assert torch.equal(
        wm._obs_tokens_from_obs(predicted_with_proprio),
        predicted_with_proprio,
    )


def test_chunk_loss_reports_visual_one_step_and_persistence_cosines() -> None:
    """Comparable cosine metrics use one normalized visual transition only."""

    torch.manual_seed(7)
    wm = _tiny_chunk_wm(token_normalization="layer_norm")
    time_steps = wm.num_hist + wm.chunk_size
    obs = torch.randn(2, time_steps, wm.token_count, wm.token_dim)
    normalized = wm._normalize_raw_vision_tokens(obs)
    target = normalized[:, wm.num_hist : wm.num_hist + wm.chunk_size]

    def perfect_chunk(self, latent, action_chunk):
        del action_chunk
        hidden_seq = target.to(
            device=latent["history"].device,
            dtype=latent["history"].dtype,
        )
        return {
            "hidden": hidden_seq[:, -1],
            "hidden_seq": hidden_seq,
            "history": latent["history"],
            "actions": latent["actions"],
            "lang": latent.get("lang"),
        }

    wm.predict_next_chunk = types.MethodType(perfect_chunk, wm)
    metrics = wm.chunk_loss(
        {
            "obs_embedding": obs,
            "actions": torch.randn(2, time_steps, wm.action_dim),
        }
    )

    expected_persistence = torch.nn.functional.cosine_similarity(
        normalized[:, wm.num_hist - 1].float(),
        normalized[:, wm.num_hist].float(),
        dim=-1,
    ).mean()
    assert torch.allclose(metrics["one_step_cosine_similarity"], torch.ones(()))
    assert torch.allclose(metrics["chunk_cosine_similarity"], torch.ones(()))
    assert torch.allclose(
        metrics["persistence_cosine_similarity"],
        expected_persistence,
    )
    assert not metrics["one_step_cosine_similarity"].requires_grad
    assert not metrics["persistence_cosine_similarity"].requires_grad


def test_comparable_chunk_cosine_excludes_proprio_dimensions() -> None:
    """Proprio reconstruction quality must not contaminate visual-token cosine."""

    torch.manual_seed(11)
    wm = _tiny_chunk_wm(
        proprio_dim=3,
        proprio_emb_dim=2,
        model_dim=8,
        token_normalization="layer_norm",
    )
    time_steps = wm.num_hist + wm.chunk_size
    obs = torch.randn(2, time_steps, wm.token_count, wm.token_dim)
    proprio = torch.randn(2, time_steps, wm.proprio_dim)
    normalized = wm._normalize_raw_vision_tokens(obs)
    obs_tokens = wm._observation_tokens(normalized, proprio)
    target = obs_tokens[:, wm.num_hist : wm.num_hist + wm.chunk_size].clone()
    target[..., wm.token_dim :] *= -1.0

    def wrong_proprio_chunk(self, latent, action_chunk):
        del action_chunk
        hidden_seq = target.to(
            device=latent["history"].device,
            dtype=latent["history"].dtype,
        )
        return {
            "hidden": hidden_seq[:, -1],
            "hidden_seq": hidden_seq,
            "history": latent["history"],
            "actions": latent["actions"],
            "lang": latent.get("lang"),
        }

    wm.predict_next_chunk = types.MethodType(wrong_proprio_chunk, wm)
    metrics = wm.chunk_loss(
        {
            "obs_embedding": obs,
            "proprio": proprio,
            "actions": torch.randn(2, time_steps, wm.action_dim),
        }
    )

    assert torch.allclose(metrics["one_step_cosine_similarity"], torch.ones(()))
    assert metrics["hidden_cosine_loss"] > 0.0


def test_chunk_wm_normalizes_online_observations_but_not_imagined_history() -> None:
    wm = _tiny_chunk_wm(token_normalization="layer_norm")
    calls = 0

    def count_norm(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = wm.obs_norm.register_forward_hook(count_norm)
    raw_history = torch.arange(
        1,
        1 + wm.num_hist * wm.token_count * wm.token_dim,
        dtype=torch.float32,
    ).reshape(1, wm.num_hist, wm.token_count, wm.token_dim)
    latent = wm.encode_latent(raw_history)
    assert calls == 1
    assert torch.allclose(
        latent["history"].mean(dim=-1),
        torch.zeros(1, wm.num_hist, wm.token_count),
        atol=1.0e-6,
    )

    next_raw = raw_history[:, :1] * 3.0 + 7.0
    observed = wm.observe_next(latent, next_raw, torch.zeros(1, wm.action_dim))
    assert calls == 2
    assert torch.allclose(
        observed["hidden"].mean(dim=-1),
        torch.zeros(1, wm.token_count),
        atol=1.0e-6,
    )

    wm.predict_next(observed, torch.zeros(1, wm.action_dim))
    assert calls == 2
    handle.remove()


def test_chunk_rollout_grad_checkpoint_is_numerically_equivalent() -> None:
    # Gradient checkpointing the autoregressive rollout must not change the math
    # (it only trades compute for activation memory). dropout=0 in the fixture.
    plain = _tiny_chunk_wm(grad_checkpoint=False).train()
    ckpt = _tiny_chunk_wm(grad_checkpoint=True).train()
    ckpt.load_state_dict(plain.state_dict())

    seq_plain, grads_plain = _run_chunk_rollout(plain)
    seq_ckpt, grads_ckpt = _run_chunk_rollout(ckpt)

    assert torch.allclose(seq_plain, seq_ckpt, atol=1e-6)
    assert grads_plain.keys() == grads_ckpt.keys()
    for key in grads_plain:
        assert torch.allclose(grads_plain[key], grads_ckpt[key], atol=1e-6), key


def test_official_wm_objective_has_a_static_parameter_and_buffer_graph() -> None:
    """The reward-disabled tutorial objective is safe for DDP static_graph.

    The configured binary reward head is intentionally retained for later online
    use but has zero loss weight during official-data WM training. Its parameters
    are therefore unused, and that unused set must not vary by batch. Likewise,
    forward must not mutate buffers when per-forward DDP buffer broadcast is off.
    """

    wm = _tiny_chunk_wm(
        reward_head_type="binary",
        reward_loss_scale=0.0,
        chunk_rollout_chunks=4,
        chunk_rollout_loss_scale=0.2,
        grad_checkpoint=True,
        attn_impl="sdpa",
        dropout=0.1,
    ).train()
    before = {name: value.detach().clone() for name, value in wm.named_buffers()}
    unused_by_batch: list[tuple[str, ...]] = []

    for seed in (11, 12):
        torch.manual_seed(seed)
        time_steps = wm.num_hist + wm.chunk_rollout_chunks * wm.chunk_size
        batch = {
            "mode": "chunk_loss",
            "obs_embedding": torch.randn(
                2, time_steps, wm.token_count, wm.token_dim
            ),
            "actions": torch.randn(2, time_steps, wm.action_dim),
            "current_actions": torch.randn(2, time_steps, wm.action_dim),
        }
        wm.zero_grad(set_to_none=True)
        wm(batch)["_loss"].backward()
        unused_by_batch.append(
            tuple(
                name
                for name, parameter in wm.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            )
        )

    expected_unused = (
        "reward_norm.weight",
        "reward_norm.bias",
        "reward_head.0.weight",
        "reward_head.0.bias",
        "reward_head.2.weight",
        "reward_head.2.bias",
    )
    assert unused_by_batch == [expected_unused, expected_unused]
    assert all(
        torch.equal(before[name], value)
        for name, value in wm.named_buffers()
    )


def test_chunk_wm_requires_wm_concat_model_dim() -> None:
    with pytest.raises(ValueError, match="model_dim.*token_dim.*action_emb_dim"):
        _tiny_chunk_wm(model_dim=4)


def test_chunk_wm_source_uses_role_based_wm_wording() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "dreamervla"
        / "models"
        / "embodiment"
        / "world_model"
        / "wm_chunk.py"
    ).read_text(encoding="utf-8")
    assert ("DINO" + "-WM") not in source
    assert ("dino" + "_wm") not in source.lower()
    assert ("dino" + "wm") not in source.lower()


def test_encode_concats_action_to_each_obs_token_without_adding_slots() -> None:
    wm = _tiny_chunk_wm()
    history = torch.zeros(1, 3, 2, 4)
    zero_actions = torch.zeros(1, 3, 2)
    one_actions = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]])

    zero_z = wm._condition_tokens(history, zero_actions)
    one_z = wm._condition_tokens(history, one_actions)

    assert wm.slots_per_step == wm.token_count == 2
    assert zero_z.shape == (1, 3, 2, 6)
    assert one_z.shape == (1, 3, 2, 6)
    assert torch.allclose(zero_z[..., :4], history)
    assert torch.allclose(one_z[..., :4], history)
    assert torch.allclose(zero_z[:, :, 0, 4:], zero_z[:, :, 1, 4:])
    assert torch.allclose(one_z[:, :, 0, 4:], one_z[:, :, 1, 4:])
    assert not torch.allclose(zero_z[..., 4:], one_z[..., 4:])


def test_predict_next_uses_action_conditioned_tokens_with_nondivisible_dim() -> None:
    wm = _tiny_chunk_wm()
    history = torch.zeros(1, 3, 2, 4)
    latent = {
        "hidden": history[:, -1],
        "history": history,
        "actions": torch.zeros(1, 3, 2),
    }

    out = wm.predict_next(latent, torch.ones(1, 2))

    assert out["hidden"].shape == (1, 2, 4)
    assert out["history"].shape == (1, 3, 2, 4)


def test_chunk_wm_position_embedding_scales_with_history_not_max_seq_len() -> None:
    wm = _tiny_chunk_wm(token_count=5, obs_dim=20, max_seq_len=128)

    assert wm.pos_embedding.shape == (1, wm.num_hist * wm.token_count, wm.model_dim)


def test_predict_next_chunk_rolls_forward_autoregressively() -> None:
    wm = _tiny_chunk_wm()
    history = torch.arange(1 * 3 * 2 * 4, dtype=torch.float32).reshape(1, 3, 2, 4)
    actions = torch.zeros(1, 3, 2)
    latent = {
        "hidden": history[:, -1],
        "history": history.clone(),
        "actions": torch.zeros(1, 3, 2),
    }
    seen_histories: list[torch.Tensor] = []

    def fake_predict_next(self, latent_arg, action_arg):
        del action_arg
        current_history = latent_arg["history"].clone()
        seen_histories.append(current_history)
        next_hidden = current_history[:, -1] + 100.0 + len(seen_histories)
        next_history = torch.cat([current_history[:, 1:], next_hidden[:, None]], dim=1)
        return {
            "hidden": next_hidden,
            "hidden_seq": next_hidden[:, None],
            "history": next_history,
            "actions": latent_arg["actions"],
        }

    wm.predict_next = types.MethodType(fake_predict_next, wm)

    out = wm.predict_next_chunk(latent, actions)

    assert len(seen_histories) == 3
    assert torch.allclose(seen_histories[0], history)
    assert torch.allclose(seen_histories[1][:, -1], out["hidden_seq"][:, 0])
    assert torch.allclose(seen_histories[2][:, -2], out["hidden_seq"][:, 0])
    assert torch.allclose(seen_histories[2][:, -1], out["hidden_seq"][:, 1])
    assert torch.allclose(out["hidden"], out["hidden_seq"][:, -1])


def test_chunk_loss_uses_current_actions_for_transition_targets() -> None:
    wm = _tiny_chunk_wm(chunk_rollout_chunks=2, chunk_rollout_loss_scale=1.0)
    H = wm.num_hist
    K = wm.chunk_size
    T = H + 2 * K
    obs = torch.randn(1, T, wm.token_count, wm.token_dim)
    previous_actions = torch.full((1, T, wm.action_dim), -10.0)
    current_actions = torch.arange(
        T * wm.action_dim,
        dtype=torch.float32,
    ).reshape(1, T, wm.action_dim)
    captured_chunks: list[torch.Tensor] = []
    captured_history_actions: list[torch.Tensor] = []

    def fake_predict_next_chunk(self, latent_arg, action_chunk):
        captured_chunks.append(action_chunk.detach().clone())
        captured_history_actions.append(latent_arg["actions"].detach().clone())
        hidden_seq = torch.zeros(1, K, self.token_count, self.token_dim)
        next_hidden = hidden_seq[:, -1]
        next_history = torch.cat(
            [latent_arg["history"][:, 1:], next_hidden[:, None]],
            dim=1,
        )
        next_actions = torch.zeros_like(latent_arg["actions"])
        return {
            "hidden": next_hidden,
            "hidden_seq": hidden_seq,
            "history": next_history,
            "actions": next_actions,
        }

    wm.predict_next_chunk = types.MethodType(fake_predict_next_chunk, wm)

    wm(
        {
            "mode": "chunk_loss",
            "obs_embedding": obs,
            "actions": previous_actions,
            "current_actions": current_actions,
        }
    )

    assert len(captured_chunks) == 2
    assert torch.allclose(captured_chunks[0], current_actions[:, H - 1 : H - 1 + K])
    assert torch.allclose(
        captured_chunks[1],
        current_actions[:, H - 1 + K : H - 1 + 2 * K],
    )
    assert torch.allclose(
        captured_history_actions[0][:, : H - 1],
        current_actions[:, : H - 1],
    )
    assert torch.allclose(captured_history_actions[0][:, -1], current_actions[:, H - 1])
