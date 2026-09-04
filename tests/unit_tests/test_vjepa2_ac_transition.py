from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dreamervla.models.embodiment.world_model.vjepa2_ac_transition import (
    VJEPA2ACTransition,
    build_frame_causal_attention_mask,
)
from dreamervla.models.embodiment.world_model.wm_chunk import ChunkAwareWorldModel


def _tiny_transition(**overrides) -> VJEPA2ACTransition:
    config = {
        "input_dim": 5,
        "output_dim": 4,
        "action_dim": 2,
        "state_dim": 3,
        "token_count": 2,
        "predictor_dim": 16,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "use_rope": True,
        "spatial_grid": None,
    }
    config.update(overrides)
    return VJEPA2ACTransition(**config)


def _tiny_chunk_wm(**overrides) -> ChunkAwareWorldModel:
    config = {
        "obs_dim": 8,
        "action_dim": 2,
        "token_count": 2,
        "token_dim": 4,
        "time_horizon": 2,
        "latent_stage": "query_after",
        "latent_source": "tiny non-spatial token sequence",
        "action_emb_dim": 2,
        "num_action_repeat": 1,
        "model_dim": 6,
        "depth": 1,
        "heads": 2,
        "dim_head": 4,
        "mlp_dim": 16,
        "dropout": 0.0,
        "num_hist": 2,
        "chunk_size": 2,
        "max_seq_len": 16,
        "reward_head_type": "none",
        "transition_type": "vjepa2_ac",
        "transition_init": "random",
        "vjepa2_predictor_dim": 16,
        "vjepa2_depth": 2,
        "vjepa2_num_heads": 4,
        "vjepa2_mlp_ratio": 2.0,
        "vjepa2_spatial_grid": None,
    }
    config.update(overrides)
    return ChunkAwareWorldModel(**config)


def test_frame_causal_mask_exposes_current_and_past_frames_only() -> None:
    mask = build_frame_causal_attention_mask(frames=3, tokens_per_frame=2)
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_future_frame_cannot_change_past_transition_outputs() -> None:
    torch.manual_seed(7)
    transition = _tiny_transition().eval()
    tokens = torch.randn(1, 3, 2, 5)
    actions = torch.randn(1, 3, 2)
    states = torch.randn(1, 3, 3)
    reference = transition(tokens, actions, states)

    changed_tokens = tokens.clone()
    changed_actions = actions.clone()
    changed_states = states.clone()
    changed_tokens[:, -1].add_(100.0)
    changed_actions[:, -1].add_(100.0)
    changed_states[:, -1].add_(100.0)
    changed = transition(changed_tokens, changed_actions, changed_states)

    assert torch.equal(reference[:, :-1], changed[:, :-1])
    assert not torch.equal(reference[:, -1], changed[:, -1])


def test_non_spatial_tokens_must_not_be_assigned_a_fake_grid() -> None:
    with pytest.raises(ValueError, match="genuine spatial patch tokens"):
        _tiny_transition(spatial_grid=(1, 3))


def test_pretrained_load_is_explicit_and_never_resizes_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(11)
    transition = _tiny_transition(depth=1)
    source: dict[str, torch.Tensor] = {}
    for key, value in transition.state_dict().items():
        if key.startswith(("input_adapter.", "output_adapter.")):
            continue
        if key.startswith(("state_encoder.", "state_input_adapter.")):
            continue
        source[f"module.{key}"] = torch.full_like(value, 0.125)

    # Official predictor_embed/proj and state_encoder widths intentionally do
    # not match this representation.  The loader must report and skip them.
    source["module.predictor_embed.weight"] = torch.randn(16, 7)
    source["module.predictor_embed.bias"] = torch.randn(16)
    source["module.predictor_proj.weight"] = torch.randn(7, 16)
    source["module.predictor_proj.bias"] = torch.randn(7)
    source["module.state_encoder.weight"] = torch.full((16, 2), 0.125)
    source["module.state_encoder.bias"] = torch.randn(16)
    source["module.extrinsics_encoder.weight"] = torch.randn(16, 1)
    checkpoint_path = tmp_path / "vjepa2-ac.pt"
    torch.save({"predictor": source}, checkpoint_path)

    log_messages: list[str] = []

    def capture_log(message: str, *args: object) -> None:
        log_messages.append(message % args)

    monkeypatch.setattr(
        "dreamervla.models.embodiment.world_model.vjepa2_ac_transition.logger.info",
        capture_log,
    )
    report = transition.load_pretrained(checkpoint_path)

    assert "predictor_blocks.0.attn.qkv.weight" in report.loaded_keys
    assert any(
        "input_adapter.weight <- predictor_embed.weight" in key for key in report.mismatched_keys
    )
    assert any(
        "output_adapter.weight <- predictor_proj.weight" in key for key in report.mismatched_keys
    )
    assert "state_encoder.weight" in report.loaded_keys
    assert any("state_input_adapter.weight" in key for key in report.missing_keys)
    assert "extrinsics_encoder.weight" in report.unused_checkpoint_keys
    assert 0.0 < report.pretrained_parameter_ratio < 1.0
    assert torch.equal(
        transition.predictor_blocks[0].attn.qkv.weight,
        torch.full_like(transition.predictor_blocks[0].attn.qkv.weight, 0.125),
    )
    logs = "\n".join(log_messages)
    assert "loaded keys" in logs
    assert "missing keys" in logs
    assert "mismatched keys" in logs
    assert "pretrained parameter ratio" in logs


def test_chunk_world_model_vjepa_forward_backward_and_checkpoint_reload(tmp_path: Path) -> None:
    torch.manual_seed(19)
    model = _tiny_chunk_wm()
    history = torch.randn(2, model.num_hist, model.token_count, model.token_dim)
    latent = {
        "hidden": history[:, -1],
        "history": history,
        "actions": torch.zeros(2, model.num_hist, model.action_dim),
    }
    action_chunk = torch.randn(2, model.chunk_size, model.action_dim)

    output = model.predict_next_chunk(latent, action_chunk)
    assert output["hidden"].shape == (2, model.token_count, model.token_dim)
    assert output["hidden_seq"].shape == (
        2,
        model.chunk_size,
        model.token_count,
        model.token_dim,
    )
    loss = output["hidden_seq"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.vjepa2_transition is not None
    assert model.vjepa2_transition.predictor_blocks[0].attn.qkv.weight.grad is not None
    assert model.vjepa2_transition.input_adapter.weight.grad is not None
    assert model.vjepa2_transition.action_encoder.weight.grad is not None
    assert model.vjepa2_transition.output_adapter.weight.grad is not None

    checkpoint_path = tmp_path / "world_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    reloaded = _tiny_chunk_wm()
    reloaded.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=True)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        expected = model.predict_next_chunk(latent, action_chunk)["hidden_seq"]
        actual = reloaded.predict_next_chunk(latent, action_chunk)["hidden_seq"]
    assert torch.equal(expected, actual)


def test_vjepa_adapters_keep_visual_language_and_proprio_boundaries_separate() -> None:
    model = _tiny_chunk_wm(
        proprio_dim=3,
        proprio_emb_dim=2,
        lang_dim=5,
        lang_emb_dim=2,
        model_dim=10,
    )
    assert model.vjepa2_transition is not None
    transition = model.vjepa2_transition
    assert transition.input_adapter.in_features == model.token_dim == 4
    assert transition.output_adapter.out_features == model.token_dim == 4
    assert transition.action_encoder.in_features == model.action_dim == 2
    assert isinstance(transition.state_input_adapter, torch.nn.Linear)
    assert transition.state_input_adapter.in_features == model.proprio_dim == 3
    assert transition.state_input_adapter.out_features == model.action_dim == 2
    assert transition.state_encoder.in_features == model.action_dim == 2
    assert transition.context_encoder is not None
    assert transition.context_encoder.in_features == model.lang_condition_dim == 2
    assert transition.state_output_adapter is not None
    assert transition.state_output_adapter.out_features == model.proprio_condition_dim == 2

    history = torch.randn(2, model.num_hist, model.token_count, model.token_dim)
    latent = {
        "hidden": history[:, -1],
        "history": history,
        "actions": torch.zeros(2, model.num_hist, model.action_dim),
        "proprio": torch.randn(2, model.proprio_dim),
        "lang": torch.randn(2, model.lang_dim),
    }
    output = model.predict_next(latent, torch.randn(2, model.action_dim))
    assert output["hidden"].shape == (2, model.token_count, model.obs_token_dim)
    assert output["proprio"].shape == (2, model.proprio_dim)


def test_vjepa_chunk_loss_trains_for_multiple_steps_without_nan() -> None:
    torch.manual_seed(23)
    model = _tiny_chunk_wm()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    time_steps = model.num_hist + model.chunk_size
    batch = {
        "obs_embedding": torch.randn(2, time_steps, model.token_count, model.token_dim),
        "actions": torch.randn(2, time_steps, model.action_dim),
    }

    losses: list[float] = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model.chunk_loss(batch)
        loss = output["_loss"]
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert all(math_value == math_value for math_value in losses)


def test_vjepa_rollout_truncation_preserves_values_and_stops_recurrent_gradient() -> None:
    torch.manual_seed(29)
    truncated = _tiny_chunk_wm(vjepa2_truncate_rollout_gradients=True).train()
    full_bptt = _tiny_chunk_wm(vjepa2_truncate_rollout_gradients=False).train()
    full_bptt.load_state_dict(truncated.state_dict())

    history_truncated = torch.randn(
        2,
        truncated.num_hist,
        truncated.token_count,
        truncated.token_dim,
        requires_grad=True,
    )
    history_full = history_truncated.detach().clone().requires_grad_(True)
    actions = torch.randn(2, truncated.chunk_size, truncated.action_dim)

    def rollout(model: ChunkAwareWorldModel, history: torch.Tensor) -> torch.Tensor:
        latent = {
            "hidden": history[:, -1],
            "history": history,
            "actions": torch.zeros(2, model.num_hist, model.action_dim),
        }
        return model.predict_next_chunk(latent, actions)["hidden_seq"]

    truncated_output = rollout(truncated, history_truncated)
    full_output = rollout(full_bptt, history_full)
    assert torch.equal(truncated_output, full_output)

    truncated_output[:, -1].square().mean().backward()
    full_output[:, -1].square().mean().backward()
    assert history_truncated.grad is None
    assert history_full.grad is not None
    assert torch.isfinite(history_full.grad).all()


def test_vjepa_truncated_multichunk_loss_has_finite_gradients() -> None:
    torch.manual_seed(31)
    model = _tiny_chunk_wm(
        chunk_rollout_chunks=4,
        chunk_rollout_loss_scale=0.2,
        vjepa2_truncate_rollout_gradients=True,
    ).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    time_steps = model.num_hist + model.chunk_rollout_chunks * model.chunk_size
    batch = {
        "obs_embedding": torch.randn(2, time_steps, model.token_count, model.token_dim),
        "actions": torch.randn(2, time_steps, model.action_dim),
    }

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model.chunk_loss(batch)["_loss"]
        assert torch.isfinite(loss)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        assert torch.isfinite(grad_norm)
        optimizer.step()


def test_transition_ablation_rejects_pretrained_original_model() -> None:
    with pytest.raises(ValueError, match="requires transition_type='vjepa2_ac'"):
        _tiny_chunk_wm(transition_type="original", transition_init="pretrained")


def test_pi05_rgb_wm_config_declares_strict_transition_ablation_without_sidecars() -> None:
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        baseline = compose(
            config_name="train",
            overrides=["experiment=wm_pi05_collected_train"],
        )
        openvla = compose(
            config_name="train",
            overrides=["experiment=wm_full_dataset_train"],
        )
        config = compose(
            config_name="train",
            overrides=[
                "experiment=wm_pi05_collected_train",
                "world_model.transition_type=vjepa2_ac",
                "world_model.transition_init=random",
            ],
        )

    world_model = config.world_model
    worker = config.ray_components.world_model.kwargs
    assert baseline.world_model.transition_type == "original"
    assert baseline.world_model.transition_init == "random"
    assert "transition_type" not in openvla.world_model
    assert config.offline_warmup.hidden_dir is None
    assert config.offline_warmup.online_latent.enabled is True
    assert config.offline_warmup.online_latent.policy._target_.endswith("Pi05Policy")
    assert world_model.latent_source.startswith("RLinf-aligned pi0.5")
    assert world_model.token_count == 768
    assert world_model.token_dim == 2048
    assert world_model.action_dim == 7
    assert world_model.proprio_dim == 8
    assert world_model.vjepa2_predictor_dim == 1024
    assert world_model.vjepa2_depth == 24
    assert world_model.vjepa2_num_heads == 16
    assert world_model.vjepa2_spatial_grid is None
    assert world_model.vjepa2_truncate_rollout_gradients is True
    assert worker.transition_type == "vjepa2_ac"
    assert worker.transition_init == "random"
    assert worker.vjepa2_predictor_dim == world_model.vjepa2_predictor_dim
    assert worker.vjepa2_truncate_rollout_gradients == world_model.vjepa2_truncate_rollout_gradients
