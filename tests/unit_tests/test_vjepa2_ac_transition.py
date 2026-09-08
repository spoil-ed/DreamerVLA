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


def _raw_state_wm(**overrides) -> ChunkAwareWorldModel:
    config = dict(
        model_dim=10,
        proprio_dim=3,
        proprio_emb_dim=4,
        vjepa2_proprio_representation="raw_padded",
        vjepa2_proprio_loss_scale=1.0,
        vjepa2_residual_prediction=True,
        vjepa2_rollout_bptt_steps=2,
    )
    config.update(overrides)
    return _tiny_chunk_wm(**config)


def test_decoded_auxiliary_receives_all_chunks_and_preserves_rollout_interface() -> None:
    class ReadoutLoss(torch.nn.Module):
        def forward(self, prediction, target, anchor, **kwargs):
            self.seen = (prediction, target, anchor)
            return {"_loss": (prediction - target).square().mean()}

    auxiliary = ReadoutLoss()
    wm = _raw_state_wm(
        chunk_rollout_chunks=2,
        chunk_rollout_loss_scale=0.2,
        grad_checkpoint=True,
        vjepa2_truncate_rollout_gradients=False,
        decoded_visual_loss=auxiliary,
    )
    batch = dict(
        obs_embedding=torch.randn(1, 6, 2, 4),
        actions=torch.randn(1, 6, 2),
        proprio=torch.randn(1, 6, 3),
    )
    result = wm.chunk_loss(batch)
    prediction, target, anchor = auxiliary.seen
    assert prediction.shape == target.shape == (1, 4, 2, 4)
    assert prediction.requires_grad and not target.requires_grad and not anchor.requires_grad
    torch.testing.assert_close(
        anchor, wm._normalize_raw_vision_tokens(batch["obs_embedding"])[:, 1:2]
    )
    result["_loss"].backward()
    assert wm.vjepa2_transition.output_adapter.weight.grad.norm() > 0
    with pytest.raises(ValueError, match="full closed-loop"):
        _raw_state_wm(decoded_visual_loss=ReadoutLoss())


def test_raw_state_codec_preserves_absolute_values_and_ignores_padding() -> None:
    wm = _raw_state_wm()
    raw = torch.tensor([[[10.0, -7.0, 0.4], [21.0, -5.0, 0.8]]])
    obs = wm._observation_tokens(torch.randn(1, 2, 2, 4), raw)
    torch.testing.assert_close(wm._raw_proprio_from_obs_tokens(obs), raw, rtol=0, atol=0)
    mask = torch.tensor([[[True, False], [True, False]]])
    obs[:, :, 1] = 1000.0
    torch.testing.assert_close(wm._raw_proprio_from_obs_tokens(obs, mask), raw, rtol=0, atol=0)
    assert not list(wm.proprio_encoder.parameters())
    assert not list(wm.proprio_decoder.parameters())


@pytest.mark.parametrize("checkpointed", [False, True])
def test_raw_state_history_survives_steps_chunks_and_checkpointing(checkpointed: bool) -> None:
    torch.manual_seed(47)
    wm = _raw_state_wm(
        grad_checkpoint=checkpointed, chunk_rollout_chunks=2, chunk_rollout_loss_scale=0.2
    )
    raw = torch.randn(1, 6, 3)
    observed_states = []

    def capture(_module, args):
        observed_states.append(args[2].detach().clone())

    handle = wm.vjepa2_transition.register_forward_pre_hook(capture)
    batch = {
        "obs_embedding": torch.randn(1, 6, 2, 4),
        "actions": torch.randn(1, 6, 2),
        "proprio": raw,
        "prefix_attention_mask": torch.tensor([[[True, False]] * 6]),
    }
    output = wm.chunk_loss(batch)
    torch.testing.assert_close(observed_states[0], raw[:, :2])
    torch.testing.assert_close(observed_states[1][:, 0], raw[:, 1])
    # At the next chunk, history must retain the previous predicted state,
    # not broadcast its latest value over the whole history.
    torch.testing.assert_close(observed_states[2][:, 0], observed_states[1][:, 1])
    assert "rollout_proprio_reconstruction_loss" in output
    output["_loss"].backward()
    grad = wm.vjepa2_transition.state_output_adapter.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    handle.remove()


@pytest.mark.parametrize("segment_steps", [1, 2])
def test_bptt_segment_controls_credit_into_previous_prediction(segment_steps: int) -> None:
    torch.manual_seed(51)
    wm = _raw_state_wm(vjepa2_rollout_bptt_steps=segment_steps)
    outputs = []

    def capture(_module, _args, output):
        output.retain_grad()
        outputs.append(output)

    handle = wm.vjepa2_transition.register_forward_hook(capture)
    state = wm.initial_imagination_state(torch.randn(1, 2, 2, 4), proprio=torch.randn(1, 2, 3))
    out = wm.predict_next_chunk(state, torch.randn(1, 2, 2))
    out["proprio_seq"][:, -1].square().mean().backward()
    if segment_steps == 1:
        assert outputs[0].grad is None or outputs[0].grad.norm() == 0
    else:
        assert outputs[0].grad is not None and outputs[0].grad.norm() > 0
    handle.remove()


def test_raw_state_requires_direct_supervision_and_sufficient_slots() -> None:
    with pytest.raises(ValueError, match="requires vjepa2_proprio_loss_scale"):
        _raw_state_wm(vjepa2_proprio_loss_scale=0)
    with pytest.raises(ValueError, match="slots must fit"):
        _raw_state_wm(proprio_emb_dim=2, model_dim=8)


def test_raw_state_history_can_be_recovered_from_masked_observation_slots() -> None:
    wm = _raw_state_wm().eval()
    raw = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    mask = torch.tensor([[[True, False], [True, False]]])
    state = wm.initial_imagination_state(
        torch.randn(1, 2, 2, 4), proprio=raw, prefix_attention_mask=mask
    )
    old_state = dict(state)
    old_state.pop("proprio_history")
    old_state["history"] = old_state["history"].masked_fill(~mask[..., None], 0)
    action = torch.randn(1, 2)
    with torch.no_grad():
        expected = wm.predict_next(state, action)
        actual = wm.predict_next(old_state, action)
    torch.testing.assert_close(actual["proprio_history"], expected["proprio_history"])
    torch.testing.assert_close(actual["hidden"], expected["hidden"])


def test_random_ac_uses_backbone_and_adapter_groups_for_matched_init_comparison() -> None:
    wm = _raw_state_wm()
    groups = {
        g["group_name"]: {id(p) for p in g["params"]} for g in wm.optimizer_parameter_groups()
    }
    assert (
        id(wm.vjepa2_transition.predictor_blocks[0].attn.qkv.weight)
        in groups["pretrained_backbone"]
    )
    assert id(wm.vjepa2_transition.action_encoder.weight) in groups["pretrained_backbone"]
    assert id(wm.vjepa2_transition.state_output_adapter.weight) in groups["adapter"]
    assert id(wm.vjepa2_transition.output_adapter.weight) in groups["adapter"]


def test_temporal_supervision_penalizes_static_predictions_with_valid_gradients(
    monkeypatch,
) -> None:
    wm = _tiny_chunk_wm(
        token_normalization="none",
        hidden_loss_scale=0.0,
        vjepa2_temporal_difference_loss_scale=1.0,
    )
    obs = torch.arange(4.0).view(1, 4, 1, 1).expand(1, 4, 2, 4).clone()
    obs[:, :, 1] *= 1000
    prediction = torch.nn.Parameter(torch.ones(1, 2, 2, 4))
    monkeypatch.setattr(wm, "predict_next_chunk", lambda *_: {"hidden_seq": prediction})
    output = wm.chunk_loss(
        dict(
            obs_embedding=obs,
            actions=torch.zeros(1, 4, 2),
            prefix_attention_mask=torch.tensor([[[True, False]] * 4]),
        )
    )
    assert output["temporal_difference_loss"].item() == pytest.approx(1.0)
    output["_loss"].backward()
    assert prediction.grad is not None and prediction.grad.norm() > 0
    assert prediction.grad[:, :, 1].count_nonzero() == 0


def test_training_and_both_evaluation_paths_agree_with_raw_state_and_mask() -> None:
    from dreamervla.diagnostics.compare_wm_libero_rollout import _rollout_closed_loop
    from dreamervla.runtime.cotrain_eval import (
        EncodedEvalTrajectory,
        closed_loop_world_model_trajectory,
    )

    torch.manual_seed(61)
    wm = _raw_state_wm(chunk_rollout_chunks=2, chunk_rollout_loss_scale=0.2).eval()
    obs = torch.randn(1, 6, 2, 4) * 4 + 3
    raw = torch.randn(1, 6, 3)
    actions = torch.randn(1, 6, 2)
    mask = torch.tensor([[[True, False]] * 6])
    seen = []
    handle = wm.vjepa2_transition.register_forward_pre_hook(
        lambda _m, args: seen.append(args[2].clone())
    )
    with torch.no_grad():
        expected, _ = _rollout_closed_loop(
            wm, obs[0], actions[0], 2, proprio=raw[0], attention_mask=mask[0]
        )
        diagnostic_states = list(seen)
        seen.clear()
        wm.chunk_loss(
            dict(obs_embedding=obs, actions=actions, proprio=raw, prefix_attention_mask=mask)
        )
        for a, b in zip(diagnostic_states, seen, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        result = closed_loop_world_model_trajectory(
            wm,
            EncodedEvalTrajectory(
                task_id=0,
                success=True,
                hidden=obs[0],
                actions=actions[0],
                proprio=raw[0],
                prefix_attention_mask=mask[0],
            ),
        )
    handle.remove()
    torch.testing.assert_close(result.predicted_hidden, expected[..., :4], rtol=0, atol=0)
    torch.testing.assert_close(
        result.predicted_proprio,
        wm._raw_proprio_from_obs_tokens(expected, mask[0, 2:]),
        rtol=0,
        atol=0,
    )


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


def test_grouped_spatial_rope_masks_padded_image_slots() -> None:
    torch.manual_seed(9)
    transition = _tiny_transition(
        token_count=6,
        spatial_grid=(1, 2),
        spatial_group_count=3,
    ).eval()
    tokens = torch.randn(1, 2, 6, 5)
    actions = torch.randn(1, 2, 2)
    states = torch.randn(1, 2, 3)
    token_mask = torch.tensor([[[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]]], dtype=torch.bool)

    reference = transition(tokens, actions, states, token_mask=token_mask)
    changed = tokens.clone()
    changed[:, :, 4:].add_(1000.0)
    actual = transition(changed, actions, states, token_mask=token_mask)

    assert transition.spatial_group_embedding is not None
    torch.testing.assert_close(reference, actual)
    assert torch.count_nonzero(actual[:, :, 4:]).item() == 0


def test_residual_output_adapter_starts_near_persistence_with_gradient_flow() -> None:
    torch.manual_seed(13)
    transition = _tiny_transition(output_dim=5, residual_prediction=True).eval()
    tokens = torch.randn(1, 2, 2, 5)
    output = transition(
        tokens,
        torch.randn(1, 2, 2),
        torch.randn(1, 2, 3),
    )

    delta = output - tokens
    assert torch.count_nonzero(delta).item() > 0
    assert 0.0 < delta.square().mean().sqrt().item() < 0.1
    output.square().mean().backward()
    assert transition.predictor_blocks[0].attn.qkv.weight.grad is not None
    assert transition.predictor_blocks[0].attn.qkv.weight.grad.norm().item() > 0.0
    assert transition.action_encoder.weight.grad is not None
    assert transition.action_encoder.weight.grad.norm().item() > 0.0
    torch.testing.assert_close(
        transition.action_input_adapter.weight,
        torch.eye(transition.action_dim),
    )


def test_unit_layer_scales_preserve_block_function_and_nonzero_scales_train() -> None:
    torch.manual_seed(71)
    original = _tiny_transition()
    scaled = _tiny_transition(layer_scale_init=1.0)
    result = scaled.load_state_dict(original.state_dict(), strict=False)
    assert len(result.missing_keys) == 4
    assert all(key.endswith("_layer_scale") for key in result.missing_keys)
    assert not result.unexpected_keys
    args = (torch.randn(1, 2, 2, 5), torch.randn(1, 2, 2), torch.randn(1, 2, 3))
    torch.testing.assert_close(scaled(*args), original(*args), rtol=0, atol=0)
    for block in scaled.predictor_blocks:
        with torch.no_grad():
            block.attn_layer_scale.fill_(0.01)
            block.mlp_layer_scale.fill_(0.01)
    scaled(*args).square().mean().backward()
    for block in scaled.predictor_blocks:
        assert block.mlp.fc1.weight.grad.norm() > 0
        assert block.attn.qkv.weight.grad.norm() > 0
        assert block.mlp_layer_scale.grad.norm() > 0
        assert block.attn_layer_scale.grad.norm() > 0


def test_random_ac_layer_scales_belong_to_adapter_group() -> None:
    wm = _raw_state_wm(vjepa2_layer_scale_init=0.01)
    groups = {
        g["group_name"]: {id(p) for p in g["params"]} for g in wm.optimizer_parameter_groups()
    }
    for block in wm.vjepa2_transition.predictor_blocks:
        assert id(block.attn_layer_scale) in groups["adapter"]
        assert id(block.mlp_layer_scale) in groups["adapter"]


def test_masked_hidden_loss_ignores_padding_tokens() -> None:
    model = _tiny_chunk_wm()
    target = torch.randn(1, 2, 2, 4)
    prediction = target.clone()
    prediction[:, :, 1].add_(1000.0)
    mask = torch.tensor([[[True, False], [True, False]]])

    _loss, mse, cosine = model._hidden_loss_terms(prediction, target, token_mask=mask)

    assert mse.item() == 0.0
    assert cosine.item() == pytest.approx(0.0, abs=1e-7)


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
        "input_adapter.bias <- predictor_embed.bias" in key for key in report.mismatched_keys
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
    assert list(world_model.vjepa2_spatial_grid) == [16, 16]
    assert world_model.vjepa2_spatial_group_count == 3
    assert world_model.vjepa2_residual_prediction is True
    assert world_model.vjepa2_residual_output_init_std == 1.0e-3
    assert config.offline_warmup.online_latent.encoder_method.endswith("bundle_batch")
    assert world_model.vjepa2_truncate_rollout_gradients is True
    assert worker.transition_type == "vjepa2_ac"
    assert worker.transition_init == "random"
    assert worker.vjepa2_predictor_dim == world_model.vjepa2_predictor_dim
    assert worker.vjepa2_residual_output_init_std == world_model.vjepa2_residual_output_init_std
    assert worker.vjepa2_truncate_rollout_gradients == world_model.vjepa2_truncate_rollout_gradients
    assert config.optim.world_model.lr_scheduler == "cosine"
    assert config.optim.world_model.adapter_alignment_steps == 500
    assert config.optim.world_model.parameter_group_lrs.adapter == 1.0e-4
    assert config.optim.world_model.parameter_group_lrs.pretrained_backbone == 1.0e-5
