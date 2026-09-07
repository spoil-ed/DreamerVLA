"""Opt-in real-checkpoint BF16 regression for the PI0.5 residual transition."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_VJEPA2_AC_SMOKE") != "1",
    reason="Set RUN_VJEPA2_AC_SMOKE=1 and VJEPA2_AC_CKPT for the GPU/checkpoint smoke",
)


def test_pretrained_residual_bf16_gradient_and_reload(tmp_path: Path) -> None:
    """Check real loading, first-step gradients, causality, updates, and reload."""
    assert torch.cuda.is_available(), "This opt-in smoke requires CUDA"
    assert torch.cuda.is_bf16_supported(), "This smoke requires BF16 support"
    checkpoint = Path(os.environ["VJEPA2_AC_CKPT"])
    assert checkpoint.is_file(), f"Missing pretrained checkpoint: {checkpoint}"
    torch.manual_seed(13)
    torch.set_num_threads(4)
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "configs"), version_base=None
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=wm_pi05_collected_train",
                "task=pi05_libero_object",
                "world_model.transition_type=vjepa2_ac",
                "world_model.transition_init=pretrained",
            ],
        )
    wm = instantiate(cfg.world_model).cuda()
    transition = wm.vjepa2_transition
    assert transition is not None
    transition.use_activation_checkpointing = True
    report = transition.pretrained_load_report
    assert report is not None and report.pretrained_parameter_ratio > 0.95
    assert all(
        f"predictor_blocks.{layer}.attn.qkv.weight" in report.loaded_keys for layer in range(24)
    )
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    source = {
        transition._canonical_checkpoint_key(key): value
        for key, value in transition._unwrap_predictor_state(payload).items()
    }
    for key in report.loaded_keys:
        torch.testing.assert_close(
            transition.state_dict()[key].cpu(),
            source[transition._source_key_for_model_key(key)],
            rtol=0,
            atol=0,
        )
    del payload, source
    initial_std = transition.output_adapter.weight.std().item()
    assert initial_std == pytest.approx(cfg.world_model.vjepa2_residual_output_init_std, rel=0.01)
    assert "output_adapter.weight" not in report.loaded_keys

    tokens = torch.randn(1, 3, wm.token_count, wm.token_dim, device="cuda")
    actions = torch.randn(1, 3, wm.action_dim, device="cuda")
    states = torch.randn(1, 3, wm.proprio_dim, device="cuda")
    mask = torch.ones(tokens.shape[:3], device="cuda", dtype=torch.bool)
    mask[:, :, 512:] = False

    def predict(action: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return transition(tokens, action, states, token_mask=mask)[..., : wm.token_dim]

    transition.eval()
    with torch.no_grad():
        initial = predict(actions)
        assert initial.shape == tokens.shape
        delta_rms = (initial[mask] - tokens[mask]).square().mean().sqrt().item()
        assert 0.0 < delta_rms < 0.1
        changed = actions.clone()
        changed[:, -1] += 1.0
        counterfactual = predict(changed)
        torch.testing.assert_close(initial[:, :-1], counterfactual[:, :-1], rtol=0, atol=0)
        action_response = (initial[:, -1] - counterfactual[:, -1]).abs().max().item()
        assert action_response > 0.0, "Current-frame action must affect the residual"

    # A controlled synthetic target checks numerical optimization only. It is
    # deliberately not evidence of accuracy or motion on real PI0.5 episodes.
    target = tokens + 0.05 * actions[..., :1, None]
    optimizer = torch.optim.AdamW(transition.parameters(), lr=1.0e-5, weight_decay=0.0)
    transition.train()
    losses = []
    gradient_norms = {}
    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        prediction = predict(actions)
        loss = (prediction[mask] - target[mask]).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        if step == 0:
            for name, parameter in (
                ("qkv", transition.predictor_blocks[0].attn.qkv.weight),
                ("input_adapter", transition.input_adapter.weight),
                ("action_encoder", transition.action_encoder.weight),
            ):
                assert parameter.grad is not None
                norm = parameter.grad.norm().item()
                assert 0.0 < norm < float("inf"), f"Invalid first-update gradient for {name}"
                gradient_norms[name] = norm
        torch.nn.utils.clip_grad_norm_(transition.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], "The fixed synthetic target should be learnable"
    transition.eval()
    with torch.no_grad():
        expected = predict(actions)
    saved = tmp_path / "world_model.pt"
    torch.save({"world_model": wm.state_dict()}, saved)
    with torch.no_grad():
        transition.output_adapter.weight.add_(1.0)
    payload = torch.load(saved, weights_only=True, map_location="cpu", mmap=True)
    wm.load_state_dict(payload["world_model"], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(predict(actions), expected, rtol=0, atol=0)
    print(
        json.dumps(
            {
                "loaded_parameters": report.loaded_parameters,
                "pretrained_parameter_ratio": report.pretrained_parameter_ratio,
                "output_init_std": initial_std,
                "initial_delta_rms": delta_rms,
                "current_action_max_response": action_response,
                "first_update_gradient_norms": gradient_norms,
                "synthetic_losses": losses,
            },
            indent=2,
        )
    )
