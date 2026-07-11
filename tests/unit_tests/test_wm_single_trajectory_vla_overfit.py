from __future__ import annotations

from pathlib import Path

from dreamervla.diagnostics import wm_single_trajectory_vla_overfit as diag


def test_vla_args_use_runtime_hidden_token_mode() -> None:
    args = diag._vla_args(Path("/tmp/vla"), "agentview_rgb")

    assert args.policy_mode == "discrete"
    assert args.include_state is False
    assert args.num_images_in_input == 1
    assert args.image_keys == ("agentview_rgb",)
