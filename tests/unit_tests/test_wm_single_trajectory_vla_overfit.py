from __future__ import annotations

from pathlib import Path

from dreamervla.diagnostics import wm_single_trajectory_vla_overfit as diag


def test_vla_args_use_runtime_input_token_mode() -> None:
    args = diag._vla_args(Path("/tmp/vla"), "agentview_rgb")

    assert args.policy_mode == "discrete"
    assert args.include_state is False
    assert args.num_images_in_input == 1
    assert args.image_keys == ("agentview_rgb",)


def test_vla_overfit_launcher_is_registered() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/experiments/wm_single_trajectory_vla_overfit.sh"
    assert script.is_file()
    assert "wm_single_trajectory_vla_overfit" in script.read_text(encoding="utf-8")
