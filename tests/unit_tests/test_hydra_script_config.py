from __future__ import annotations

import pytest

from dreamervla.utils.hydra_config import (
    SCRIPT_CONFIG_DIR,
    script_config,
    script_namespace,
)


@pytest.mark.parametrize(
    "config_name",
    [path.stem for path in sorted(SCRIPT_CONFIG_DIR.glob("*.yaml"))],
)
def test_all_script_configs_compose(config_name: str) -> None:
    assert isinstance(script_config(config_name), dict)


def test_script_config_composes_real_hydra_yaml() -> None:
    cfg = script_config(
        "preprocess_remaining_steps_reward",
        [
            "input_dir=/tmp/in",
            "output_dir=/tmp/out",
            "overwrite=true",
        ],
    )

    assert cfg["input_dir"] == "/tmp/in"
    assert cfg["output_dir"] == "/tmp/out"
    assert cfg["overwrite"] is True


def test_script_config_rejects_legacy_flags() -> None:
    with pytest.raises(SystemExit, match="Use Hydra override syntax"):
        script_config("preprocess_remaining_steps_reward", ["--input-dir", "/tmp/in"])


def test_script_namespace_preserves_config_values() -> None:
    args = script_namespace(
        "preprocess_rynn_pixel_hidden",
        [
            "image_keys=[agentview_rgb,eye_in_hand_rgb]",
            "history=2",
        ],
    )

    assert args.image_keys == ["agentview_rgb", "eye_in_hand_rgb"]
    assert args.history == 2
