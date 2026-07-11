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


def test_script_config_rejects_compat_flags() -> None:
    with pytest.raises(SystemExit, match="Use Hydra override syntax"):
        script_config("preprocess_remaining_steps_reward", ["--input-dir", "/tmp/in"])


def test_script_namespace_preserves_config_values() -> None:
    args = script_namespace(
        "preprocess_oft_input_tokens",
        [
            "max_files=3",
            "output_dtype=float32",
        ],
    )

    assert args.max_files == 3
    assert args.output_dtype == "float32"


def test_oft_input_token_preprocess_has_one_public_output() -> None:
    cfg = script_config("preprocess_oft_input_tokens")

    assert cfg["out_input_token_dir"].endswith(
        "_oft_input_token_embedding_vla_policy_h1"
    )
    assert cfg["obs_hidden_source"] == "input_token_embedding"
    assert cfg["image_keys"] == ["agentview_rgb"]
    assert cfg["history"] == 1
    assert cfg["patches_per_image"] == 256
    for removed in (
        "out_c_dir",
        "out_d_dir",
        "out_hidden_token_flat_dir",
        "out_hidden_token_dir",
        "skip_cd_sidecars",
        "save_hidden_token",
    ):
        assert removed not in cfg
