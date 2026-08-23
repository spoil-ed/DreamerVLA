from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from dreamervla.diagnostics.openvla_oft_obs_action_policy import (
    OpenVLAOFTObsActionPolicy,
    resolve_unnorm_key,
    set_runtime_env,
)


def test_resolve_unnorm_key_prefers_no_noops_suffix() -> None:
    model = SimpleNamespace(norm_stats={"libero_spatial_no_noops": {}})

    assert resolve_unnorm_key(model, "libero_spatial", "") == "libero_spatial_no_noops"


def test_obs_action_policy_returns_action_chunk_from_backend() -> None:
    calls = []

    def fake_backend(**kwargs):
        calls.append(kwargs)
        return [np.ones(7), np.zeros(7)]

    policy = OpenVLAOFTObsActionPolicy.from_backend(
        cfg=SimpleNamespace(
            task_suite_name="libero_spatial", num_images_in_input=1, use_proprio=False
        ),
        model=object(),
        processor=object(),
        action_backend=fake_backend,
    )
    obs = {
        "full_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.ones((224, 224, 3), dtype=np.uint8),
        "state": np.ones(8, dtype=np.float32),
    }

    actions = policy(obs, "pick up the bowl")

    assert len(actions) == 2
    assert set(calls[0]["obs"]) == {"full_image"}
    assert calls[0]["obs"]["full_image"] is obs["full_image"]
    assert calls[0]["task_label"] == "pick up the bowl"


@pytest.mark.parametrize(
    "num_images_in_input,use_proprio",
    [(2, False), (1, True)],
)
def test_obs_action_policy_rejects_removed_multiview_and_proprio_routes(
    num_images_in_input: int,
    use_proprio: bool,
) -> None:
    with pytest.raises(ValueError, match="mainline"):
        OpenVLAOFTObsActionPolicy.from_backend(
            cfg=SimpleNamespace(
                task_suite_name="libero_spatial",
                num_images_in_input=num_images_in_input,
                use_proprio=use_proprio,
            ),
            model=object(),
            processor=object(),
            action_backend=lambda **_: [np.ones(7)],
        )


def test_runtime_env_defaults_to_osmesa_without_overriding_existing_values(monkeypatch) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)

    set_runtime_env()

    assert os.environ["MUJOCO_GL"] == "osmesa"
    assert os.environ["PYOPENGL_PLATFORM"] == "osmesa"

    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")

    set_runtime_env()

    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["PYOPENGL_PLATFORM"] == "egl"
