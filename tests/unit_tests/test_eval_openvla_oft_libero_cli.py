from __future__ import annotations

import pytest

from dreamervla.diagnostics.eval_openvla_oft_libero import (
    parse_suite_name,
    resolve_num_images_for_camera_inputs,
)


def test_parse_suite_name_accepts_libero_10_aliases() -> None:
    assert parse_suite_name("libero_10") == "libero_10"
    assert parse_suite_name("libero10") == "libero_10"
    assert parse_suite_name("libero_long") == "libero_10"


def test_camera_inputs_selects_effective_num_images() -> None:
    assert resolve_num_images_for_camera_inputs("primary", None) == 1
    assert resolve_num_images_for_camera_inputs("primary+wrist", None) == 2
    assert resolve_num_images_for_camera_inputs(None, 2) == 2


def test_camera_inputs_rejects_conflicting_num_images() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        resolve_num_images_for_camera_inputs("primary", 2)
