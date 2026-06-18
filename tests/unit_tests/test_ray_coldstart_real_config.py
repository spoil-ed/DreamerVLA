from __future__ import annotations


def test_oft_collect_common_exposes_shared_helpers() -> None:
    from dreamervla.runners.oft_collect_common import (
        assert_policy_mode_matches,
        load_policy,
        make_preprocess_config,
        resolve_num_images_in_input,
    )

    for fn in (
        load_policy,
        make_preprocess_config,
        assert_policy_mode_matches,
        resolve_num_images_in_input,
    ):
        assert callable(fn)
