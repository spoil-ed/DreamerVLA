from __future__ import annotations


def test_online_cotrain_runner_has_extracted_methods():
    from dreamervla.runners.online_cotrain_runner import OnlineCotrainRunner

    assert hasattr(OnlineCotrainRunner, "_build_components")
    assert hasattr(OnlineCotrainRunner, "_online_cotrain_loop")
