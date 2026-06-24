"""Bridge: the ray async runner must load pipeline warmup ckpts by component.

The pipeline writes per-component warmup files (wm_warmup.ckpt -> {"world_model": sd},
classifier_warmup.ckpt -> {"classifier": sd, "classifier_threshold": ...}). The ray
async runner's component-mapping init_ckpt extracts each via _load_component_state_dict.
"""

from __future__ import annotations

import torch

from dreamervla.runners.online_cotrain_ray_runner import _load_component_state_dict


def _save(tmp_path, name, payload):
    p = tmp_path / name
    torch.save(payload, p)
    return str(p)


def test_extracts_world_model_from_pipeline_wm_warmup(tmp_path):
    wm_sd = {"layer.weight": torch.zeros(2, 2)}
    path = _save(tmp_path, "wm_warmup.ckpt", {"global_step": 7, "world_model": wm_sd})
    got = _load_component_state_dict(path, "world_model")
    # The actual state_dict, NOT the wrapper dict (no "global_step" key leaks in).
    assert set(got) == {"layer.weight"}


def test_extracts_classifier_from_pipeline_cls_warmup(tmp_path):
    cls_sd = {"head.bias": torch.ones(3)}
    path = _save(
        tmp_path, "classifier_warmup.ckpt", {"classifier": cls_sd, "classifier_threshold": 0.5}
    )
    got = _load_component_state_dict(path, "classifier")
    assert set(got) == {"head.bias"}


def test_runner_format_state_dicts_still_works(tmp_path):
    wm_sd = {"w": torch.zeros(1)}
    path = _save(tmp_path, "unified.ckpt", {"state_dicts": {"world_model": wm_sd}})
    got = _load_component_state_dict(path, "world_model")
    assert set(got) == {"w"}
