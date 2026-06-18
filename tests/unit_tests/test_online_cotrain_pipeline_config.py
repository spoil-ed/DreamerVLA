from __future__ import annotations


def test_validate_cfg_warmup(tmp_path):
    import pytest
    from omegaconf import OmegaConf

    from dreamervla.config import validate_cfg
    base = OmegaConf.create({
        "_target_": "dreamervla.runners.OnlineCotrainPipelineRunner",
        "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
        "training": {"wm_warmup_steps": 10, "classifier_warmup_steps": 10},
    })
    validate_cfg(base)  # existing dir -> ok
    bad = OmegaConf.create({
        "_target_": "dreamervla.runners.OnlineCotrainPipelineRunner",
        "offline_warmup": {"data_dir": str(tmp_path / "nope"), "hidden_dir": str(tmp_path)},
        "training": {"wm_warmup_steps": 10, "classifier_warmup_steps": 10},
    })
    with pytest.raises(Exception, match="offline_warmup.data_dir"):
        validate_cfg(bad)
    neg = OmegaConf.create({
        "_target_": "dreamervla.runners.OnlineCotrainPipelineRunner",
        "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
        "training": {"wm_warmup_steps": -1, "classifier_warmup_steps": 10},
    })
    with pytest.raises(Exception, match="wm_warmup_steps"):
        validate_cfg(neg)
