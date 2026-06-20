# tests/unit_tests/test_base_runner_config_gate.py
import json
import types

from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner


def _fake(cfg, manifest_path):
    """Minimal object exposing the two methods under test without full init."""
    obj = types.SimpleNamespace()
    obj.cfg = cfg
    obj.config = cfg
    obj.is_main_process = True
    obj.print_config = types.MethodType(BaseRunner.print_config, obj)
    obj.append_model_summary = types.MethodType(BaseRunner.append_model_summary, obj)
    obj.get_run_manifest_path = lambda: manifest_path
    return obj


def test_print_config_suppressed_by_default(capsys):
    cfg = OmegaConf.create({"a": 1})
    _fake(cfg, None).print_config()
    assert capsys.readouterr().out == ""


def test_print_config_emitted_when_enabled(capsys):
    cfg = OmegaConf.create({"a": 1, "training": {"print_config": True}})
    _fake(cfg, None).print_config()
    assert "'a': 1" in capsys.readouterr().out


def test_append_model_summary_updates_manifest(tmp_path):
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps({"schema_version": 1}) + "\n", encoding="utf-8")
    cfg = OmegaConf.create({})
    _fake(cfg, path).append_model_summary({"total_trainable": 12_300_000})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["model"]["total_trainable"] == 12_300_000
