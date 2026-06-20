import torch

from dreamervla.utils.hf_module import load_module_pretrained, save_module_pretrained


def test_save_load_roundtrip(tmp_path):
    m = torch.nn.Linear(4, 3)
    with torch.no_grad():
        m.weight.fill_(0.5)
        m.bias.fill_(-0.25)
    d = tmp_path / "wm"
    save_module_pretrained(
        m, str(d), target="torch.nn.Linear", init_args={"in_features": 4, "out_features": 3}
    )
    assert (d / "config.json").is_file()
    assert (d / "model.safetensors").is_file()
    loaded = load_module_pretrained(str(d))
    assert isinstance(loaded, torch.nn.Linear)
    for k, v in m.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v)
