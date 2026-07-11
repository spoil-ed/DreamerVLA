"""RNG state capture/restore for bit-exact runner resume."""

import random

import torch
from omegaconf import OmegaConf

from dreamervla.runners._dreamer_runner_common import DreamerCkptResumeMixin


class _MiniDreamer(DreamerCkptResumeMixin):
    """Smallest mixin host exercising _save_ckpt / _maybe_resume."""

    def __init__(self, tmp, cfg):
        self.is_main_process = True
        self.global_step = 5
        self.epoch = 2
        self.cfg = cfg
        self.ckpt_dir = tmp


def test_restore_rng_state_reproduces_torch_and_python_draws():
    from dreamervla.utils.seed import capture_rng_state, restore_rng_state

    state = capture_rng_state()
    ref = [torch.rand(()).item() for _ in range(4)] + [random.random() for _ in range(4)]

    # Perturb both generators so a no-op restore would change the draws.
    torch.manual_seed(123456)
    random.seed(123456)

    restore_rng_state(state)
    got = [torch.rand(()).item() for _ in range(4)] + [random.random() for _ in range(4)]

    assert got == ref


def test_restore_rng_state_tolerates_missing_or_none_payload():
    from dreamervla.utils.seed import restore_rng_state

    # Backward compatibility: old checkpoints have no "rng" key.
    restore_rng_state(None)
    restore_rng_state({})  # partial / empty payloads must not raise


def test_dreamerv3_save_ckpt_routes_rng_through_shared_helper(tmp_path):
    # Unifying DreamerV3 onto the shared helper means its payload now also
    # carries python `random` state (previously torch+cuda only).
    runner = _MiniDreamer(tmp_path, OmegaConf.create({}))
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters())
    path = tmp_path / "latest.ckpt"

    runner._save_ckpt(model, opt, path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "python" in payload["rng"]


def test_dreamerv3_resume_restores_python_rng_bit_exact(tmp_path):
    cfg = OmegaConf.create({"training": {"resume": True}})
    path = tmp_path / "latest.ckpt"

    torch.manual_seed(11)
    random.seed(11)
    _MiniDreamer(tmp_path, cfg)._save_ckpt(
        torch.nn.Linear(3, 2), torch.optim.Adam(torch.nn.Linear(3, 2).parameters()), path
    )
    ref = [torch.rand(()).item() for _ in range(3)] + [random.random() for _ in range(3)]

    torch.manual_seed(999)
    random.seed(999)
    resumed = _MiniDreamer(tmp_path, cfg)._maybe_resume(
        torch.nn.Linear(3, 2), torch.optim.Adam(torch.nn.Linear(3, 2).parameters())
    )
    got = [torch.rand(()).item() for _ in range(3)] + [random.random() for _ in range(3)]

    assert resumed is True
    assert got == ref
