"""Q2: removing the redundant `.clone()` from CPU-state-dict helpers must keep
each captured tensor an INDEPENDENT copy.

On CUDA the trailing `.clone()` after `.cpu()` is redundant (D2H already copies),
but on CPU `.cpu()` is a no-op alias of the live param, so the helpers must still
return a tensor that does NOT change when the source is mutated in place (the §4
weight-sync constraint). These tests prove independence on the CPU branch — the
exact branch that still needs a copy — without needing a GPU.
"""

from __future__ import annotations

import torch

from dreamervla.hybrid_engines.weight_syncer.compression import DTypeTensorCompressor
from dreamervla.hybrid_engines.weight_syncer.objectstore import _to_cpu_tensor
from dreamervla.workers.actor.learner_worker import _cpu_state_dict as _learner_cpu_state_dict
from dreamervla.workers.inference.inference_worker import (
    _cpu_state_dict as _inference_cpu_state_dict,
)


class _TwoParamModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.b = torch.nn.Parameter(torch.tensor([4.0, 5.0]))


def test_to_cpu_tensor_tensor_branch_is_independent() -> None:
    src = torch.tensor([1.0, 2.0, 3.0])
    out = _to_cpu_tensor(src)
    src.add_(100.0)  # simulate an optimizer step mutating the live param in place
    assert torch.equal(out, torch.tensor([1.0, 2.0, 3.0]))
    assert out.data_ptr() != src.data_ptr()


def test_to_cpu_tensor_non_tensor_branch_returns_tensor() -> None:
    out = _to_cpu_tensor([1.0, 2.0, 3.0])
    assert isinstance(out, torch.Tensor)
    assert torch.equal(out, torch.tensor([1.0, 2.0, 3.0]))


def test_compress_state_dict_payload_is_independent() -> None:
    src = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    state = {"w": src}
    packed = DTypeTensorCompressor(transport_dtype="fp16").compress_state_dict(state)
    src.add_(100.0)
    restored = packed["w"].tensor.to(dtype=packed["w"].original_dtype)
    assert torch.equal(restored, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))


def test_learner_cpu_state_dict_is_independent_of_live_params() -> None:
    module = _TwoParamModule()
    captured = _learner_cpu_state_dict(module)
    with torch.no_grad():
        module.w.add_(100.0)
        module.b.add_(100.0)
    assert torch.equal(captured["w"], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(captured["b"], torch.tensor([4.0, 5.0]))


def test_inference_cpu_state_dict_is_independent_of_live_params() -> None:
    module = _TwoParamModule()
    captured = _inference_cpu_state_dict(module)
    with torch.no_grad():
        module.w.add_(100.0)
        module.b.add_(100.0)
    assert torch.equal(captured["w"], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(captured["b"], torch.tensor([4.0, 5.0]))
