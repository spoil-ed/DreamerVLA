"""CPU equivalence test for the H5 switchable SDPA attention path.

The WM chunk attention (`_WMStyleAttention`) gains an ``attn_impl`` flag
defaulting to ``"manual"`` (the existing hand-rolled QK^T softmax) with an opt-in
``"sdpa"`` path through ``F.scaled_dot_product_attention``.  With dropout disabled
and the modules in eval mode the two paths must agree within float tolerance
(SDPA uses a different reduction order, so they are close-but-not-bit-identical),
and the ``"manual"`` path must stay byte-for-byte the pre-H5 code.
"""

from __future__ import annotations

import torch

from dreamervla.models.world_model.wm_chunk import _WMStyleAttention

# Float32 CPU tolerance: SDPA's fused reduction differs from the manual matmul.
_ATOL = 1e-5
_RTOL = 1e-4


def _make_pair(dim: int, heads: int, dim_head: int):
    """Return (manual, sdpa) attention modules sharing identical weights."""
    torch.manual_seed(0)
    manual = _WMStyleAttention(dim, heads=heads, dim_head=dim_head, dropout=0.0)
    sdpa = _WMStyleAttention(
        dim, heads=heads, dim_head=dim_head, dropout=0.0, attn_impl="sdpa"
    )
    sdpa.load_state_dict(manual.state_dict())
    manual.eval()
    sdpa.eval()
    return manual, sdpa


def _reference_manual_forward(
    module: _WMStyleAttention,
    x: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """The literal pre-H5 manual formula, recomputed from the module weights."""
    bsz, seq_len, _dim = x.shape
    x = module.norm(x)
    qkv = module.to_qkv(x).chunk(3, dim=-1)
    q, k, v = (
        t.reshape(bsz, seq_len, module.heads, module.dim_head).transpose(1, 2)
        for t in qkv
    )
    dots = torch.matmul(q, k.transpose(-1, -2)) * module.scale
    if mask is not None:
        dots = dots + mask.to(device=dots.device, dtype=dots.dtype)[None, None]
    attn = torch.softmax(dots, dim=-1)
    attn = module.dropout(attn)
    out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, -1)
    return module.to_out(out)


def test_sdpa_matches_manual_no_mask() -> None:
    manual, sdpa = _make_pair(dim=32, heads=4, dim_head=8)
    torch.manual_seed(1)
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        out_manual = manual(x)
        out_sdpa = sdpa(x)
    assert torch.allclose(out_manual, out_sdpa, atol=_ATOL, rtol=_RTOL)


def test_sdpa_matches_manual_with_additive_mask() -> None:
    manual, sdpa = _make_pair(dim=32, heads=4, dim_head=8)
    torch.manual_seed(2)
    x = torch.randn(2, 6, 32)
    seq_len = x.shape[1]
    # Additive float bias mask [S, S]: causal -inf above the diagonal.
    mask = torch.zeros(seq_len, seq_len)
    mask = mask.masked_fill(
        torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    with torch.no_grad():
        out_manual = manual(x, mask=mask)
        out_sdpa = sdpa(x, mask=mask)
    assert torch.allclose(out_manual, out_sdpa, atol=_ATOL, rtol=_RTOL)


def test_manual_path_is_byte_for_byte_unchanged() -> None:
    torch.manual_seed(0)
    manual = _WMStyleAttention(32, heads=4, dim_head=8, dropout=0.0)
    manual.eval()
    torch.manual_seed(3)
    x = torch.randn(2, 6, 32)
    seq_len = x.shape[1]
    mask = torch.zeros(seq_len, seq_len)
    mask = mask.masked_fill(
        torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    with torch.no_grad():
        golden_no_mask = _reference_manual_forward(manual, x, None)
        golden_mask = _reference_manual_forward(manual, x, mask)
        out_no_mask = manual(x)
        out_mask = manual(x, mask=mask)
    assert torch.equal(out_no_mask, golden_no_mask)
    assert torch.equal(out_mask, golden_mask)


def test_manual_is_the_default_attn_impl() -> None:
    module = _WMStyleAttention(32, heads=4, dim_head=8, dropout=0.0)
    assert module.attn_impl == "manual"


def test_invalid_attn_impl_rejected() -> None:
    try:
        _WMStyleAttention(32, heads=4, dim_head=8, attn_impl="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid attn_impl")
