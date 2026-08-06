"""Tests for the MPS attention replacement.

:func:`aifs_mps.patches.banded_attention` must be *numerically equivalent* to what flash-attn
would have computed, not merely "close enough": a wrong attention band or a
silently-overflowing MPS kernel does not crash, it just produces a plausible-looking but
wrong forecast.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from aifs_mps.patches.attention import ATTENTION_DTYPES
from aifs_mps.patches.attention import MAX_SCORE_ELEMENTS
from aifs_mps.patches.attention import _safe_block
from aifs_mps.patches.attention import banded_attention
from aifs_mps.patches.attention import resolve_attention_dtype

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _dense_banded_reference(q, k, v, window):
    """Full-matrix masked SDPA -- the definition of sliding-window attention."""
    seq = q.shape[-2]
    i = torch.arange(seq, device=q.device)
    mask = (i[:, None] - i[None, :]).abs() <= window
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", [64, 333, 1024, 4096])
def test_matches_dense_band(device, block):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 1000, 64, device=device) for _ in range(3))
    out = banded_attention(q, k, v, 113, block=block)
    assert torch.allclose(out, _dense_banded_reference(q, k, v, 113), atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("window", [None, 10_000])
def test_full_attention_paths_agree(device, window):
    """window=None and window >= seq_len must both give plain full attention."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 500, 64, device=device) for _ in range(3))
    out = banded_attention(q, k, v, window, block=128)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5)


def test_safe_block_respects_the_mps_score_limit():
    """A block big enough to overflow 32-bit score indexing must be shrunk."""
    rows, seq, span = 16, 40320, 1120

    assert _safe_block(1024, rows, seq, span) == 1024  # already safe, untouched

    clamped = _safe_block(16384, rows, seq, span)
    assert clamped < 16384
    assert rows * clamped * min(seq, clamped + 2 * span) <= MAX_SCORE_ELEMENTS


@pytest.mark.skipif("mps" not in DEVICES, reason="requires Apple silicon")
def test_aifs_shape_is_block_size_invariant():
    """At the real AIFS v2 processor shape, an oversized block request must be clamped
    rather than silently returning wrong numbers."""
    torch.manual_seed(0)
    heads, seq, dim, window = 16, 40320, 64, 1120
    q, k, v = (torch.randn(1, heads, seq, dim, device="mps") for _ in range(3))
    small = banded_attention(q, k, v, window, block=1024)
    huge = banded_attention(q, k, v, window, block=16384)
    assert (small - huge).abs().max().item() < 1e-5


# -- attention compute dtype -----------------------------------------------------


def test_attention_dtype_names_resolve():
    assert resolve_attention_dtype("float32") is torch.float32
    assert resolve_attention_dtype("bfloat16") is torch.bfloat16
    # "inherit" means "do not cast", which is a real value, not a missing one.
    assert resolve_attention_dtype("inherit") is None
    assert set(ATTENTION_DTYPES) == {"float32", "float16", "bfloat16", "inherit"}


def test_unknown_attention_dtype_is_rejected():
    with pytest.raises(ValueError, match="Unknown attention dtype"):
        resolve_attention_dtype("float64")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(ATTENTION_DTYPES))
def test_every_attention_dtype_runs_and_preserves_output_dtype(device, name):
    """Whatever the scores are computed in, the result must come back in the input dtype
    so the surrounding autocast graph is unaffected."""
    torch.manual_seed(0)
    in_dtype = torch.float16
    q, k, v = (torch.randn(1, 4, 300, 64, device=device, dtype=in_dtype) for _ in range(3))
    out = banded_attention(q, k, v, 50, block=128, compute_dtype=resolve_attention_dtype(name))
    assert out.dtype == in_dtype
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("device", DEVICES)
def test_inherit_does_not_cast(device):
    """'inherit' must be exactly the same computation as asking for the input dtype."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 300, 64, device=device, dtype=torch.float16) for _ in range(3))
    inherited = banded_attention(q, k, v, 50, block=128, compute_dtype=None)
    explicit = banded_attention(q, k, v, 50, block=128, compute_dtype=torch.float16)
    assert torch.equal(inherited, explicit)


@pytest.mark.skipif("mps" not in DEVICES, reason="requires Apple silicon")
def test_float32_scores_beat_float16_against_an_fp32_reference():
    """The fp32 default exists to be more accurate; assert it actually is.

    MPS only: since torch 2.13 the CPU SDPA path upcasts fp16 internally, so both dtypes
    give bit-identical error there and the option is a no-op. On MPS -- the device this
    default exists for -- fp32 scores really are closer to an fp32 reference.
    """
    device = "mps"
    torch.manual_seed(0)
    q32, k32, v32 = (torch.randn(1, 4, 400, 64, device=device) for _ in range(3))
    reference = _dense_banded_reference(q32, k32, v32, 60)

    q, k, v = (t.half() for t in (q32, k32, v32))
    err32 = (banded_attention(q, k, v, 60, block=128, compute_dtype=torch.float32).float() - reference).abs().max()
    err16 = (banded_attention(q, k, v, 60, block=128, compute_dtype=torch.float16).float() - reference).abs().max()
    assert err32 < err16


# -- FlexAttention backend -------------------------------------------------------


@pytest.mark.skipif("mps" not in DEVICES, reason="flex is compiled for the GPU path")
class TestFlexBackend:
    """The flex backend must be interchangeable with the banded one. It is not the default
    (5-15x slower on Metal), so nothing else would notice if it silently diverged."""

    def test_it_matches_a_dense_masked_reference(self):
        from aifs_mps.patches.flex import flex_available
        from aifs_mps.patches.flex import flex_band_attention

        if not flex_available():
            pytest.skip("FlexAttention unavailable in this torch build")

        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 4, 512, 64, device="mps") for _ in range(3))
        got = flex_band_attention(q, k, v, 128)
        assert torch.allclose(got, _dense_banded_reference(q, k, v, 128), atol=1e-4)

    def test_it_agrees_with_the_banded_implementation(self):
        from aifs_mps.patches.flex import flex_available
        from aifs_mps.patches.flex import flex_band_attention

        if not flex_available():
            pytest.skip("FlexAttention unavailable in this torch build")

        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 4, 512, 64, device="mps") for _ in range(3))
        for window in (64, 128, None):
            a = flex_band_attention(q, k, v, window)
            b = banded_attention(q, k, v, window)
            assert torch.allclose(a, b, atol=1e-4), f"window={window}"

    def test_the_block_mask_is_cached_by_geometry(self):
        """Rebuilding it per call would dominate the cost; it depends only on the shape."""
        from aifs_mps.patches.flex import _band_block_mask
        from aifs_mps.patches.flex import flex_available

        if not flex_available():
            pytest.skip("FlexAttention unavailable in this torch build")
        assert _band_block_mask(64, 512, "mps") is _band_block_mask(64, 512, "mps")
        assert _band_block_mask(64, 512, "mps") is not _band_block_mask(32, 512, "mps")
