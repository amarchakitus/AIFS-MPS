"""Sliding-window attention that runs on Metal, replacing flash-attn.

Used by both AIFS variants -- the processor is a windowed transformer in each, only the head
count differs (Single v2: 16 heads x 64 dims; ENS: 8 x 128). Anemoi's non-CUDA
``SDPAAttentionWrapper`` is unusable at this scale: it builds a dense ``(seq_len, seq_len)``
mask and forces the MATH SDPA backend, so on the 40320-token o96 hidden mesh both models run
their processor on, the score tensor alone is 16 x 40320^2 = 26e9 elements, 52 GB in fp16,
with softmax materialised on top.

Both use *sliding-window* attention (``window_size = 1120``), i.e. flash-attn's
``window_size=(1120, 1120)``, which is exactly the band ``|i - j| <= 1120``.
:func:`banded_attention` evaluates that band block by block, so memory is O(seq * window)
instead of O(seq^2) -- same mask, same softmax, just a different tiling.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)

__all__ = [
    "ATTENTION_DTYPES",
    "ATTENTION_IMPLS",
    "banded_attention",
    "patch_attention",
    "resolve_attention_dtype",
]

# Which implementation backs the window. "banded" tiles the band over SDPA by hand;
# "flex" expresses it as a FlexAttention BlockMask. Flex is far more readable and is the
# upstream direction, but on Metal it measures 5-15x slower at our shapes -- see
# aifs_mps/patches/flex.py for the numbers. Selected with --attention-impl.
ATTENTION_IMPLS = ("banded", "flex")
DEFAULT_ATTENTION_IMPL = os.environ.get("AIFS_MPS_ATTN_IMPL", "banded")


# Queries are processed in blocks of this many tokens.  Each block reads
# ``block + 2 * window`` keys, so peak score memory is
# ``heads * block * (block + 2*window)`` elements.  Measured on an M4 Max the step time
# is flat from 512 to ~1536 and degrades above that, so 1024 (~200 MB of fp32 scores at
# window=1120, 16 heads) is both the fast and the cheap choice.
# Override with AIFS_MPS_ATTN_BLOCK.
DEFAULT_BLOCK = int(os.environ.get("AIFS_MPS_ATTN_BLOCK", 1024))

# Hard ceiling on ``heads * q_len * k_len`` for a single SDPA call on MPS.
#
# This is not a memory limit, it is a *correctness* limit.  Metal kernels index the
# attention score tensor with 32-bit offsets, and torch 2.7 does not check for overflow:
# an SDPA call whose implied score tensor exceeds 2**32 elements returns silently wrong
# numbers.  Verified directly -- (heads=16, q=16384, k=18624) = 1.14 * 2**32 scores gives
# a max error of 7e-2 against the CPU result, while the same shape at 1.00 * 2**32 agrees
# to 2e-6.  Left unguarded this shows up downstream as a ~6 K error in forecast 2t.
MAX_SCORE_ELEMENTS = 2**32

# Dtype the attention scores and softmax are computed in, independent of the autocast
# dtype the rest of the model runs in.  flash-attn accumulates softmax statistics in fp32
# even when q/k/v are fp16; the MPS SDPA kernels make no such promise, so the default is
# fp32 to stay close to the reference implementation.  Measured on an M4 Max, float16 is
# only ~6% faster end to end, so the accuracy is close to free.
#
#   float32   upcast q/k/v to fp32 for the score computation, cast the result back
#   float16   compute in fp16 (fastest; matches the model's autocast dtype)
#   bfloat16  compute in bf16 -- wider exponent, fewer mantissa bits than fp16
#   inherit   no cast at all; use whatever autocast handed us
#
# Selected with --attention-dtype, or the AIFS_MPS_ATTN_DTYPE environment variable.
ATTENTION_DTYPES: dict[str, torch.dtype | None] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "inherit": None,
}


def resolve_attention_dtype(name: str) -> torch.dtype | None:
    """Map an :data:`ATTENTION_DTYPES` key to a dtype (or None for 'no cast')."""
    try:
        return ATTENTION_DTYPES[name]
    except KeyError:
        raise ValueError(
            f"Unknown attention dtype {name!r}; choose from {', '.join(ATTENTION_DTYPES)}"
        ) from None


DEFAULT_ATTN_DTYPE = resolve_attention_dtype(os.environ.get("AIFS_MPS_ATTN_DTYPE", "float32"))

# Distinguishes "the caller did not pass compute_dtype" from an explicit None, which is a
# meaningful value here ("inherit the input dtype, do not cast").
_UNSET = object()

# Cache of band masks.  Keyed by (q_len, k_len, offset) where ``offset`` is the index of
# the block's first query within its key slice.  Interior blocks all share one entry, so
# this holds ~3 masks per (seq_len, window, block) combination.
_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def _band_mask(q_len: int, k_len: int, offset: int, window: int, device: torch.device) -> torch.Tensor:
    """Boolean ``(q_len, k_len)`` mask, True where ``|i - j| <= window``."""
    key = (q_len, k_len, offset, window, str(device))
    mask = _MASK_CACHE.get(key)
    if mask is None:
        qi = torch.arange(q_len, device=device).unsqueeze(1) + offset
        kj = torch.arange(k_len, device=device).unsqueeze(0)
        mask = (qi - kj).abs() <= window
        _MASK_CACHE[key] = mask
    return mask


def _safe_block(block: int, rows: int, seq_len: int, span: int) -> int:
    """Shrink ``block`` (by repeated halving) until its SDPA scores fit the MPS limit.

    ``rows`` is ``batch * heads``; each query block of length ``b`` reads at most
    ``min(seq_len, b + 2*span)`` keys, so the score tensor holds
    ``rows * b * k_len`` elements.
    """
    block = min(block, seq_len)
    while block > 1:
        k_len = min(seq_len, block + 2 * span)
        if rows * block * k_len <= MAX_SCORE_ELEMENTS:
            return block
        block = (block + 1) // 2
    return 1


def banded_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int | None,
    dropout_p: float = 0.0,
    block: int | None = None,
    compute_dtype: torch.dtype | None = _UNSET,
) -> torch.Tensor:
    """Sliding-window attention over ``(batch, heads, seq, dim)`` tensors.

    Mathematically identical to ``flash_attn_func(..., window_size=(w, w))``: every query
    ``i`` attends to keys ``j`` with ``|i - j| <= w``.  ``window_size=None`` means full
    attention, which is still evaluated blockwise so that no single SDPA call trips the
    MPS 2**32 score-element limit (see :data:`MAX_SCORE_ELEMENTS`).

    ``block`` falls back to :data:`DEFAULT_BLOCK` when None and ``compute_dtype`` to
    :data:`DEFAULT_ATTN_DTYPE` when omitted; both are read at call time so they stay
    tunable after import. Passing ``compute_dtype=None`` explicitly means "do not cast".
    """
    block = DEFAULT_BLOCK if block is None else block
    compute_dtype = DEFAULT_ATTN_DTYPE if compute_dtype is _UNSET else compute_dtype

    batch, heads, seq_len, _ = query.shape

    # Full attention == a band as wide as the sequence.  Folding it into the same path
    # keeps the overflow guard in one place.
    full = window_size is None or window_size >= seq_len - 1
    span = seq_len if full else window_size

    block = _safe_block(block, batch * heads, seq_len, span)

    in_dtype = query.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        query, key, value = query.to(compute_dtype), key.to(compute_dtype), value.to(compute_dtype)

    out = torch.empty_like(query)

    for start in range(0, seq_len, block):
        end = min(start + block, seq_len)
        k_lo = max(0, start - span)
        k_hi = min(seq_len, end + span)

        # An all-True mask is a waste of memory and bandwidth; skip it when the block's
        # key slice is entirely inside the band.
        mask = None
        if not full:
            mask = _band_mask(end - start, k_hi - k_lo, start - k_lo, window_size, query.device)

        out[..., start:end, :] = F.scaled_dot_product_attention(
            query[..., start:end, :],
            key[..., k_lo:k_hi, :],
            value[..., k_lo:k_hi, :],
            attn_mask=mask,
            dropout_p=dropout_p,
        )

    return out.to(in_dtype)


def patch_attention() -> None:
    """Swap ``FlashAttentionWrapper.forward`` for :func:`banded_attention`."""
    from anemoi.models.layers import attention as anemoi_attention

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,  # noqa: ARG001 - must match upstream signature
        causal: bool = False,
        window_size: int | None = None,
        dropout_p: float = 0.0,
        softcap=None,
        alibi_slopes=None,
    ):
        # AIFS Single v2 sets softcap=0.0 (flash-attn's "disabled") and
        # use_alibi_slopes=False, so neither feature is exercised.  Refuse rather than
        # silently ignore them, in case this module is reused with another checkpoint.
        if softcap:
            raise NotImplementedError(f"softcap={softcap} is not supported by the MPS attention patch")
        if alibi_slopes is not None:
            raise NotImplementedError("alibi slopes are not supported by the MPS attention patch")
        if causal:
            raise NotImplementedError("causal attention is not supported by the MPS attention patch")
        if getattr(self, "use_rotary_embeddings", False):
            # Rotary embeddings would come from flash_attn.layers.rotary, which is a
            # stub here.  AIFS Single v2 has use_rotary_embeddings=False.
            raise NotImplementedError("rotary embeddings are not supported by the MPS attention patch")

        if DEFAULT_ATTENTION_IMPL == "flex":
            from .flex import flex_band_attention

            if dropout_p:
                raise NotImplementedError("dropout is not supported by the flex backend")
            return flex_band_attention(
                query, key, value, window_size, compute_dtype=DEFAULT_ATTN_DTYPE
            )

        return banded_attention(query, key, value, window_size, dropout_p=dropout_p)

    anemoi_attention.FlashAttentionWrapper.forward = forward
    LOG.info("Attention implementation: %s", DEFAULT_ATTENTION_IMPL)
    LOG.info("Patched FlashAttentionWrapper.forward -> banded SDPA (block=%d, dtype=%s)", DEFAULT_BLOCK, DEFAULT_ATTN_DTYPE)
