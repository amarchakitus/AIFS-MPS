"""FlexAttention backend for the sliding-window processor attention.

``torch.nn.attention.flex_attention`` expresses the window as a ``mask_mod`` and builds a
block-sparse ``BlockMask``, so the band is described declaratively rather than tiled by
hand. It is the natural replacement for flash-attn's ``window_size=(w, w)`` and, on CUDA,
is the fast path.

**On Metal it is not.** Measured on an M4 Max at the real processor shape (o96 hidden mesh,
40320 tokens, window 1120), warm, against :func:`~aifs_mps.patches.attention.banded_attention`:

    model   dtype     banded    flex     ratio
    single  fp32        63 ms   581 ms    9.2x
    single  bf16        71 ms   359 ms    5.1x
    ENS     fp32        86 ms  1328 ms   15.4x
    ENS     bf16        94 ms   930 ms    9.9x

Peak memory is identical and the outputs agree (1e-6 fp32, 1e-3 bf16 -- accumulation order).
Tuning does not close the gap: BLOCK_SIZE 64/128/256 and ``mode="max-autotune"`` /
``"reduce-overhead"`` all land within 919-993 ms on the ENS shape. FlexAttention's speed
comes from Triton-fused kernels, and the Inductor Metal backend does not generate anything
competitive with the hand-tuned SDPA that the banded path calls.

So this is available via ``--attention-impl flex`` but is **not** the default. It is worth
keeping: it is far more readable, it is the upstream direction of travel, and it should be
re-benchmarked whenever torch's MPS backend improves.

Two things are load-bearing here. Uncompiled ``flex_attention`` is never used -- it warns
that it "materializes the full scores matrix", which at 40320 tokens is exactly the tens of
GB the banded path exists to avoid. And the call must run with autocast **disabled**:
``flex_attention`` is a HigherOrderOperator with no ``AutocastMPS`` registration, so under
the model's autocast context it raises ``could not find kernel ... at dispatch key
DispatchKey.AutocastMPS``. Dtype is therefore handled explicitly, as the banded path does.
"""

from __future__ import annotations

import functools
import logging

import torch

LOG = logging.getLogger(__name__)

__all__ = ["flex_available", "flex_band_attention"]


@functools.lru_cache(maxsize=1)
def _flex_api():
    """Import and compile the flex entry points once, or report why we cannot."""
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention

    # Compiling is not optional: eager flex_attention materialises the full score matrix.
    return torch.compile(flex_attention, dynamic=False), create_block_mask


def flex_available() -> bool:
    try:
        _flex_api()
    except Exception as exc:
        LOG.debug("FlexAttention unavailable: %s", exc)
        return False
    return True


@functools.lru_cache(maxsize=8)
def _band_block_mask(window: int, seq_len: int, device: str):
    """Block-sparse mask for ``|q - kv| <= window``.

    Cached because it depends only on the geometry, which is fixed for a given model, and
    building it is far more expensive than applying it.
    """
    _, create_block_mask = _flex_api()

    def band(_b, _h, q_idx, kv_idx):
        return (q_idx - kv_idx).abs() <= window

    mask = create_block_mask(
        band, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device, _compile=True
    )
    LOG.info(
        "Built FlexAttention band mask: seq=%d window=%d -> %.1f%% sparse",
        seq_len,
        window,
        mask.sparsity(),
    )
    return mask


def flex_band_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int | None,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Sliding-window attention via FlexAttention, matching `banded_attention`'s contract.

    Every query ``i`` attends to keys ``j`` with ``|i - j| <= window_size``;
    ``window_size=None`` means full attention. The result is returned in the input dtype.
    """
    flex, _ = _flex_api()

    in_dtype = query.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        query, key, value = query.to(compute_dtype), key.to(compute_dtype), value.to(compute_dtype)

    seq_len = query.shape[-2]
    block_mask = None
    if window_size is not None and window_size < seq_len - 1:
        block_mask = _band_block_mask(int(window_size), int(seq_len), str(query.device))

    # flex_attention has no AutocastMPS kernel, so it must not be dispatched through the
    # autocast key; dtype was resolved above instead. Its default scale is 1/sqrt(E),
    # matching SDPA and flash-attn.
    with torch.amp.autocast(device_type=query.device.type, enabled=False):
        out = flex(query, key, value, block_mask=block_mask)
    return out.to(in_dtype)
