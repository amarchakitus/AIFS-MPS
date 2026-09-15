"""Silence a torch deprecation in anemoi's NaN imputer by fixing its cause.

``BaseImputer.get_nans`` builds its index as a Python *list*::

    idx = [slice(None), slice(None)] + [0] * (x.ndim - 4) + [slice(None), slice(None)]
    return torch.isnan(x[idx])

Indexing with a non-tuple sequence is deprecated, so every forecast step emits::

    UserWarning: Using a non-tuple sequence for multidimensional indexing is deprecated
    ... use x[tuple(seq)] instead of x[seq]. In pytorch 2.9 this will be interpreted as
    x[torch.tensor(seq)], which will result either in an error or a different result.

The results are **not** currently wrong: verified on the real input shapes (ndim 4/5/6) that
``x[idx]`` and ``x[tuple(idx)]`` give identical NaN masks under torch 2.14. The warning's
own text is stale -- it says the change lands in 2.9, and it has not by 2.14. And when it
does land, ``idx`` holds ``slice`` objects, which ``torch.tensor`` cannot represent, so this
would raise rather than quietly return something different.

Still worth fixing rather than suppressing: the tuple form is what upstream recommends, it
is provably identical today, and it removes both the per-step noise and the future break.
Both pinned anemoi versions (0.9.3 and 0.11.2) have the same one-line body.
"""

from __future__ import annotations

import logging

import torch

LOG = logging.getLogger(__name__)

__all__ = ["patch_imputer_indexing"]


def patch_imputer_indexing() -> bool:
    """Reindex `BaseImputer.get_nans` with a tuple. Returns whether it patched."""
    try:
        from anemoi.models.preprocessing.imputer import BaseImputer
    except ImportError:
        return False

    if not hasattr(BaseImputer, "get_nans"):
        return False

    def get_nans(self, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG001 - must match upstream signature
        """NaN locations of `x`, shaped (batch, time, ..., grid).

        Identical to upstream but indexes with a tuple, which is the non-deprecated form.
        """
        idx = (slice(None), slice(None), *([0] * (x.ndim - 4)), slice(None), slice(None))
        return torch.isnan(x[idx])

    BaseImputer.get_nans = get_nans
    LOG.info("Patched BaseImputer.get_nans: tuple indexing (silences a torch deprecation)")
    return True
