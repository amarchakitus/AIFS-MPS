"""Runtime patches that let the AIFS models run on Apple silicon (MPS / Metal).

Call :func:`apply_all` **before the checkpoint is loaded** -- some of it has to be in place
before ``torch.load`` even starts. The caller need not say which model is running: each
patch detects whether it applies to the anemoi-models version in this interpreter and skips
itself otherwise, so both runtimes share one call site.

============================  =======  =====  ==========================================
patch                         Single   ENS    why
============================  =======  =====  ==========================================
``stubs``                     yes      yes    flash_attn (both) and Triton (ENS) cannot
                                              be imported on macOS, but the pickles name
                                              them
``attention``                 yes      yes    flash-attn replaced with a banded SDPA that
                                              fits in memory
``graph_transformer``         no       yes    ENS pickles a Triton kernel; reroute to
                                              anemoi's own PyG backend
``sparse_projector``          no       yes    ENS's noise projection is a sparse matmul,
                                              which MPS has no kernel for
``subgraph``                  yes      yes    MPS intermittently mis-sizes the mapper's
                                              boolean edge selection (torch 2.7)
``imputer``                   yes      yes    anemoi indexes with a list, which torch
                                              deprecates; same result, no warning
============================  =======  =====  ==========================================

Nothing else needs patching: every remaining op, including the scatter reductions behind the
sparse softmax, is implemented natively for MPS in torch 2.7. ``PYTORCH_ENABLE_MPS_FALLBACK``
is deliberately *not* set, so a genuine gap raises rather than quietly degrading to the CPU.
"""

from __future__ import annotations

import logging

from .attention import ATTENTION_DTYPES
from .attention import banded_attention
from .attention import patch_attention
from .attention import resolve_attention_dtype
from .graph_transformer import patch_graph_transformer
from .imputer import patch_imputer_indexing
from .sparse_projector import patch_sparse_projector
from .stubs import install as install_stubs
from .subgraph import patch_bipartite_subgraph

LOG = logging.getLogger(__name__)

__all__ = [
    "ATTENTION_DTYPES",
    "apply_all",
    "banded_attention",
    "resolve_attention_dtype",
]


def apply_all() -> None:
    """Install every MPS patch applicable to the anemoi-models version in this process."""
    install_stubs()
    patch_attention()
    applied = {
        "graph_transformer": patch_graph_transformer(),
        "sparse_projector": patch_sparse_projector(),
        "subgraph": patch_bipartite_subgraph(),
        "imputer": patch_imputer_indexing(),
    }
    skipped = [name for name, was_applied in applied.items() if not was_applied]
    if skipped:
        LOG.debug("Patches not applicable to this anemoi-models version: %s", ", ".join(skipped))
