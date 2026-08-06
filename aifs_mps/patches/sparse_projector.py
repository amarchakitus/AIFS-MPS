"""Run AIFS-ENS's sparse noise projection entirely on MPS (AIFS-ENS only).

AIFS-ENS v2 draws its ensemble spread through ``NoiseConditioning`` -> ``SparseProjector``,
whose sparse COO matrix (40320 x 5248, ~2.0M non-zeros) projects a low-rank noise vector
onto the hidden mesh. MPS has no sparse backend at all, which breaks two separate things:
``torch.load(map_location="mps")`` cannot rebuild the tensor (its indices and values arrive
as MPS tensors and ``torch.sparse_coo_tensor`` raises), and ``SparseProjector.forward``
calls ``torch.sparse.mm``, for which MPS has no kernel.

A COO matmul is just gather-multiply-scatter, so the matrix is decomposed once into
(row, col, value) triples on the GPU and applied with ``index_add_``: numerically identical
to ``torch.sparse.mm`` (verified to 8e-7), entirely on-device, and needing one
``(nnz, channels)`` temporary -- 32 MB at 4 noise channels -- rather than the 846 MB a dense
matrix would cost. AIFS Single v2 is deterministic and has no noise projector, so this is a
no-op there.
"""

from __future__ import annotations

import logging

import torch

LOG = logging.getLogger(__name__)

__all__ = ["patch_sparse_projector"]


def _triples(self, device):
    """Decompose the COO matrix into device-resident (rows, cols, values), once."""
    cached = getattr(self, "_aifs_mps_triples", None)
    if cached is not None and cached[0].device == device:
        return cached

    # `projection_matrix` is stored transposed, which leaves it uncoalesced;
    # `.indices()` requires a coalesced tensor.
    matrix = self.projection_matrix
    if not matrix.is_coalesced():
        matrix = matrix.coalesce()
    indices = matrix.indices()
    triples = (
        indices[0].to(device),
        indices[1].to(device),
        matrix.values().to(device),
        matrix.shape[0],
    )
    self._aifs_mps_triples = triples
    return triples


def patch_sparse_projector() -> bool:
    """Patch sparse loading and projection if this anemoi-models has a SparseProjector."""
    # Rebuilding sparse tensors on the CPU is needed regardless of which model is loading:
    # it is cheap, and only sparse tensors are affected, so everything else still lands
    # straight on MPS (unlike forcing map_location="cpu" for the whole checkpoint).
    original_rebuild = torch._utils._rebuild_sparse_tensor

    def _rebuild_sparse_tensor(layout, data):
        data = tuple(t.cpu() if torch.is_tensor(t) else t for t in data)
        return original_rebuild(layout, data)

    torch._utils._rebuild_sparse_tensor = _rebuild_sparse_tensor

    try:
        from anemoi.models.layers.sparse_projector import SparseProjector
    except ImportError:
        return False  # anemoi-models 0.9.3: deterministic model, no noise projector

    def forward(self, x, *args, **kwargs):  # noqa: ARG001 - must match upstream signature
        rows, cols, values, n_rows = _triples(self, x.device)

        with torch.amp.autocast(device_type=x.device.type, enabled=self.autocast):
            out = torch.zeros((x.shape[0], n_rows, x.shape[2]), device=x.device, dtype=values.dtype)
            for i in range(x.shape[0]):
                # out[rows] += values * x[cols] -- exactly a COO matmul, one nnz-sized
                # temporary at a time so memory does not scale with the dense shape.
                out[i].index_add_(
                    0, rows, x[i].to(values.dtype).index_select(0, cols) * values.unsqueeze(1)
                )

        return out.to(dtype=x.dtype)

    SparseProjector.forward = forward
    LOG.info("Patched SparseProjector.forward: sparse noise projection via index_add_ on-device")
    return True
