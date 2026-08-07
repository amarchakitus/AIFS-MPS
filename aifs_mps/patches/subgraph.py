"""Make the mapper's edge-sharding subgraph selection reliable on MPS.

``GraphTransformer*Mapper`` slices a per-chunk subgraph with
``torch_geometric.utils.bipartite_subgraph``, which selects edges by boolean mask::

    edge_mask = src_node_mask[edge_index[0]] & dst_node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr  = edge_attr[edge_mask]

Both come from the same mask, so they cannot legitimately disagree -- but on MPS they
sometimes do, and the run dies inside ``GraphTransformerConv`` on a ``key_j`` /
``edge_attr`` size mismatch.

Root cause, reproduced in isolation: ``torch.nonzero`` (and boolean-mask indexing, which
shares its machinery) has a *data-dependent output shape*, so the backend counts the set
entries on device and reads that count back. On MPS that count is intermittently **too
large**, and only for big masks:

    mask elements   wrong out of 200
        2,000,000          0
        4,000,000          0
        8,000,000          2
       13,009,920          3      <- our decoder edge count
       20,000,000         10

The excess is arbitrary (observed ratios 1.01x to 1.95x), not a clean multiple, and
``torch.mps.synchronize()` does not help -- so it is not a missing host sync but a fault in
the kernel's own count aggregation, which only kicks in once the reduction spans more than
one threadgroup. Load-dependent too: the same 60-step forecast died at step 23 on one run
and step 33 on another.

The fix avoids the faulty op entirely on MPS. The selection is expressed with only
*static-shape* primitives -- ``cumsum`` assigns each set entry its output slot, unset
entries are scattered into a dump slot that is then sliced off -- so nothing has to infer a
shape from device data. The one host readback left is the scalar total, an explicit
reduction rather than shape inference. This matched ``nonzero`` on 20/20 real-sized masks
(13M edges) and is *faster* on MPS (4.0 ms vs 6.0 ms); ``nonzero`` is kept on CPU, where it
is correct and ~16x quicker.

Both outputs then come from that single index tensor via ``index_select``, so they cannot
diverge, and its length is cross-checked against ``edge_mask.sum()`` -- an independent
reduction. That check is belt-and-braces now that the buggy op is gone, but it is nearly
free and it is what caught the bug originally: without it, a mis-sized selection leaves both
tensors consistent but *wrong*, running the decoder on the wrong edges.
"""

from __future__ import annotations

import logging

import torch

LOG = logging.getLogger(__name__)

__all__ = ["patch_bipartite_subgraph"]

# Incremented whenever the MPS readback is caught being wrong, so a run can report it.
miscount_events = 0


def _static_shape_nonzero(mask: torch.Tensor) -> torch.Tensor:
    """Indices of the True entries, without any data-dependent output shape.

    ``cumsum`` gives each set entry its 1-based output slot; unset entries are aimed at a
    dump slot at index ``n`` which is sliced away. Every intermediate has a shape known
    from ``mask.numel()`` alone, so MPS never has to size a buffer from device data.
    """
    counts = mask.cumsum(0)
    total = int(counts[-1]) if counts.numel() else 0
    destination = torch.where(mask, counts - 1, torch.full_like(counts, total))
    out = torch.zeros(total + 1, dtype=torch.long, device=mask.device)
    out.scatter_(0, destination, torch.arange(mask.numel(), device=mask.device))
    return out[:total]


def _safe_edge_selection(edge_mask: torch.Tensor) -> torch.Tensor:
    """Indices of the True entries of `edge_mask`, verified against an independent count.

    Escalates rather than always paying for the safe path: ``nonzero`` first (fastest, and
    correct on torch >= 2.13 where the MPS fault is fixed), then the static-shape
    formulation on device, then the CPU. ``mask.sum()`` -- a static-shape reduction --
    arbitrates. This keeps the fix for anyone on an affected torch without taxing everyone
    else; the expensive path measured ~1 s per forecast step on the ENS decoder.
    """
    global miscount_events

    expected = int(edge_mask.sum())
    index = edge_mask.nonzero().view(-1)
    if index.numel() == expected:
        return index

    miscount_events += 1
    LOG.warning(
        "MPS mis-sized an edge selection (%d indices for %d set entries); retrying",
        index.numel(),
        expected,
    )
    if edge_mask.device.type == "mps":
        index = _static_shape_nonzero(edge_mask)
        if index.numel() == expected:
            return index
    return edge_mask.cpu().nonzero().view(-1).to(edge_mask.device)


def patch_bipartite_subgraph() -> bool:
    """Replace the mapper's `bipartite_subgraph` with an index-select version.

    Returns whether it patched; anemoi-models without edge sharding is left alone.
    """
    from anemoi.models.layers import mapper

    original = getattr(mapper, "bipartite_subgraph", None)
    if original is None:
        return False

    def bipartite_subgraph(subset, edge_index, edge_attr=None, relabel_nodes=False, size=None, **kwargs):
        # Only the shape-critical path is reimplemented; anything exotic goes upstream.
        if kwargs or edge_attr is None or not relabel_nodes or size is None:
            return original(
                subset, edge_index, edge_attr, relabel_nodes=relabel_nodes, size=size, **kwargs
            )

        from torch_geometric.utils import index_to_mask
        from torch_geometric.utils.map import map_index

        src_subset, dst_subset = subset
        src_mask = index_to_mask(src_subset, size=size[0])
        dst_mask = index_to_mask(dst_subset, size=size[1])
        edge_mask = src_mask[edge_index[0]] & dst_mask[edge_index[1]]

        keep = _safe_edge_selection(edge_mask)
        edge_index = edge_index.index_select(1, keep)
        edge_attr = edge_attr.index_select(0, keep)

        src_index, _ = map_index(edge_index[0], src_subset, max_index=size[0], inclusive=True)
        dst_index, _ = map_index(edge_index[1], dst_subset, max_index=size[1], inclusive=True)
        return torch.stack([src_index, dst_index], dim=0), edge_attr

    mapper.bipartite_subgraph = bipartite_subgraph
    LOG.info("Patched mapper.bipartite_subgraph: index-select edge selection with count check")
    return True
