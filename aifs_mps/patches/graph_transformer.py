"""Route Triton-backed GraphTransformer blocks onto anemoi's PyG backend (AIFS-ENS only).

The AIFS-ENS v2 checkpoint was trained with anemoi-models 0.11.2's Triton GraphTransformer
kernel selected, so every ``GraphTransformerBlock`` pickles
``graph_attention_backend = "triton"`` and ``conv = GraphTransformerFunction.apply``.
Triton has no macOS build.

Anemoi already copes with a Triton-less machine -- ``GraphTransformerBlock.__init__`` falls
back to the pure-PyG ``GraphTransformerConv`` -- but ``__init__`` never runs when a whole
model is unpickled. So we intercept ``apply_gt`` and take that same PyG branch at call time,
exactly the code path anemoi would have chosen had it built the model here.
"""

from __future__ import annotations

import logging

LOG = logging.getLogger(__name__)

__all__ = ["patch_graph_transformer"]

# GraphTransformerConv holds no learnable parameters -- just `out_channels` and a dropout
# rate -- so one instance per out_channels can be shared by every block that needs it, and
# building it here loses nothing from the checkpoint. Constructing a PyG MessagePassing
# object is not free (it introspects method signatures), hence the cache.
_PYG_CONV_CACHE: dict[int, object] = {}


def patch_graph_transformer() -> bool:
    """Patch ``apply_gt`` if this anemoi-models has a Triton backend. Returns whether it did."""
    from anemoi.models.layers.block import GraphTransformerBaseBlock
    from anemoi.models.layers.conv import GraphTransformerConv

    original_apply_gt = getattr(GraphTransformerBaseBlock, "apply_gt", None)
    if original_apply_gt is None:
        # anemoi-models 0.9.3 (AIFS Single v2): no Triton backend, no `apply_gt`, nothing
        # to reroute.
        return False

    def apply_gt(self, query, key, value, edges, edge_index, size):
        if getattr(self, "graph_attention_backend", "pyg") != "triton":
            return original_apply_gt(self, query, key, value, edges, edge_index, size)

        out_channels = self.out_channels_conv
        conv = _PYG_CONV_CACHE.get(out_channels)
        if conv is None:
            conv = GraphTransformerConv(out_channels=out_channels)
            _PYG_CONV_CACHE[out_channels] = conv

        conv_size = (size, size) if isinstance(size, int) else size
        return conv(query, key, value, edges, edge_index, conv_size)

    GraphTransformerBaseBlock.apply_gt = apply_gt
    LOG.info("Patched GraphTransformerBaseBlock.apply_gt: triton backend -> PyG backend")
    return True
