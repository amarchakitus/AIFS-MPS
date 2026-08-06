"""Stub modules for CUDA-only packages the AIFS checkpoints reference.

Both checkpoints are whole pickled ``nn.Module`` objects, so ``torch.load`` *imports* the
libraries they were trained with before any monkey-patch can run. Two of those cannot be
imported at all on Apple silicon, so they are stubbed in ``sys.modules`` first:

``flash_attn`` (both models)
    The pickle names ``flash_attn.flash_attn_interface.flash_attn_func`` because
    ``FlashAttentionWrapper`` stores it as a plain instance attribute.
``anemoi.models.triton.gt`` (AIFS-ENS only)
    The ENS checkpoint was trained with anemoi-models 0.11.2's Triton GraphTransformer
    kernel selected. Triton has no macOS build and the real module *raises* on import, so
    the checkpoint cannot even be opened.

Every stub raises if actually called, so a missed patch fails loudly rather than silently
producing plausible-looking but wrong weather.
"""

from __future__ import annotations

import sys
import types

__all__ = ["install", "install_flash_attn", "install_triton_gt"]

# Must be >= 2.6.0 and < 3: anemoi gates rotary-embedding support on this.
_FAKE_FLASH_ATTN_VERSION = "2.7.4.post1"


def _unavailable(*args, **kwargs):  # noqa: ARG001 - stands in for arbitrary flash_attn calls
    raise RuntimeError(
        "flash_attn is a stub on this platform (no CUDA). "
        "aifs_mps.patches.attention should have replaced every call site. "
        "If you see this, the patch did not apply."
    )


class _RotaryEmbedding:
    """Placeholder for ``flash_attn.layers.rotary.RotaryEmbedding``.

    Neither AIFS checkpoint enables rotary embeddings, but the symbol must exist for the
    import inside ``FlashAttentionWrapper.__init__``.
    """

    def __init__(self, *args, **kwargs):  # noqa: ARG002 - must match upstream signature
        _unavailable()


class _TritonGraphTransformerFunction:
    """Placeholder for ``anemoi.models.triton.gt.GraphTransformerFunction``.

    Only ``apply`` is referenced by the pickle; ``patches.graph_transformer`` reroutes the
    blocks onto anemoi's own PyG backend before it could ever be called.
    """

    @staticmethod
    def apply(*args, **kwargs):  # noqa: ARG004 - must match upstream signature
        raise RuntimeError(
            "The Triton GraphTransformer kernel is a stub on this platform. "
            "aifs_mps.patches.graph_transformer should have routed this block to the PyG "
            "backend. If you see this, the patch did not apply."
        )


def _stub(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module._aifs_mps_stub = True
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _already_stubbed(name: str) -> bool:
    return getattr(sys.modules.get(name), "_aifs_mps_stub", False)


def install_flash_attn() -> None:
    """Register the stub ``flash_attn`` modules in :data:`sys.modules` (idempotent)."""
    if _already_stubbed("flash_attn"):
        return

    interface = _stub(
        "flash_attn.flash_attn_interface",
        flash_attn_func=_unavailable,
        flash_attn_varlen_func=_unavailable,
        flash_attn_qkvpacked_func=_unavailable,
    )
    rotary = _stub("flash_attn.layers.rotary", RotaryEmbedding=_RotaryEmbedding)
    layers = _stub("flash_attn.layers", rotary=rotary)
    layers.__path__ = []
    flash_attn = _stub(
        "flash_attn",
        __version__=_FAKE_FLASH_ATTN_VERSION,
        flash_attn_interface=interface,
        flash_attn_func=interface.flash_attn_func,
        flash_attn_varlen_func=interface.flash_attn_varlen_func,
        layers=layers,
    )
    flash_attn.__path__ = []  # mark as a package so submodule imports resolve

    sys.modules.update(
        {
            "flash_attn": flash_attn,
            "flash_attn.flash_attn_interface": interface,
            "flash_attn.layers": layers,
            "flash_attn.layers.rotary": rotary,
        }
    )


def install_triton_gt() -> None:
    """Register a stub ``anemoi.models.triton.gt`` (idempotent, harmless on 0.9.3).

    ``anemoi.models.triton`` is a namespace package with no ``__init__.py``, so importing
    the parent is harmless; only ``.gt`` raises. Stubbing that one module is enough.
    """
    name = "anemoi.models.triton.gt"
    if _already_stubbed(name):
        return
    sys.modules[name] = _stub(name, GraphTransformerFunction=_TritonGraphTransformerFunction)


def install() -> None:
    """Install every stub needed before ``torch.load`` (idempotent)."""
    install_flash_attn()
    install_triton_gt()
