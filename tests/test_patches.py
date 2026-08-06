"""The patches that have to be in place before `torch.load` sees the checkpoint.

Both AIFS checkpoints are whole pickled `nn.Module` objects, so the unpickler imports the
CUDA-only libraries they were trained with before any of our code could intervene. That
makes three properties load-bearing, and none of them can be checked by running a forecast
on this machine -- by the time they are wrong, the checkpoint has already failed to open or,
worse, has opened and is quietly computing something else:

* the stubs must be registered under the *exact* dotted names the pickles name. A typo is
  invisible until a 500 MB checkpoint refuses to load;
* every stub must raise when called. The entire safety argument for stubbing CUDA kernels
  is that a missed patch stops the run rather than returning plausible-looking weather;
* the capability detection must be a quiet no-op on the anemoi-models version that does not
  have the symbol, since both runtimes share one `apply_all()` call site.

The suite runs under anemoi-models 0.9.3 (AIFS Single v2), so the "absent" half is asserted
against the real library and the 0.11.2 half is simulated with stand-ins. Everything here
mutates global state -- `sys.modules`, `torch._utils`, anemoi's own classes -- so each
fixture restores exactly what it replaced.
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from anemoi.models.layers.block import GraphTransformerBaseBlock
from packaging.version import Version

from aifs_mps.patches import graph_transformer as gt_patch
from aifs_mps.patches import stubs
from aifs_mps.patches.graph_transformer import patch_graph_transformer
from aifs_mps.patches.sparse_projector import patch_sparse_projector

# Which half of the version fork this interpreter is on.
HAS_TRITON_BACKEND = hasattr(GraphTransformerBaseBlock, "apply_gt")
SPARSE_PROJECTOR_MODULE = "anemoi.models.layers.sparse_projector"
HAS_SPARSE_PROJECTOR = importlib.util.find_spec(SPARSE_PROJECTOR_MODULE) is not None

FLASH_ATTN_MODULES = (
    "flash_attn",
    "flash_attn.flash_attn_interface",
    "flash_attn.layers",
    "flash_attn.layers.rotary",
)
TRITON_GT_MODULE = "anemoi.models.triton.gt"

# Every name the stubs claim in sys.modules, saved and restored around each test.
STUB_MODULES = (*FLASH_ATTN_MODULES, TRITON_GT_MODULE)

# (module, attribute) pairs the pickled checkpoints reference by name.
CHECKPOINT_GLOBALS = [
    ("flash_attn.flash_attn_interface", "flash_attn_func"),
    ("flash_attn.layers.rotary", "RotaryEmbedding"),
    ("anemoi.models.triton.gt", "GraphTransformerFunction"),
]

_ABSENT = object()


@pytest.fixture
def installed_stubs():
    """Install the stubs into a `sys.modules` that is put back exactly as it was."""
    saved = {name: sys.modules.get(name, _ABSENT) for name in STUB_MODULES}
    for name in STUB_MODULES:
        sys.modules.pop(name, None)

    stubs.install()
    yield

    for name, module in saved.items():
        if module is _ABSENT:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _pickled_reference(module: str, name: str) -> bytes:
    """The bytes a checkpoint holds for ``module.name`` (the GLOBAL opcode, verbatim)."""
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n."


# -- stub registration -----------------------------------------------------------


@pytest.mark.parametrize(("module", "name"), CHECKPOINT_GLOBALS, ids=lambda v: v)
def test_the_checkpoints_globals_resolve_through_the_stubs(module, name, installed_stubs):
    """Resolve each name the way the unpickler does, so a typo in a dotted name fails here.

    `torch.load` reaches these through `Unpickler.find_class`, which imports the module and
    getattrs the name -- exactly what `pickle.loads` of a bare GLOBAL opcode does.
    """
    resolved = pickle.loads(_pickled_reference(module, name))
    assert resolved is not None
    assert sys.modules[module]._aifs_mps_stub, f"{module} must be our stub, not a real import"


def test_install_is_idempotent(installed_stubs):
    """`apply_all()` may run more than once; re-registering would orphan references.

    An already-unpickled `FlashAttentionWrapper` holds `flash_attn_func` as a plain instance
    attribute, so replacing the module object later would leave live objects pointing at a
    stub that is no longer the installed one.
    """
    before = {name: sys.modules[name] for name in STUB_MODULES}

    stubs.install()

    assert {name: sys.modules[name] for name in STUB_MODULES} == before
    for name, module in before.items():
        assert sys.modules[name] is module, f"{name} was rebuilt on the second install()"


# -- the safety property: a missed patch must fail loudly -------------------------


def test_every_flash_attn_stub_raises_when_called(installed_stubs):
    """Nothing in the flash_attn surface may quietly return; that would be wrong weather."""
    called = []
    for module_name in FLASH_ATTN_MODULES:
        module = sys.modules[module_name]
        for attr, value in vars(module).items():
            if attr.startswith("_") or isinstance(value, types.ModuleType):
                continue
            called.append(f"{module_name}.{attr}")
            with pytest.raises(RuntimeError, match="patch did not apply"):
                value()

    assert "flash_attn.flash_attn_interface.flash_attn_func" in called
    assert "flash_attn.layers.rotary.RotaryEmbedding" in called


def test_the_triton_graph_transformer_kernel_raises_when_called(installed_stubs):
    function = sys.modules[TRITON_GT_MODULE].GraphTransformerFunction
    with pytest.raises(RuntimeError, match="patch did not apply"):
        function.apply()


def test_flash_attn_version_is_in_the_range_anemoi_gates_on(installed_stubs):
    """anemoi parses this string to decide v2 vs v3 and whether rotary is supported.

    A version that does not parse, or that lands outside [2.6, 3), silently sends anemoi
    down a different code path than the one the checkpoints were trained with.
    """
    version = Version(sys.modules["flash_attn"].__version__)
    assert Version("2.6") <= version < Version("3")


# -- capability detection: graph_transformer --------------------------------------


@pytest.fixture
def graph_transformer_block():
    """The real anemoi block class, with `apply_gt` and the conv cache restored after."""
    had = "apply_gt" in GraphTransformerBaseBlock.__dict__
    original = GraphTransformerBaseBlock.__dict__.get("apply_gt")
    cached_convs = dict(gt_patch._PYG_CONV_CACHE)

    yield GraphTransformerBaseBlock

    if had:
        GraphTransformerBaseBlock.apply_gt = original
    elif "apply_gt" in GraphTransformerBaseBlock.__dict__:
        del GraphTransformerBaseBlock.apply_gt
    gt_patch._PYG_CONV_CACHE.clear()
    gt_patch._PYG_CONV_CACHE.update(cached_convs)


@pytest.fixture
def simulated_triton_anemoi(graph_transformer_block, monkeypatch):
    """Stand in for anemoi-models 0.11.2: an `apply_gt` to wrap and a conv to reroute to.

    Both are recorders, so the tests can tell which backend a block was sent to without
    building a real PyG MessagePassing object.
    """
    import anemoi.models.layers.conv as conv_module

    recorded = SimpleNamespace(original=[], conv_built=[], conv_called=[])

    def original_apply_gt(self, query, key, value, edges, edge_index, size):
        recorded.original.append((self, query, key, value, edges, edge_index, size))
        return "triton-result"

    class RecordingConv:
        def __init__(self, out_channels):
            recorded.conv_built.append(out_channels)

        def __call__(self, *args):
            recorded.conv_called.append(args)
            return "pyg-result"

    graph_transformer_block.apply_gt = original_apply_gt
    monkeypatch.setattr(conv_module, "GraphTransformerConv", RecordingConv)

    assert patch_graph_transformer() is True, "a Triton-capable anemoi must be patched"
    return recorded


@pytest.mark.skipif(
    HAS_TRITON_BACKEND, reason="asserts the 0.9.3 shape, which has no Triton backend at all"
)
def test_no_triton_backend_means_no_patch_and_no_error(graph_transformer_block):
    """0.9.3 has no `apply_gt`; both runtimes share one call site, so this must be quiet."""
    assert patch_graph_transformer() is False
    assert not hasattr(graph_transformer_block, "apply_gt"), (
        "the no-op branch must not invent an apply_gt that anemoi never had"
    )


def test_non_triton_blocks_are_passed_through_untouched(simulated_triton_anemoi):
    """A checkpoint can mix backends; only the Triton ones may be rerouted."""
    block = SimpleNamespace(graph_attention_backend="pyg", out_channels_conv=16)
    args = ("q", "k", "v", "edges", "edge_index", 8)

    result = GraphTransformerBaseBlock.apply_gt(block, *args)

    assert result == "triton-result", "the original's return value must not be swallowed"
    assert simulated_triton_anemoi.original == [(block, *args)], "args were altered on the way"
    assert simulated_triton_anemoi.conv_built == [], "a pyg block must not build a PyG conv"


def test_triton_blocks_are_rerouted_to_the_pyg_backend(simulated_triton_anemoi):
    block = SimpleNamespace(graph_attention_backend="triton", out_channels_conv=16)

    result = GraphTransformerBaseBlock.apply_gt(block, "q", "k", "v", "edges", "edge_index", 8)

    assert result == "pyg-result"
    assert simulated_triton_anemoi.original == [], "the Triton kernel would raise if reached"
    assert simulated_triton_anemoi.conv_built == [16], "conv must be sized from the block"
    # PyG wants a (source, target) pair where the Triton kernel took a single int.
    assert simulated_triton_anemoi.conv_called == [("q", "k", "v", "edges", "edge_index", (8, 8))]


def test_the_pyg_conv_is_shared_between_blocks_of_the_same_width(simulated_triton_anemoi):
    """The processor has dozens of identical blocks; building a conv per call is not free."""
    for _ in range(3):
        block = SimpleNamespace(graph_attention_backend="triton", out_channels_conv=16)
        GraphTransformerBaseBlock.apply_gt(block, "q", "k", "v", "e", "ei", 8)

    assert simulated_triton_anemoi.conv_built == [16]


# -- capability detection: sparse_projector ---------------------------------------


@pytest.fixture
def sparse_rebuild():
    """Restore `torch._utils._rebuild_sparse_tensor`, which the patch replaces globally."""
    original = torch._utils._rebuild_sparse_tensor
    yield
    torch._utils._rebuild_sparse_tensor = original


def test_sparse_rebuild_is_patched_even_when_there_is_no_noise_projector(sparse_rebuild):
    """Returning "not patched" refers to `SparseProjector.forward` only.

    The rebuild hook is what lets the ENS checkpoint be *opened* at all, so it must be
    installed unconditionally -- including from the deterministic runtime, which is where
    this assertion runs.
    """
    before = torch._utils._rebuild_sparse_tensor

    applied = patch_sparse_projector()

    assert torch._utils._rebuild_sparse_tensor is not before, (
        "sparse rebuilding must be patched regardless of the anemoi-models version"
    )
    if not HAS_SPARSE_PROJECTOR:
        assert applied is False, "no SparseProjector here, so the patch must report no-op"


def test_sparse_rebuild_moves_tensor_data_to_cpu_and_leaves_the_rest_alone(sparse_rebuild):
    """`map_location="mps"` hands the unpickler MPS tensors, which `sparse_coo_tensor` rejects.

    The shape/`is_coalesced` entries in `data` are not tensors, so a blanket `.cpu()` would
    break every sparse load instead.
    """
    seen = []
    torch._utils._rebuild_sparse_tensor = lambda layout, data: seen.append((layout, data))
    patch_sparse_projector()

    indices = MagicMock(spec=torch.Tensor)
    values = MagicMock(spec=torch.Tensor)
    torch._utils._rebuild_sparse_tensor(torch.sparse_coo, (indices, values, (4, 4), True))

    [(layout, data)] = seen
    assert layout is torch.sparse_coo
    assert isinstance(data, tuple), "the original unpacks `data` positionally"
    assert data == (indices.cpu(), values.cpu(), (4, 4), True)


def test_the_patched_rebuild_still_produces_the_same_sparse_tensor(sparse_rebuild):
    """The wrapper sits in front of every sparse tensor in every checkpoint."""
    patch_sparse_projector()

    rebuilt = torch._utils._rebuild_sparse_tensor(
        torch.sparse_coo,
        (torch.tensor([[0, 1], [1, 0]]), torch.tensor([1.0, 2.0]), (2, 2)),
    )

    assert rebuilt.is_sparse
    assert torch.equal(rebuilt.to_dense(), torch.tensor([[0.0, 1.0], [2.0, 0.0]]))


# -- the replacement projection: it must be the same matmul -----------------------


@pytest.fixture
def sparse_projector(sparse_rebuild):
    """Stand in for anemoi 0.11.2's `SparseProjector` so the replacement forward is reachable.

    The forward is ours, not anemoi's, so a fake holder is enough -- and it is the only way
    to exercise it from the deterministic runtime the suite runs in.
    """
    saved = sys.modules.get(SPARSE_PROJECTOR_MODULE, _ABSENT)

    class FakeSparseProjector:
        def __init__(self, projection_matrix):
            self.projection_matrix = projection_matrix
            self.autocast = False

    sys.modules[SPARSE_PROJECTOR_MODULE] = types.ModuleType(SPARSE_PROJECTOR_MODULE)
    sys.modules[SPARSE_PROJECTOR_MODULE].SparseProjector = FakeSparseProjector

    assert patch_sparse_projector() is True, "a SparseProjector is present, so it must be patched"
    yield FakeSparseProjector

    if saved is _ABSENT:
        sys.modules.pop(SPARSE_PROJECTOR_MODULE, None)
    else:
        sys.modules[SPARSE_PROJECTOR_MODULE] = saved


def test_the_index_add_projection_equals_torch_sparse_mm(sparse_projector):
    """This draws the ensemble spread; a row/col swap here is plausible weather, but wrong.

    The real `projection_matrix` is stored transposed and therefore uncoalesced, which is
    why the matrix is built that way here: `.indices()` raises on an uncoalesced tensor, so
    the patch has to coalesce first, and the non-square shape catches an orientation slip.
    """
    torch.manual_seed(0)
    dense = torch.rand(7, 5)
    dense[dense < 0.6] = 0.0  # sparse, and with an all-zero row or two
    matrix = dense.t().contiguous().to_sparse().t()  # transposed => uncoalesced, like the ckpt
    assert not matrix.is_coalesced()

    x = torch.rand(2, 5, 3)
    expected = torch.stack([torch.sparse.mm(matrix, x[i]) for i in range(x.shape[0])])

    projector = sparse_projector(matrix)
    out = projector.forward(x)

    assert out.shape == (2, 7, 3), "output is (batch, mesh nodes, channels)"
    assert torch.allclose(out, expected, atol=1e-6)
    # The decomposition is cached on the instance; a stale or mis-keyed cache would show here.
    assert torch.allclose(projector.forward(x), expected, atol=1e-6)
