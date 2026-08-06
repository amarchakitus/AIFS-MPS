"""Interpolation between the 0.25 deg lat/lon grid and the model's N320 reduced Gaussian grid.

``earthkit.regrid.interpolate`` (what the ECMWF notebooks use) fetches its matrix from ECMWF
on first use. Instead the ``support/regrid`` matrices -- obtained from ECMWF's earthkit-geo
package -- are applied directly: each is a SciPy CSR operator in an ``.npz``, and
interpolation is one sparse matrix-vector product against the row-major-flattened source
field. Bit-identical to ``earthkit.regrid.interpolate`` (verified: max abs difference 0.0 on
2t and msl), with no network access. Both directions are used (lat/lon -> N320 for the
initial conditions, N320 -> lat/lon for the forecast output) and the right operator is
chosen by *shape*, because the files are named by content hash.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .paths import regrid_dir

LOG = logging.getLogger(__name__)

__all__ = [
    "LATLON_SHAPE",
    "N320_POINTS",
    "N_LATLON",
    "latlon_to_n320_matrix",
    "n320_to_latlon_matrix",
    "to_latlon",
    "to_n320",
]

LATLON_SHAPE = (721, 1440)  # 0.25 deg global
N_LATLON = LATLON_SHAPE[0] * LATLON_SHAPE[1]  # 1038240
N320_POINTS = 542080

# Output grid coordinates, matching the 0..360 convention the fields are rolled to.
LATITUDES = np.linspace(90, -90, LATLON_SHAPE[0])
LONGITUDES = np.linspace(0, 359.75, LATLON_SHAPE[1])


@functools.lru_cache(maxsize=4)
def load_matrix(n_target: int, n_source: int, directory: str | Path | None = None) -> sp.csr_matrix:
    """Load the ``(n_target, n_source)`` CSR operator, identified by shape not filename."""
    directory = Path(directory) if directory is not None else regrid_dir()
    candidates = sorted(directory.glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No regrid matrices (*.npz) found in {directory}")

    found = {}
    for path in candidates:
        matrix = sp.load_npz(path)
        found[path.name] = matrix.shape
        if matrix.shape == (n_target, n_source):
            LOG.debug("Regridding with %s (%d x %d, %d nnz)", path.name, n_target, n_source, matrix.nnz)
            return matrix.tocsr()

    listing = "\n  ".join(f"{name}: {shape}" for name, shape in found.items())
    raise FileNotFoundError(f"No ({n_target}, {n_source}) matrix in {directory}. Found:\n  {listing}")


def latlon_to_n320_matrix(directory: str | Path | None = None) -> sp.csr_matrix:
    """Operator for initial conditions: 0.25 deg (721x1440) -> N320."""
    return load_matrix(N320_POINTS, N_LATLON, directory)


def n320_to_latlon_matrix(directory: str | Path | None = None) -> sp.csr_matrix:
    """Operator for forecast output: N320 -> 0.25 deg (721x1440)."""
    return load_matrix(N_LATLON, N320_POINTS, directory)


def to_n320(values: np.ndarray, matrix: sp.csr_matrix) -> np.ndarray:
    """Interpolate one (721, 1440) field onto the N320 grid.

    NaNs propagate to every target point that draws on them, which is what we want: the
    wave fields are undefined over land and must stay undefined.
    """
    if values.shape != LATLON_SHAPE:
        raise ValueError(f"Unexpected source grid {values.shape}, expected {LATLON_SHAPE}")
    return matrix @ values.reshape(-1)


def to_latlon(
    values: np.ndarray, matrix: sp.csr_matrix, shape: tuple[int, int] = LATLON_SHAPE
) -> np.ndarray:
    """Interpolate one N320 field back onto the 0.25 deg grid.

    Sizes are validated against `matrix` rather than the module constants, so the same code
    path works for the small stand-in grids used in the tests.
    """
    values = np.asarray(values).reshape(-1)
    if values.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix expects {matrix.shape[1]} source points, got {values.shape[0]}")
    if matrix.shape[0] != shape[0] * shape[1]:
        raise ValueError(f"Matrix produces {matrix.shape[0]} points, incompatible with {shape}")
    return (matrix @ values).reshape(shape)
