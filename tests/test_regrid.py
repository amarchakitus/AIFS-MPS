"""Tests for the local 0.25 deg -> N320 interpolation matrices.

We deliberately do not call `earthkit.regrid` here (that would need the network); these
assert the structural properties that make the local matmul a valid substitute. The
bit-for-bit equivalence with `earthkit.regrid.interpolate` was checked once, by hand,
against live open-data fields.
"""

from __future__ import annotations

import numpy as np
import pytest

from aifs_mps.regrid import LATLON_SHAPE
from aifs_mps.regrid import N320_POINTS
from aifs_mps.regrid import N_LATLON
from aifs_mps.regrid import latlon_to_n320_matrix
from aifs_mps.regrid import to_n320


@pytest.fixture(scope="module")
def matrix():
    return latlon_to_n320_matrix()


def test_matrix_is_selected_by_shape_not_filename(matrix):
    """`regrid/` also holds the inverse (N320 -> 0.25 deg); we must pick the forward one."""
    assert matrix.shape == (N320_POINTS, N_LATLON)


def test_rows_are_a_partition_of_unity(matrix):
    """Every target point is a weighted average, so each row must sum to 1.

    This is what makes the operator an interpolation rather than an arbitrary linear map:
    a constant field in must give the same constant out.
    """
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-9)
    assert matrix.data.min() >= 0.0  # no negative weights -> no spurious over/undershoot


def test_constant_field_is_preserved(matrix):
    out = to_n320(np.full(LATLON_SHAPE, 42.0), matrix)
    assert out.shape == (N320_POINTS,)
    assert np.allclose(out, 42.0)


def test_nans_propagate_locally(matrix):
    """Wave fields are undefined over land; those NaNs must survive regridding but not
    contaminate the whole grid."""
    field = np.zeros(LATLON_SHAPE)
    field[0, 0] = np.nan
    out = to_n320(field, matrix)
    assert np.isnan(out).any(), "NaN was silently dropped"
    assert np.isnan(out).mean() < 0.01, "a single NaN spread far too widely"


def test_rejects_wrong_source_grid(matrix):
    with pytest.raises(ValueError, match="Unexpected source grid"):
        to_n320(np.zeros((181, 360)), matrix)


def test_missing_matrix_reports_what_it_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="No regrid matrices"):
        latlon_to_n320_matrix(tmp_path)
