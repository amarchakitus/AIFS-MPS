"""Store-layout decisions that only fail at the very end of a multi-hour forecast.

`v3_encoding` and `apply_spec` are called once, after the rollout, when the dataset is
handed to zarr. Everything they get wrong is therefore expensive: a `KeyError` on an
unexpected dimension throws away the whole run, and a silently dropped `BitRound` filter
writes a store that is fine to read but ~3x larger than the archive it is meant to match.

`apply_spec`'s two guards are the other half of that: both fire on inputs a user can
plausibly ask for -- daily aggregates on a 12 h forecast, or a step sequence that does not
start at +6 h -- and both must say *why* rather than raising something from inside xarray's
coarsen.

The reference-equivalence of the aggregations themselves is covered in test_zarr_stream.py,
which runs this module as the oracle; these tests are only about the edges it does not hit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr
from zarr.errors import ZarrUserWarning

from aifs_mps.config import Encoding
from aifs_mps.config import OutputSpec
from aifs_mps.config import StoreSpec
from aifs_mps.zarr_layout import apply_spec
from aifs_mps.zarr_layout import quiet_consolidated_metadata_warning
from aifs_mps.zarr_layout import v3_encoding

FORECAST_DIMS = ("time", "prediction_timedelta", "lat", "lon")


def _keepbits(entry: dict) -> int:
    (bitround,) = entry["filters"]
    return bitround.codec_config["keepbits"]


@pytest.fixture
def ds():
    """Two 4-D forecast variables on a grid small enough to be free."""
    values = np.zeros((1, 2, 4, 6), dtype=np.float32)
    return xr.Dataset({"2t": (FORECAST_DIMS, values), "msl": (FORECAST_DIMS, values.copy())})


# -- v3_encoding ------------------------------------------------------------------


def test_each_dim_gets_its_own_chunk_and_shard_in_variable_order(ds):
    """The tuples are positional, so a dim looked up out of order is silently accepted.

    Distinct sizes per dimension here: any permutation, or reading both tuples from the
    same table, changes the answer.
    """
    entry = v3_encoding(
        ds,
        Encoding(
            chunks={"time": 1, "prediction_timedelta": 2, "lat": 3, "lon": 4},
            shards={"time": 1, "prediction_timedelta": 6, "lat": 9, "lon": 12},
        ),
    )["2t"]

    assert entry["chunks"] == (1, 2, 3, 4)
    assert entry["shards"] == (1, 6, 9, 12)


def test_a_per_variable_keepbits_override_wins_over_the_global_one(ds):
    """Precipitation and other skewed fields need more mantissa than the 11-bit default."""
    encoding = Encoding(keepbits=11, keepbits_by_variable={"msl": 5})
    out = v3_encoding(ds, encoding)

    assert _keepbits(out["2t"]) == 11
    assert _keepbits(out["msl"]) == 5


def test_keepbits_none_omits_the_bitround_filter_entirely(ds):
    """`keepbits: null` means store exact float32 -- not BitRound with a null argument."""
    assert "filters" not in v3_encoding(ds, Encoding(keepbits=None))["2t"]

    per_variable = v3_encoding(ds, Encoding(keepbits=11, keepbits_by_variable={"msl": None}))
    assert "filters" not in per_variable["msl"]
    assert "filters" in per_variable["2t"], "opting one variable out must not disable the rest"


def test_variables_whose_dims_are_not_in_the_chunk_table_are_skipped(ds):
    """Anything unrecognised falls back to xarray's own encoding instead of killing the write."""
    ds["spread"] = (("time", "quantile"), np.zeros((1, 3), dtype=np.float32))
    ds["n_members"] = ((), np.float32(3))

    assert set(v3_encoding(ds, Encoding())) == {"2t", "msl"}


# -- apply_spec -------------------------------------------------------------------


def _forecast(n_steps: int, *, first_hour: int = 6, init: str = "2026-08-06T00") -> xr.Dataset:
    steps = np.array(
        [np.timedelta64(first_hour + 6 * i, "h") for i in range(n_steps)], dtype="timedelta64[ns]"
    )
    values = np.arange(n_steps * 6, dtype=np.float32).reshape(1, n_steps, 2, 3)
    return xr.Dataset(
        {"2t": (FORECAST_DIMS, values)},
        coords={
            "time": [np.datetime64(init, "ns")],
            "prediction_timedelta": steps,
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0, 2.0],
        },
    )


NATIVE_SPEC = StoreSpec((OutputSpec("2t", "2t", "native", "K"),))
DAILY_SPEC = StoreSpec((OutputSpec("2t", "2t", "daily_mean", "K"),))


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(_forecast(4, first_hour=12), id="does-not-start-at-+6h"),
        pytest.param(_forecast(4).isel(prediction_timedelta=[0, 1, 3]), id="gap-in-the-middle"),
    ],
)
def test_steps_that_are_not_contiguous_6_hourly_are_rejected(steps):
    """Every window in this module assumes step *index* == lead time / 6 h.

    A subset or a reordered dataset would otherwise be aggregated into windows that are
    labelled with the wrong lead time, which is unrecoverable once written.
    """
    with pytest.raises(ValueError, match=r"contiguous 6-hourly steps starting at \+6h"):
        apply_spec(steps, NATIVE_SPEC)


def test_daily_aggregation_on_a_forecast_too_short_for_one_day_says_what_to_do():
    """`--lead-time 12` with the default config: the store would otherwise be empty."""
    with pytest.raises(ValueError, match="no complete UTC calendar day") as excinfo:
        apply_spec(_forecast(2), DAILY_SPEC)

    assert "disable" in str(excinfo.value), "the message must name the way out"


def test_a_short_forecast_is_fine_when_no_daily_output_is_configured():
    """The complete-day requirement must be scoped to the aggregates that need it.

    This is the `--no-daily-aggregates` path, and it is the only way to get output at all
    out of a forecast shorter than a calendar day.
    """
    out = apply_spec(_forecast(2), NATIVE_SPEC)

    assert out["2t"].sizes["prediction_timedelta"] == 2
    assert "prediction_timedelta_daily" not in out.dims


# -- warning suppression ----------------------------------------------------------


def test_the_consolidated_metadata_filter_does_not_leak_out_of_the_context():
    """A module that quietly mutates global warning filters silences its importer too."""
    before = warnings.filters[:]

    with quiet_consolidated_metadata_warning():
        assert warnings.filters != before, "the context did not actually install a filter"

    assert warnings.filters == before


def test_only_the_consolidated_metadata_warning_is_silenced():
    """We write consolidated metadata on purpose; every other zarr complaint still matters."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with quiet_consolidated_metadata_warning():
            warnings.warn(
                "Consolidated metadata is currently not part in the Zarr version 3 specification.",
                ZarrUserWarning,
                stacklevel=1,
            )
            warnings.warn("something else is wrong with this store", ZarrUserWarning, stacklevel=1)

    assert [str(w.message) for w in seen] == ["something else is wrong with this store"]
