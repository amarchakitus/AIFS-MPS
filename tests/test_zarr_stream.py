"""The streaming writer must reproduce the reference batch path, for both model shapes.

`aifs_mps.zarr_layout` is the port of the reference scripts' batch helpers (`build_dataset`
+ `apply_spec`). `aifs_mps.zarr_stream` computes the same thing incrementally on a
background thread, into a preallocated store, across up to three region dimensions
(`number`, `prediction_timedelta`, `prediction_timedelta_daily`).

The ensemble dimension is where this can go wrong silently: a member written into the
wrong `number` slot, or members bleeding into each other through the shared accumulator,
produces a perfectly well-formed store with scrambled data. These tests give every member
distinct data and check each one lands in its own slot unchanged.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest
import scipy.sparse as sp
import xarray as xr

from aifs_mps import zarr_layout
from aifs_mps import zarr_stream
from aifs_mps.config import Encoding
from aifs_mps.config import OutputSpec
from aifs_mps.config import StoreSpec
from aifs_mps.zarr_layout import complete_day_windows
from aifs_mps.zarr_stream import ForecastZarrWriter

SMALL_SHAPE = (4, 6)
N_SMALL = SMALL_SHAPE[0] * SMALL_SHAPE[1]
N_SRC = 10

MEAN_FIELDS = ["z_500", "u_850"]
SUM_FIELDS = ["tp"]
NATIVE_FIELDS = ["2t", "msl"]
ALL_FIELDS = NATIVE_FIELDS + MEAN_FIELDS + SUM_FIELDS


@pytest.fixture
def small_grid(monkeypatch):
    """Shrink grid, chunks and shards so full stores fit in a test."""
    for module in (zarr_layout, zarr_stream):
        monkeypatch.setattr(module, "LATLON_SHAPE", SMALL_SHAPE, raising=False)
        monkeypatch.setattr(module, "LATITUDES", np.linspace(90, -90, SMALL_SHAPE[0]), raising=False)
        monkeypatch.setattr(module, "LONGITUDES", np.linspace(0, 300, SMALL_SHAPE[1]), raising=False)

    return SMALL_ENCODING


SMALL_CHUNKS = {
    "time": 1,
    "number": 1,
    "prediction_timedelta": 4,
    "prediction_timedelta_daily": 2,
    "lat": SMALL_SHAPE[0],
    "lon": SMALL_SHAPE[1],
}
SMALL_ENCODING = Encoding(
    chunks=SMALL_CHUNKS,
    shards={**SMALL_CHUNKS, "prediction_timedelta": 8, "prediction_timedelta_daily": 4},
    keepbits=None,  # exact float32 so streaming and batch can be compared directly
)


def _spec(*, daily=True, extra=()):
    """StoreSpec over the small test grid."""
    outputs = [OutputSpec(n, n, "native", "1") for n in NATIVE_FIELDS]
    if daily:
        outputs += [OutputSpec(n, n, "daily_mean", "1") for n in MEAN_FIELDS]
        outputs += [OutputSpec(n, n, "daily_sum", "1") for n in SUM_FIELDS]
    else:
        outputs += [OutputSpec(n, n, "native", "1") for n in MEAN_FIELDS + SUM_FIELDS]
    outputs += list(extra)
    return StoreSpec(tuple(outputs), SMALL_ENCODING)


@pytest.fixture
def matrix():
    rng = np.random.default_rng(0)
    dense = rng.random((N_SMALL, N_SRC))
    dense /= dense.sum(axis=1, keepdims=True)  # partition of unity, like the real operator
    return sp.csr_matrix(dense)


def _states(init, n_steps, member):
    """Per-member data, seeded by member so every member is distinguishable."""
    rng = np.random.default_rng(1000 + member)
    return [
        {
            "date": init + datetime.timedelta(hours=6 * (i + 1)),
            "fields": {f: rng.normal(size=N_SRC) for f in ALL_FIELDS},
        }
        for i in range(n_steps)
    ]


def _batch_reference(states, init, member, matrix, spec=None):
    steps = [zarr_layout.process_step(s, matrix, ALL_FIELDS) for s in states]
    ds = zarr_layout.build_dataset(steps, init, member)
    return zarr_layout.apply_spec(ds, spec or _spec()).compute()


def _stream(path, init, n_steps, n_members, matrix, **kwargs):
    with ForecastZarrWriter(
        path, init, n_steps, n_members, matrix, _spec(), **kwargs
    ) as writer:
        for m in range(n_members):
            writer.start_member(m)
            for state in _states(init, n_steps, m):
                writer.submit(state)
    return xr.open_zarr(path).compute()


# -- day-window planning ---------------------------------------------------------


def test_windows_for_00z_init_are_aligned():
    assert complete_day_windows(datetime.datetime(2021, 6, 21), 8) == [(0, 4), (4, 8)]


def test_windows_drop_incomplete_edge_days():
    assert complete_day_windows(datetime.datetime(2021, 6, 21, 6), 8) == [(3, 7)]


# -- streaming vs batch, per member ----------------------------------------------


@pytest.mark.parametrize("n_members", [1, 3])
@pytest.mark.parametrize("init_hour", [0, 6])
def test_every_member_matches_the_batch_path(tmp_path, small_grid, matrix, n_members, init_hour):
    init = datetime.datetime(2021, 6, 21, init_hour)
    n_steps = 8
    got = _stream(tmp_path / "e.zarr", init, n_steps, n_members, matrix)

    assert got.sizes["number"] == n_members
    for m in range(n_members):
        expected = _batch_reference(_states(init, n_steps, m), init, m, matrix)
        for name in expected.data_vars:
            np.testing.assert_allclose(
                got[name].isel(number=m).values,
                expected[name].isel(number=0).values,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"member {m} field {name}",
            )


def test_members_are_not_swapped_or_shared(tmp_path, small_grid, matrix):
    """Each member's data must land in its own `number` slot and nowhere else."""
    init = datetime.datetime(2021, 6, 21)
    n_members = 4
    got = _stream(tmp_path / "e.zarr", init, 8, n_members, matrix)

    for m in range(n_members):
        mine = _batch_reference(_states(init, 8, m), init, m, matrix)["2t"].isel(number=0).values
        for other in range(n_members):
            same = np.allclose(got["2t"].isel(number=other).values, mine, rtol=1e-6, atol=1e-6)
            assert same == (other == m), f"member {m} data found in slot {other}"


def test_batch_size_does_not_change_output(tmp_path, small_grid, matrix):
    init = datetime.datetime(2021, 6, 21)
    a = _stream(tmp_path / "a.zarr", init, 12, 2, matrix, batch_steps=4)
    b = _stream(tmp_path / "b.zarr", init, 12, 2, matrix, batch_steps=1)
    for name in a.data_vars:
        np.testing.assert_array_equal(a[name].values, b[name].values, err_msg=name)


def test_daily_accumulator_resets_between_members(tmp_path, small_grid, matrix):
    """A running sum leaking across the member boundary is the obvious bug here."""
    init = datetime.datetime(2021, 6, 21)
    got = _stream(tmp_path / "e.zarr", init, 8, 3, matrix)
    for m in range(3):
        states = _states(init, 8, m)
        for day, window in enumerate([slice(0, 4), slice(4, 8)]):
            raw = np.stack(
                [
                    zarr_layout.to_latlon(s["fields"]["tp"], matrix, SMALL_SHAPE)
                    for s in states[window]
                ]
            )
            np.testing.assert_allclose(
                got["tp"].isel(number=m, time=0, prediction_timedelta_daily=day).values,
                raw.sum(axis=0),
                rtol=1e-6,
                err_msg=f"member {m} day {day}",
            )


def test_disabling_aggregates_keeps_everything_six_hourly(tmp_path, small_grid, matrix):
    init = datetime.datetime(2021, 6, 21)
    path = tmp_path / "e.zarr"
    with ForecastZarrWriter(
        path, init, 8, 2, matrix, _spec(daily=False)
    ) as writer:
        for m in range(2):
            writer.start_member(m)
            for state in _states(init, 8, m):
                writer.submit(state)
    ds = xr.open_zarr(path)
    assert "prediction_timedelta_daily" not in ds.dims
    for name in ALL_FIELDS:
        assert ds[name].dims == ("time", "number", "prediction_timedelta", "lat", "lon")


# -- failure handling ------------------------------------------------------------


def test_submit_before_start_member_is_rejected(tmp_path, small_grid, matrix):
    init = datetime.datetime(2021, 6, 21)
    writer = ForecastZarrWriter(tmp_path / "e.zarr", init, 8, 2, matrix, _spec())
    with writer:
        with pytest.raises(RuntimeError, match="start_member"):
            writer.submit(_states(init, 8, 0)[0])
        for m in range(2):  # finish properly so __exit__ does not raise too
            writer.start_member(m)
            for state in _states(init, 8, m):
                writer.submit(state)


def test_member_out_of_range_is_rejected(tmp_path, small_grid, matrix):
    init = datetime.datetime(2021, 6, 21)
    writer = ForecastZarrWriter(tmp_path / "e.zarr", init, 8, 2, matrix, _spec())
    writer.__enter__()
    with pytest.raises(ValueError, match="outside the preallocated range"):
        writer.start_member(5)


def test_short_member_leaves_no_final_store(tmp_path, small_grid, matrix):
    """A member that stops early must not be renamed into place as if complete.

    One member only, so the step-count guard is what fires rather than the
    missing-member guard exercised by the next test.
    """
    init = datetime.datetime(2021, 6, 21)
    path = tmp_path / "e.zarr"
    writer = ForecastZarrWriter(path, init, 8, 1, matrix, _spec())
    writer.__enter__()
    writer.start_member(0)
    for state in _states(init, 8, 0)[:5]:
        writer.submit(state)
    with pytest.raises(RuntimeError, match="submitted 5 of 8 steps"):
        writer.close()
    assert not path.exists()
    assert writer.partial_path.exists()


def test_missing_member_leaves_no_final_store(tmp_path, small_grid, matrix):
    init = datetime.datetime(2021, 6, 21)
    path = tmp_path / "e.zarr"
    writer = ForecastZarrWriter(path, init, 8, 3, matrix, _spec())
    writer.__enter__()
    for m in range(2):  # 3 preallocated, only 2 run
        writer.start_member(m)
        for state in _states(init, 8, m):
            writer.submit(state)
    with pytest.raises(RuntimeError, match=r"missing members after run: \[2\]"):
        writer.close()
    assert not path.exists()


def test_duplicate_member_is_rejected(tmp_path, small_grid, matrix):
    init = datetime.datetime(2021, 6, 21)
    writer = ForecastZarrWriter(tmp_path / "e.zarr", init, 8, 2, matrix, _spec())
    writer.__enter__()
    writer.start_member(0)
    for state in _states(init, 8, 0):
        writer.submit(state)
    with pytest.raises(RuntimeError, match="already started"):
        writer.start_member(0)


# -- deterministic mode (no `number` dimension) ----------------------------------


def test_deterministic_store_has_no_number_dimension(tmp_path, small_grid, matrix):
    """n_members=None must produce exactly the deterministic reference layout, not a
    length-1 ensemble."""
    init = datetime.datetime(2021, 6, 21)
    path = tmp_path / "det.zarr"
    with ForecastZarrWriter(
        path, init, 8, None, matrix, _spec()
    ) as writer:
        writer.start_member(0)
        for state in _states(init, 8, 0):
            writer.submit(state)

    ds = xr.open_zarr(path).compute()
    assert "number" not in ds.dims
    assert ds["2t"].dims == ("time", "prediction_timedelta", "lat", "lon")
    assert ds["tp"].dims == ("time", "prediction_timedelta_daily", "lat", "lon")

    expected = zarr_layout.apply_spec(
        zarr_layout.build_dataset(
            [zarr_layout.process_step(s, matrix, ALL_FIELDS) for s in _states(init, 8, 0)],
            init,
        ),
        _spec(),
    ).compute()
    for name in expected.data_vars:
        np.testing.assert_allclose(
            ds[name].values, expected[name].values, rtol=1e-6, atol=1e-6, err_msg=name
        )


def test_deterministic_and_single_member_ensemble_agree(tmp_path, small_grid, matrix):
    """The same forecast written both ways must hold identical numbers -- the `number`
    dimension is packaging, not content."""
    init = datetime.datetime(2021, 6, 21)
    det = tmp_path / "det.zarr"
    with ForecastZarrWriter(det, init, 8, None, matrix, _spec()) as w:
        w.start_member(0)
        for state in _states(init, 8, 0):
            w.submit(state)
    ens = _stream(tmp_path / "ens.zarr", init, 8, 1, matrix)

    a = xr.open_zarr(det).compute()
    for name in a.data_vars:
        np.testing.assert_array_equal(
            a[name].values, ens[name].isel(number=0).values, err_msg=name
        )


# -- the new aggregations: min, max, and multi-output variables ------------------


def _agg_spec(aggregations):
    """Store `2t` under each of `aggregations`, plus one plain native field."""
    outputs = [OutputSpec("msl", "msl", "native", "Pa")]
    suffix = {"native": "", "daily_mean": "_mean", "daily_min": "_min",
              "daily_max": "_max", "daily_sum": "_sum"}
    multi = len(aggregations) > 1
    outputs += [
        OutputSpec("2t" + (suffix[a] if multi else ""), "2t", a, "K") for a in aggregations
    ]
    return StoreSpec(tuple(outputs), SMALL_ENCODING)


@pytest.mark.parametrize("aggregation", ["daily_mean", "daily_min", "daily_max", "daily_sum"])
def test_daily_aggregation_matches_numpy(tmp_path, small_grid, matrix, aggregation):
    """Each reducer must be the plain numpy reduction over the four steps of the day."""
    init = datetime.datetime(2021, 6, 21)
    states = _states(init, 8, 0)
    spec = _agg_spec([aggregation])

    path = tmp_path / "a.zarr"
    with ForecastZarrWriter(path, init, 8, None, matrix, spec) as w:
        w.start_member(0)
        for state in states:
            w.submit(state)
    got = xr.open_zarr(path).compute()

    reducer = {"daily_mean": np.mean, "daily_min": np.min,
               "daily_max": np.max, "daily_sum": np.sum}[aggregation]
    for day, window in enumerate([slice(0, 4), slice(4, 8)]):
        raw = np.stack(
            [zarr_layout.to_latlon(s["fields"]["2t"], matrix, SMALL_SHAPE) for s in states[window]]
        )
        np.testing.assert_allclose(
            got["2t"].isel(time=0, prediction_timedelta_daily=day).values,
            reducer(raw, axis=0),
            rtol=1e-6,
            err_msg=f"{aggregation} day {day}",
        )


@pytest.mark.parametrize("aggregation", ["daily_mean", "daily_min", "daily_max", "daily_sum"])
def test_every_aggregation_matches_the_batch_path(tmp_path, small_grid, matrix, aggregation):
    init = datetime.datetime(2021, 6, 21)
    states = _states(init, 8, 0)
    spec = _agg_spec([aggregation])

    path = tmp_path / "a.zarr"
    with ForecastZarrWriter(path, init, 8, None, matrix, spec) as w:
        w.start_member(0)
        for state in states:
            w.submit(state)
    got = xr.open_zarr(path).compute()

    expected = zarr_layout.apply_spec(
        zarr_layout.build_dataset(
            [zarr_layout.process_step(s, matrix, ["2t", "msl"]) for s in states], init
        ),
        spec,
    ).compute()
    for name in expected.data_vars:
        np.testing.assert_allclose(
            got[name].values, expected[name].values, rtol=1e-6, atol=1e-6, err_msg=name
        )


def test_min_and_max_of_one_variable_coexist(tmp_path, small_grid, matrix):
    """The motivating case: 2t stored 6-hourly plus its daily min and max, from one source."""
    init = datetime.datetime(2021, 6, 21)
    states = _states(init, 8, 0)
    spec = _agg_spec(["native", "daily_min", "daily_max"])
    assert spec.source_fields == ("msl", "2t"), "2t must only be regridded once per step"

    path = tmp_path / "mm.zarr"
    with ForecastZarrWriter(path, init, 8, None, matrix, spec) as w:
        w.start_member(0)
        for state in states:
            w.submit(state)
    got = xr.open_zarr(path).compute()

    assert got["2t"].dims == ("time", "prediction_timedelta", "lat", "lon")
    assert got["2t_min"].dims == ("time", "prediction_timedelta_daily", "lat", "lon")
    assert got["2t_max"].dims == ("time", "prediction_timedelta_daily", "lat", "lon")
    assert got["2t_min"].attrs["aggregation"] == "daily_min"
    assert got["2t_max"].attrs["aggregation"] == "daily_max"
    assert got["2t"].attrs["units"] == "K"

    # min <= max everywhere, and both bracket the 6-hourly values of their day
    assert (got["2t_min"].values <= got["2t_max"].values).all()
    for day, window in enumerate([slice(0, 4), slice(4, 8)]):
        native = got["2t"].isel(time=0, prediction_timedelta=window).values
        np.testing.assert_allclose(
            got["2t_min"].isel(time=0, prediction_timedelta_daily=day).values,
            native.min(axis=0), rtol=1e-6,
        )
        np.testing.assert_allclose(
            got["2t_max"].isel(time=0, prediction_timedelta_daily=day).values,
            native.max(axis=0), rtol=1e-6,
        )


def test_min_max_propagate_nan_like_the_other_reducers(tmp_path, small_grid, matrix):
    """Fields undefined over sea are NaN in every sample; the daily extreme must stay NaN
    rather than silently reporting an extreme over the valid points."""
    init = datetime.datetime(2021, 6, 21)
    states = _states(init, 4, 0)
    for state in states:
        state["fields"]["2t"][0] = np.nan  # one source point always missing

    spec = _agg_spec(["daily_min"])
    path = tmp_path / "nan.zarr"
    with ForecastZarrWriter(path, init, 4, None, matrix, spec) as w:
        w.start_member(0)
        for state in states:
            w.submit(state)
    got = xr.open_zarr(path).compute()
    assert np.isnan(got["2t"].values).any(), "NaN was silently dropped"


def test_every_configurable_aggregation_has_a_streaming_reducer():
    """A constant added to config.AGGREGATIONS but not wired into the writer must be a
    hard error, not a silent fall-through to whichever branch happens to be last.

    This regression exists because the reducers were once written as binary if/else
    branches, so any unrecognised aggregation quietly computed a maximum.
    """
    from aifs_mps.config import AGGREGATIONS
    from aifs_mps.config import NATIVE
    from aifs_mps.zarr_stream import _REDUCERS
    from aifs_mps.zarr_stream import _reducer

    daily = set(AGGREGATIONS) - {NATIVE}
    assert daily == set(_REDUCERS), "config and the writer disagree on the aggregation set"
    for aggregation in daily:
        assert _reducer(aggregation) is _REDUCERS[aggregation]


def test_unknown_aggregation_raises_rather_than_guessing():
    from aifs_mps.zarr_stream import _reducer

    with pytest.raises(ValueError, match="No streaming reducer for aggregation 'daily_median'"):
        _reducer("daily_median")
