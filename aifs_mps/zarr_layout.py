"""Zarr v3 store layout, built from a :class:`~aifs_mps.config.StoreSpec`.

Ported from the HPC production scripts (``run_AIFS_v2_REF.py``, ``run_AIFS_ENS_v2_REF.py``)
so stores written here stay readable by the same downstream tooling, with two changes: the
N320 -> 0.25 deg output matrix comes from ``support/regrid`` rather than a ``/net/monsoon``
path, and *what* is stored comes from ``config/default.yaml`` instead of being hardcoded.

Dims are ``(time[, number], prediction_timedelta, lat, lon)``: ``time`` is the
initialisation date (length 1), ``prediction_timedelta`` the lead time of each 6-hourly
step, and ``number`` is present only for ensemble models -- deterministic stores omit it
entirely rather than carry a length-1 dimension, keeping them identical to what the
deterministic reference script produced. ``number`` has chunk *and* shard size 1, so each
member occupies its own shard files and members can be written independently without
read-modify-write conflicts; that is what makes :mod:`aifs_mps.zarr_stream`'s per-member
region write safe. Daily aggregations land on a ``prediction_timedelta_daily`` coordinate,
4x smaller on disk than the 6-hourly equivalent.

The *batch* path here (:func:`process_step`, :func:`build_dataset`, :func:`apply_spec`)
builds a whole forecast in memory; it is kept as the oracle the streaming writer is tested
against, not for production use.
"""

from __future__ import annotations

import contextlib
import logging
import warnings

import numpy as np
import xarray as xr
from zarr.codecs import BloscCodec
from zarr.codecs.numcodecs import BitRound
from zarr.errors import ZarrUserWarning

from .config import DAILY_MAX
from .config import DAILY_MEAN
from .config import DAILY_MIN
from .config import DAILY_SUM
from .config import Encoding
from .config import StoreSpec
from .regrid import LATITUDES
from .regrid import LATLON_SHAPE
from .regrid import LONGITUDES
from .regrid import to_latlon

LOG = logging.getLogger(__name__)

__all__ = [
    "MODEL_STEP",
    "STEPS_PER_DAY",
    "apply_spec",
    "build_dataset",
    "complete_day_windows",
    "process_step",
    "quiet_consolidated_metadata_warning",
]

MODEL_STEP = np.timedelta64(6, "h")
STEPS_PER_DAY = 4

# Reducer used by the batch oracle for each daily aggregation. skipna=False so a day with
# any missing sample yields NaN, matching the streaming accumulator's arithmetic NaN
# propagation. (Sea-masked fields are NaN in every sample, so both agree there regardless;
# this only matters for partially missing data.)
_COARSEN_REDUCER = {
    DAILY_MEAN: lambda coarsened: coarsened.mean(skipna=False),
    DAILY_MIN: lambda coarsened: coarsened.min(skipna=False),
    DAILY_MAX: lambda coarsened: coarsened.max(skipna=False),
    DAILY_SUM: lambda coarsened: coarsened.sum(skipna=False),
}

PT_DAILY_DESCRIPTION = (
    "End of the UTC calendar-day aggregation window (00:00 UTC of the following date), "
    "as lead time relative to forecast initialization."
)


@contextlib.contextmanager
def quiet_consolidated_metadata_warning():
    """Suppress zarr's "consolidated metadata is not in the v3 spec" warning.

    We write consolidated metadata deliberately -- the archive stores have it and it is
    what makes opening a store with 25 arrays fast -- so the warning is noise. Scoped via
    ``catch_warnings`` rather than a module-level ``filterwarnings``, so importing this
    module does not quietly mutate its importer's warning filters.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*Consolidated metadata.*", category=ZarrUserWarning
        )
        yield


def v3_encoding(ds: xr.Dataset, encoding: Encoding) -> dict:
    """Per-variable zarr encoding: chunks, shards, compressor and optional BitRound."""
    # "bitshuffle" as a string: BloscShuffle.bitshuffle (used by the reference scripts) is
    # deprecated in zarr 3.3 and warns on every call. Same codec, same bytes on disk.
    compressors = (BloscCodec(**encoding.compressor),)
    out = {}
    for name, array in ds.data_vars.items():
        if not array.dims or not all(d in encoding.chunks for d in array.dims):
            continue
        enc = {
            "chunks": tuple(encoding.chunks[d] for d in array.dims),
            "shards": tuple(encoding.shards[d] for d in array.dims),
            "compressors": compressors,
        }
        keepbits = encoding.keepbits_for(name)
        if keepbits is not None:
            enc["filters"] = (BitRound(keepbits=keepbits),)
        out[name] = enc
    return out


def complete_day_windows(init_date, n_steps: int) -> list[tuple[int, int]]:
    """Step-index windows ``[start, stop)`` covering each *complete* UTC calendar day.

    A step valid at ``t`` samples the interval ``(t-6h, t]``, which lies in the calendar
    date containing ``t-6h`` -- so a 00 UTC step belongs to the previous date. Dates with
    fewer than four steps (the forecast edges) are dropped.
    """
    init = np.datetime64(init_date).astype("datetime64[ns]")
    valid = init + (np.arange(n_steps) + 1) * MODEL_STEP
    step_date = (valid - MODEL_STEP).astype("datetime64[D]")

    windows = []
    for date in np.unique(step_date):
        idx = np.flatnonzero(step_date == date)
        if idx.size != STEPS_PER_DAY:
            continue
        if idx[-1] - idx[0] != STEPS_PER_DAY - 1:
            raise ValueError(f"Steps for {date} are not contiguous; got {idx}")
        windows.append((int(idx[0]), int(idx[-1]) + 1))
    return windows


# -- batch path (test oracle) ------------------------------------------------------


def process_step(state: dict, matrix, sources) -> xr.Dataset:
    """Regrid one forecast state's source fields from N320 to the 0.25 deg lat/lon grid."""
    data_vars = {}
    for field in sources:
        values = to_latlon(np.asarray(state["fields"][field]), matrix, shape=LATLON_SHAPE)
        data_vars[field] = (["lat", "lon"], values.astype(np.float32))

    ds = xr.Dataset(data_vars, coords={"lat": LATITUDES, "lon": LONGITUDES})
    ds = ds.expand_dims("step")
    ds["step"] = [state["date"]]
    return ds


def build_dataset(step_datasets: list[xr.Dataset], date, member: int | None = None) -> xr.Dataset:
    """Assemble per-step datasets into the archive coordinate layout.

    Output dims are ``(time[, number], prediction_timedelta, lat, lon)``; pass ``member``
    to include the ensemble dimension. Variables are still named after their *sources* at
    this point; :func:`apply_spec` turns them into the configured outputs.
    """
    full_ds = xr.concat(step_datasets, dim="step")
    full_ds = full_ds.rename({"step": "prediction_timedelta"})
    full_ds["prediction_timedelta"] = (
        full_ds["prediction_timedelta"].values - np.datetime64(date).astype("datetime64[ns]")
    ).astype("timedelta64[ns]")
    if member is not None:
        full_ds = full_ds.expand_dims("number")
        full_ds["number"] = [int(member)]
    full_ds = full_ds.expand_dims("time")
    full_ds["time"] = [np.datetime64(date).astype("datetime64[ns]")]
    return full_ds


def apply_spec(ds: xr.Dataset, spec: StoreSpec) -> xr.Dataset:
    """Turn a source-named 6-hourly dataset into the configured outputs.

    Native outputs are copied through; each daily output is the corresponding reduction
    over the four steps of every complete UTC calendar day. Incomplete days at the forecast
    edges are dropped, and each day is labelled on ``prediction_timedelta_daily`` by its
    window end (the lead time of its last step).
    """
    steps = ds["prediction_timedelta"].values
    expected = (np.arange(steps.size) + 1) * MODEL_STEP
    if not np.array_equal(steps, expected):
        raise ValueError(
            "Daily aggregation requires contiguous 6-hourly steps starting at +6h; "
            f"got {steps.size} steps [{steps[0]} .. {steps[-1]}]."
        )

    parts = [ds[[o.source for o in spec.native]].rename({o.source: o.name for o in spec.native})]

    if spec.daily:
        windows = complete_day_windows(ds["time"].values[0], steps.size)
        if not windows:
            raise ValueError(
                "Daily aggregation found no complete UTC calendar day in steps "
                f"[{steps[0]} .. {steps[-1]}]. Increase the lead time or disable "
                "daily aggregates."
            )
        first, last = windows[0][0], windows[-1][1]
        if first or last != steps.size:
            LOG.info(
                "Daily aggregation dropping %d edge steps belonging to incomplete days.",
                steps.size - (last - first),
            )
        days = ds.isel(prediction_timedelta=slice(first, last))
        rename = {"prediction_timedelta": "prediction_timedelta_daily"}

        for output in spec.daily:
            coarsened = days[[output.source]].coarsen(
                prediction_timedelta=STEPS_PER_DAY, coord_func={"prediction_timedelta": "max"}
            )
            reduced = _COARSEN_REDUCER[output.aggregation](coarsened)
            parts.append(reduced.rename({**rename, output.source: output.name}))

    out = xr.merge(parts, combine_attrs="override")

    for output in spec.outputs:
        out[output.name].attrs.update(output.attrs)
    if spec.daily:
        out["prediction_timedelta_daily"].attrs["description"] = PT_DAILY_DESCRIPTION

    return out.chunk({dim: spec.encoding.shards[dim] for dim in out.dims})
